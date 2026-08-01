import argparse
import json
import os

import numpy as np


SPLITS = ("train", "val", "test")


def _load(root, prefix, split, class_count, mmap_mode="r"):
    path = os.path.expanduser(
        os.path.join(root, f"{prefix}_{split}_{class_count}Class.npy")
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode=mmap_mode)


def verify_protocol(root, class_count):
    root = os.path.expanduser(root)
    manifest_path = os.path.join(root, f"protocol_{class_count}Class.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)

    protocol = manifest["protocol"]
    domain_prefix = "RX" if protocol == "cross_rx" else "DAY" if protocol == "cross_day" else None
    report = {"root": os.path.abspath(root), "protocol": protocol, "splits": {}}
    domains = {}

    for split in SPLITS:
        x = _load(root, "X", split, class_count)
        y = _load(root, "Y", split, class_count)
        rx = _load(root, "RX", split, class_count)
        day = _load(root, "DAY", split, class_count)
        lengths = {len(x), len(y), len(rx), len(day)}
        if len(lengths) != 1:
            raise ValueError(f"Mismatched array lengths in {split}: {lengths}")
        labels = set(np.unique(y).astype(int).tolist())
        expected_labels = set(range(class_count))
        if labels != expected_labels:
            missing = sorted(expected_labels - labels)
            extra = sorted(labels - expected_labels)
            raise ValueError(
                f"Invalid labels in {split}; missing={missing}, extra={extra}"
            )
        report["splits"][split] = {
            "samples": int(len(y)),
            "rx": sorted(np.unique(rx).astype(int).tolist()),
            "day": sorted(np.unique(day).astype(int).tolist()),
        }
        if domain_prefix is not None:
            domain_array = rx if domain_prefix == "RX" else day
            domains[split] = set(np.unique(domain_array).astype(int).tolist())

    if domain_prefix is not None:
        overlaps = {
            "train_val": sorted(domains["train"] & domains["val"]),
            "train_test": sorted(domains["train"] & domains["test"]),
            "val_test": sorted(domains["val"] & domains["test"]),
        }
        if any(overlaps.values()):
            raise ValueError(
                f"{protocol} domain leakage detected: {overlaps}"
            )
        report["domain_leakage"] = False
        report["held_out_domains"] = manifest.get("held_out_domains", {})

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Verify fixed ManyTx IID/cross-domain data protocols."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--classes", type=int, default=90)
    args = parser.parse_args()
    report = verify_protocol(args.root, args.classes)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Protocol verification: PASS")


if __name__ == "__main__":
    main()
