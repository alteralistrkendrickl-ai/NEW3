import argparse
import os
import pickle

import numpy as np


def _collect_class_samples(tx_data, equalized_index):
    arrays = []
    for rx_data in tx_data:
        for date_data in rx_data:
            block = date_data[equalized_index]
            if len(block):
                arrays.append(np.asarray(block, dtype=np.float32))
    if not arrays:
        return np.empty((0, 2, 256), dtype=np.float32)
    samples = np.concatenate(arrays, axis=0)
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError(f"Unexpected sample shape: {samples.shape}")
    return samples.transpose(0, 2, 1).astype(np.float32, copy=False)


def _split_indices(count, test_ratio, rng):
    indices = np.arange(count)
    rng.shuffle(indices)
    test_count = max(1, int(round(count * test_ratio)))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return train_indices, test_indices


def _write_subset(class_samples, class_count, output_dir, seed, test_ratio):
    rng = np.random.default_rng(seed)
    x_train_parts = []
    y_train_parts = []
    x_test_parts = []
    y_test_parts = []

    for label in range(class_count):
        samples = class_samples[label]
        train_indices, test_indices = _split_indices(len(samples), test_ratio, rng)
        x_train_parts.append(samples[train_indices])
        y_train_parts.append(np.full(len(train_indices), label, dtype=np.int64))
        x_test_parts.append(samples[test_indices])
        y_test_parts.append(np.full(len(test_indices), label, dtype=np.int64))

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    x_test = np.concatenate(x_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"X_train_{class_count}Class.npy"), x_train)
    np.save(os.path.join(output_dir, f"Y_train_{class_count}Class.npy"), y_train)
    np.save(os.path.join(output_dir, f"X_test_{class_count}Class.npy"), x_test)
    np.save(os.path.join(output_dir, f"Y_test_{class_count}Class.npy"), y_test)
    print(
        f"{output_dir}: {class_count}Class "
        f"train={x_train.shape}/{y_train.shape} test={x_test.shape}/{y_test.shape}",
        flush=True,
    )


def convert_pkl(path, output_dir, class_counts, equalized_index=0, seed=2024, test_ratio=0.2):
    print(f"Loading {path}", flush=True)
    with open(path, "rb") as file:
        data = pickle.load(file)

    max_class_count = max(class_counts)
    if max_class_count > len(data["data"]):
        raise ValueError(
            f"Requested {max_class_count} classes, but dataset contains {len(data['data'])}."
        )

    print(f"Collecting first {max_class_count} transmitters", flush=True)
    class_samples = [
        _collect_class_samples(data["data"][index], equalized_index)
        for index in range(max_class_count)
    ]
    empty = [index for index, samples in enumerate(class_samples) if len(samples) == 0]
    if empty:
        raise ValueError(f"Empty transmitter classes found: {empty[:20]}")

    for class_count in class_counts:
        _write_subset(class_samples, class_count, output_dir, seed, test_ratio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", default=r"E:\Single\SingleDay.pkl")
    parser.add_argument("--manytx", default=r"E:\ManyTx\ManyTx.pkl")
    parser.add_argument("--output-root", default="Datasets")
    parser.add_argument("--equalized-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    convert_pkl(
        args.single,
        os.path.join(args.output_root, "Single"),
        class_counts=[28, 20, 10],
        equalized_index=args.equalized_index,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )
    convert_pkl(
        args.manytx,
        os.path.join(args.output_root, "ManyTx"),
        class_counts=[90, 30, 20, 10],
        equalized_index=args.equalized_index,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
