import argparse
import json
import os

import numpy as np
import torch

from utils.low_snr_baseline import (
    BASELINE_SPECS,
    evaluate,
    load_checkpoint,
    make_loader,
    run_root,
    set_reproducible_seed,
    write_json,
)


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate supervised baselines with fixed-seed AWGN."
    )
    parser.add_argument("--baseline", choices=sorted(BASELINE_SPECS), required=True)
    parser.add_argument("--dataset", "-d", default="manytx")
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument("--train_seed", type=int, default=2024)
    parser.add_argument("--eval_seed", type=int, default=2024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--snr_levels", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 15, 20]
    )
    return parser


def main(args):
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = run_root(args.baseline, args.dataset, args.train_seed)
    model, metadata = load_checkpoint(
        os.path.join(root, f"{args.checkpoint}.pth"), device
    )
    loader = make_loader(
        args.dataset, "test", args.batch_size, False, args.num_workers
    )

    set_reproducible_seed(args.eval_seed)
    clean = evaluate(model, loader, device)
    print(
        f"Clean: Acc={clean['acc']:.2f}%, Macro-F1={clean['macro_f1']:.2f}%",
        flush=True,
    )

    repeated = []
    for repeat in range(args.repeats):
        current = []
        for index, snr in enumerate(args.snr_levels):
            seed = args.eval_seed + repeat * len(args.snr_levels) + index
            set_reproducible_seed(seed)
            metrics = evaluate(model, loader, device, snr=snr)
            metrics.update({"snr": float(snr), "seed": seed})
            current.append(metrics)
            print(
                f"Repeat {repeat + 1}/{args.repeats}, SNR {snr:g} dB: "
                f"Acc={metrics['acc']:.2f}%, Macro-F1={metrics['macro_f1']:.2f}%",
                flush=True,
            )
        repeated.append(current)

    by_snr = []
    for index, snr in enumerate(args.snr_levels):
        entries = [repeat[index] for repeat in repeated]
        by_snr.append({
            "snr": float(snr),
            "acc": summary([entry["acc"] for entry in entries]),
            "macro_f1": summary([entry["macro_f1"] for entry in entries]),
        })
    low = [entry for entry in by_snr if entry["snr"] <= 0]
    result = {
        "metadata": metadata,
        "clean": clean,
        "eval_seed": args.eval_seed,
        "repeats": args.repeats,
        "by_snr": by_snr,
        "low_snr_mean": {
            "acc": float(np.mean([entry["acc"]["mean"] for entry in low])),
            "macro_f1": float(
                np.mean([entry["macro_f1"]["mean"] for entry in low])
            ),
        },
        "all_snr_mean": {
            "acc": float(np.mean([entry["acc"]["mean"] for entry in by_snr])),
            "macro_f1": float(
                np.mean([entry["macro_f1"]["mean"] for entry in by_snr])
            ),
        },
    }
    print("===== Fixed-seed summary =====")
    for entry in by_snr:
        print(
            f"SNR {entry['snr']:g} dB: "
            f"Acc={entry['acc']['mean']:.2f}% +/- {entry['acc']['std']:.2f}%, "
            f"Macro-F1={entry['macro_f1']['mean']:.2f}% +/- "
            f"{entry['macro_f1']['std']:.2f}%"
        )
    print(
        f"Low-SNR mean: Acc={result['low_snr_mean']['acc']:.2f}%, "
        f"Macro-F1={result['low_snr_mean']['macro_f1']:.2f}%"
    )
    print(
        f"All-SNR mean: Acc={result['all_snr_mean']['acc']:.2f}%, "
        f"Macro-F1={result['all_snr_mean']['macro_f1']:.2f}%"
    )
    result_path = os.path.join(root, f"evaluation_{args.checkpoint}.json")
    write_json(result_path, result)
    print(json.dumps({"result_path": result_path}, ensure_ascii=False))


if __name__ == "__main__":
    main(build_parser().parse_args())
