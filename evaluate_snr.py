import json

import numpy as np


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def summarize_results(repeated_results, snr_levels):
    by_snr = []
    for snr in snr_levels:
        matches = [
            result
            for repeat in repeated_results
            for result in repeat
            if float(result["snr"]) == float(snr)
        ]
        by_snr.append({
            "snr": float(snr),
            "acc": _summary([result["acc"] for result in matches]),
            "macro_f1": _summary([result["macro_f1"] for result in matches]),
        })

    low_snr = [item for item in by_snr if item["snr"] <= 0.0]
    return {
        "by_snr": by_snr,
        "low_snr_mean": {
            "acc": float(np.mean([item["acc"]["mean"] for item in low_snr])),
            "macro_f1": float(
                np.mean([item["macro_f1"]["mean"] for item in low_snr])
            ),
            "levels": [item["snr"] for item in low_snr],
        },
        "all_snr_mean": {
            "acc": float(np.mean([item["acc"]["mean"] for item in by_snr])),
            "macro_f1": float(
                np.mean([item["macro_f1"]["mean"] for item in by_snr])
            ),
            "levels": [item["snr"] for item in by_snr],
        },
    }


if __name__ == "__main__":
    import torch

    from evaluate_robust_sei import build_parser
    from utils.robust_eval import (
        evaluate_loader,
        load_eval_loader,
        load_robust_models,
    )

    parser = build_parser()
    parser.description = "Evaluate robust SEI across fixed-seed AWGN SNR levels."
    parser.set_defaults(snr=None)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, lfdb, classifier, run_root = load_robust_models(args, device)

    repeated_results = []
    for repeat in range(args.repeats):
        if args.repeats > 1:
            print(f"===== Repeat {repeat + 1}/{args.repeats} =====")
        repeat_results = []
        for snr_index, snr in enumerate(args.snr_levels):
            seed = args.eval_seed + repeat * len(args.snr_levels) + snr_index
            np.random.seed(seed)
            torch.manual_seed(seed)
            loader, _ = load_eval_loader(args, split=args.split, snr=snr)
            metrics = evaluate_loader(
                encoder,
                lfdb,
                classifier,
                loader,
                device,
                desc=f"Evaluating {snr:g} dB",
            )
            metrics["snr"] = float(snr)
            metrics["seed"] = int(seed)
            repeat_results.append(metrics)
            print(
                f"SNR {snr:g} dB: Acc={metrics['acc']:.2f}%, "
                f"Macro-F1={metrics['macro_f1']:.2f}%"
            )
        repeated_results.append(repeat_results)

    summary = summarize_results(repeated_results, args.snr_levels)
    print("===== Fixed-seed summary =====")
    for item in summary["by_snr"]:
        print(
            f"SNR {item['snr']:g} dB: "
            f"Acc={item['acc']['mean']:.2f}% +/- {item['acc']['std']:.2f}%, "
            f"Macro-F1={item['macro_f1']['mean']:.2f}% +/- "
            f"{item['macro_f1']['std']:.2f}%"
        )
    print(
        "Low-SNR mean (-10/-5/0 dB): "
        f"Acc={summary['low_snr_mean']['acc']:.2f}%, "
        f"Macro-F1={summary['low_snr_mean']['macro_f1']:.2f}%"
    )
    print(
        "All-SNR mean: "
        f"Acc={summary['all_snr_mean']['acc']:.2f}%, "
        f"Macro-F1={summary['all_snr_mean']['macro_f1']:.2f}%"
    )
    print(json.dumps({
        "run_root": run_root,
        "eval_seed": args.eval_seed,
        "repeats": args.repeats,
        "results": repeated_results,
        "summary": summary,
    }, indent=2, ensure_ascii=False))
