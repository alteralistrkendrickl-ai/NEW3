import argparse
import json

import torch

from evaluate_robust_sei import build_parser
from utils.robust_eval import evaluate_loader, load_eval_loader, load_robust_models


if __name__ == "__main__":
    parser = build_parser()
    parser.description = "Evaluate robust SEI across AWGN SNR levels."
    parser.set_defaults(snr=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, lfdb, classifier, run_root = load_robust_models(args, device)

    results = []
    for snr in args.snr_levels:
        loader, _ = load_eval_loader(args, split=args.split, snr=snr)
        metrics = evaluate_loader(encoder, lfdb, classifier, loader, device, desc=f"Evaluating {snr:g} dB")
        metrics["snr"] = snr
        results.append(metrics)
        print(f"SNR {snr:g} dB: Acc={metrics['acc']:.2f}%, Macro-F1={metrics['macro_f1']:.2f}%")

    print(json.dumps({"run_root": run_root, "results": results}, indent=2, ensure_ascii=False))
