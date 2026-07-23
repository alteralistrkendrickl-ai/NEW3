import argparse
import json

import torch

from utils.robust_eval import evaluate_loader, load_eval_loader, load_robust_models


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate robust SEI by E -> LFDB fingerprint -> device classifier.")
    parser.add_argument("--dataset", "-d", default="manytx")
    parser.add_argument("--encoder", "-e", default="CVTSLANet")
    parser.add_argument("--input_type", "-t", default="iq")
    parser.add_argument("--normalize_fn", "-n", default="power")
    parser.add_argument("--method_name", default="RobustSEI")
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument("--pretrain_date", default="")
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--snr_levels", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 15, 20])
    parser.add_argument("--TSLA_len", type=int, default=256)
    parser.add_argument("--TSLA_patch", type=int, default=16)
    parser.add_argument("--TSLA_channels", type=int, default=2)
    parser.add_argument("--TSLA_emb", type=int, default=128)
    parser.add_argument("--TSLA_depth", type=int, default=3)
    parser.add_argument("--TSLA_dropout", type=float, default=0.3)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, lfdb, classifier, run_root = load_robust_models(args, device)
    loader, _ = load_eval_loader(args, split=args.split, snr=args.snr)
    metrics = evaluate_loader(encoder, lfdb, classifier, loader, device, desc=f"Evaluating {args.split}")
    metrics["run_root"] = run_root
    metrics["snr"] = args.snr
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
