from evaluate_robust_sei import build_parser
from utils.robust_eval import evaluate_loader, load_eval_loader, load_robust_models

import torch


if __name__ == "__main__":
    parser = build_parser()
    parser.description = "Evaluate cross-day robust SEI on the prepared test split."
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, lfdb, classifier, _ = load_robust_models(args, device)
    loader, _ = load_eval_loader(args, split="test", snr=args.snr)
    metrics = evaluate_loader(encoder, lfdb, classifier, loader, device, desc="Evaluating cross-Day test split")
    print(f"Cross-Day Acc: {metrics['acc']:.2f}%")
    print(f"Cross-Day Macro-F1: {metrics['macro_f1']:.2f}%")
