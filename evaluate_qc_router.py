import json

import numpy as np
import torch

from evaluate_robust_sei import build_parser
from utils.robust_eval import load_eval_loader, load_robust_models


ROUTE_NAMES = ("time_raw", "time_smooth3", "time_smooth7", "time_smooth11", "freq")


def _rank_correlation(x, y):
    x_rank = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    y_rank = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


@torch.no_grad()
def collect_diagnostics(encoder, loader, device):
    qualities = []
    weights = []
    entropies = []
    for inputs, _ in loader:
        stages = encoder.forward_stages(inputs.to(device))
        qualities.append(stages["quality"].cpu().numpy())
        weights.append(stages["route_weights"].cpu().numpy())
        entropies.append(stages["route_entropy"].cpu().numpy())
    quality = np.concatenate(qualities)
    route_weights = np.concatenate(weights)
    entropy = np.concatenate(entropies)
    return {
        "quality_mean": float(quality.mean()),
        "quality_std": float(quality.std()),
        "route_entropy_mean": float(entropy.mean()),
        "route_weights": {
            name: {
                "mean": float(route_weights[:, index].mean()),
                "std": float(route_weights[:, index].std()),
            }
            for index, name in enumerate(ROUTE_NAMES)
        },
    }


if __name__ == "__main__":
    parser = build_parser()
    parser.description = "Inspect QCRouter quality and routing across SNR levels."
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _, _, run_root = load_robust_models(args, device)
    if not hasattr(encoder, "qc_router"):
        parser.error("evaluate_qc_router.py requires MSFTFNet-QCRouter")

    diagnostics = []
    for index, snr in enumerate(args.snr_levels):
        seed = args.eval_seed + index
        np.random.seed(seed)
        torch.manual_seed(seed)
        loader, _ = load_eval_loader(args, split=args.split, snr=snr)
        result = collect_diagnostics(encoder, loader, device)
        result["snr"] = float(snr)
        result["seed"] = int(seed)
        diagnostics.append(result)
        weights = ", ".join(
            f"{name}={value['mean']:.3f}"
            for name, value in result["route_weights"].items()
        )
        print(
            f"SNR {snr:g} dB: quality={result['quality_mean']:.4f}, "
            f"entropy={result['route_entropy_mean']:.4f}, {weights}"
        )

    correlation = _rank_correlation(
        [item["snr"] for item in diagnostics],
        [item["quality_mean"] for item in diagnostics],
    )
    print(f"Quality-SNR rank correlation: {correlation:.4f}")
    print(json.dumps({
        "run_root": run_root,
        "quality_snr_rank_correlation": correlation,
        "diagnostics": diagnostics,
    }, indent=2, ensure_ascii=False))
