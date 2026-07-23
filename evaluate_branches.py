import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from evaluate_robust_sei import build_parser
from utils.robust_eval import load_eval_loader, load_robust_models


def collect_branch_features(encoder, lfdb, loader, device):
    raw_features = []
    fingerprint_features = []
    interference_features = []
    labels = []
    encoder.eval()
    lfdb.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Collecting branch features"):
            inputs = inputs.to(device)
            features = encoder(inputs)
            outputs = lfdb(features, return_all=True)
            raw_features.append(features.cpu().numpy())
            fingerprint_features.append(outputs["fingerprint"].cpu().numpy())
            interference_features.append(outputs["nuisance"].cpu().numpy())
            labels.append(targets.numpy())
    return {
        "h": np.concatenate(raw_features),
        "z_fp": np.concatenate(fingerprint_features),
        "z_int": np.concatenate(interference_features),
        "y": np.concatenate(labels),
    }


def fit_and_eval(train_x, train_y, test_x, test_y):
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train_x, train_y)
    pred = classifier.predict(test_x)
    return {
        "acc": accuracy_score(test_y, pred) * 100.0,
        "macro_f1": f1_score(test_y, pred, average="macro") * 100.0,
    }


if __name__ == "__main__":
    parser = build_parser()
    parser.description = "Evaluate device identity leakage in h, z_fp, and z_int branches."
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, lfdb, _, run_root = load_robust_models(args, device)

    train_loader, _ = load_eval_loader(args, split="train", snr=args.snr)
    test_loader, _ = load_eval_loader(args, split="test", snr=args.snr)
    train = collect_branch_features(encoder, lfdb, train_loader, device)
    test = collect_branch_features(encoder, lfdb, test_loader, device)

    results = {}
    for branch in ("h", "z_fp", "z_int"):
        results[branch] = fit_and_eval(train[branch], train["y"], test[branch], test["y"])
        print(
            f"{branch}: Acc={results[branch]['acc']:.2f}%, "
            f"Macro-F1={results[branch]['macro_f1']:.2f}%"
        )
    print(json.dumps({"run_root": run_root, "results": results}, indent=2, ensure_ascii=False))
