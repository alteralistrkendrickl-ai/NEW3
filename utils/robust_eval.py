import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.lfdb import LightweightLFDB
from utils.config import PROJECT_ROOT, dataset_path_dict, is_joint_interference_method, model_path_dict
from utils.get_dataset import (
    add_noise,
    default_normalize_fn,
    entire_max_min,
    load_data,
    power_normalize_fn,
    sample_max_min,
)
from utils.utils import create_model, load_encoder_weights


def normalizer(name):
    if name == "sample":
        return sample_max_min
    if name == "dataset":
        return entire_max_min
    if name == "power":
        return power_normalize_fn
    return default_normalize_fn


def robust_run_root(args):
    dataset_base = dataset_path_dict[args.dataset]["name"]
    exp_name = f"{args.encoder}_{dataset_base}_{args.input_type}_{args.normalize_fn}Norm_{args.method_name}"
    exp_type = f"Pretext_{args.method_name}_random_rot"
    return os.path.join(PROJECT_ROOT, "runs", exp_type, exp_name, *([args.pretrain_date] if args.pretrain_date else []))


def dataset_root(args):
    platform = "windows" if os.name == "nt" else "linux"
    root = dataset_path_dict[args.dataset][platform]
    return root if args.input_type == "iq" else os.path.join(root, args.input_type)


def make_encoder(args, device):
    encoder_kwargs = {
        "feature_dim": args.feature_dim,
        "dtype": args.input_type,
    }
    if "TSLA" in args.encoder:
        encoder_kwargs.update({
            "seq_len": args.TSLA_len,
            "patch_size": args.TSLA_patch,
            "num_channels": args.TSLA_channels,
            "emb_dim": args.TSLA_emb,
            "depth": args.TSLA_depth,
            "dropout_rate": args.TSLA_dropout,
        })
    encoder = create_model(model_path_dict[args.encoder], **encoder_kwargs).to(device)
    return encoder


def load_robust_models(args, device):
    run_root = robust_run_root(args)
    encoder = make_encoder(args, device)
    load_encoder_weights(encoder, os.path.join(run_root, f"{args.checkpoint}_encoder.pth"), device)

    lfdb = LightweightLFDB(
        feat_dim=args.feature_dim,
        num_classes=dataset_path_dict[args.dataset]["pt_class"],
        snr_classes=len(args.snr_levels) if is_joint_interference_method(args.method_name) else 3,
        fading_classes=3,
    ).to(device)
    lfdb.load_state_dict(torch.load(os.path.join(run_root, f"{args.checkpoint}_lfdb.pth"), map_location=device))

    classifier = create_model(
        model_path_dict["LinearClassifier"],
        in_dim=args.feature_dim,
        num_classes=dataset_path_dict[args.dataset]["pt_class"],
    ).to(device)
    classifier_path = os.path.join(run_root, f"{args.checkpoint}_id_classifier.pth")
    if not os.path.isfile(classifier_path):
        raise FileNotFoundError(
            f"Device classifier not found: {classifier_path}. "
            "Run train_robust_sei.py with the updated code first."
        )
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    encoder.eval()
    lfdb.eval()
    classifier.eval()
    return encoder, lfdb, classifier, run_root


def load_eval_loader(args, split="test", snr=None):
    num_classes = dataset_path_dict[args.dataset]["pt_class"]
    x, y = load_data(dataset_root(args), num_classes, split)
    x = normalizer(args.normalize_fn)(x)
    if snr is not None:
        x = add_noise(x, snr=snr)
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False), y


def evaluate_loader(encoder, lfdb, classifier, loader, device, desc="Evaluating"):
    predictions = []
    labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc):
            inputs = inputs.to(device)
            features = encoder(inputs)
            fingerprint, _, _ = lfdb(features)
            logits = classifier(fingerprint)
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
            labels.append(targets.numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return {
        "acc": accuracy_score(labels, predictions) * 100.0,
        "macro_f1": f1_score(labels, predictions, average="macro") * 100.0,
    }
