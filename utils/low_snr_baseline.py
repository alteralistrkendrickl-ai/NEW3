import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.config import PROJECT_ROOT, dataset_path_dict, model_path_dict
from utils.utils import create_model


BASELINE_SPECS = {
    "CVTSLANet-Supervised": {
        "encoder": "CVTSLANet",
        "augmentation": "clean",
        "feature_dim": 1024,
        "map_channels": 128,
        "lr": 1e-3,
    },
    "MSFTFNet-Supervised": {
        "encoder": "MSFTFNet",
        "augmentation": "clean",
        "feature_dim": 1024,
        "map_channels": 128,
        "lr": 1e-3,
    },
    "MSFTFNet-OnlineAWGN": {
        "encoder": "MSFTFNet",
        "augmentation": "online_awgn",
        "feature_dim": 1024,
        "map_channels": 128,
        "lr": 1e-3,
    },
    "MSFTFNet-OnlineAWGN-Paired": {
        "encoder": "MSFTFNet",
        "augmentation": "paired_online_awgn",
        "feature_dim": 1024,
        "map_channels": 128,
        "lr": 1e-3,
    },
    "WiSigCNN-OnlineAWGN": {
        "encoder": "WiSigCNN",
        "augmentation": "paired_online_awgn",
        "feature_dim": 80,
        "lr": 5e-4,
    },
}


class PooledEncoderClassifier(nn.Module):
    """Use local feature maps directly instead of an encoder projection head."""

    def __init__(self, encoder, num_classes, map_channels=None):
        super().__init__()
        self.encoder = encoder
        self.use_feature_map = map_channels is not None
        classifier_dim = map_channels * 2 if self.use_feature_map else encoder.feature_dim
        self.classifier = nn.Linear(classifier_dim, num_classes)

    def forward(self, inputs):
        if self.use_feature_map:
            feature_map = self.encoder.forward_map(inputs)
            if feature_map.ndim != 3:
                raise ValueError(
                    "forward_map must return [batch, channels, positions]."
                )
            features = torch.cat(
                [
                    feature_map.mean(dim=-1),
                    feature_map.max(dim=-1).values,
                ],
                dim=1,
            )
        else:
            features = self.encoder(inputs)
        return self.classifier(features)


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_root(dataset):
    platform = "windows" if os.name == "nt" else "linux"
    return os.path.expanduser(dataset_path_dict[dataset][platform])


def split_paths(dataset, split, num_classes):
    root = dataset_root(dataset)
    x_path = os.path.join(root, f"X_{split}_{num_classes}Class.npy")
    y_path = os.path.join(root, f"Y_{split}_{num_classes}Class.npy")
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(f"Missing dataset split: {x_path} / {y_path}")
    return x_path, y_path


class IQClassificationDataset(Dataset):
    def __init__(self, x_path, y_path, signal_length=256, indices=None):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.signal_length = signal_length
        self.indices = (
            np.arange(len(self.y)) if indices is None else np.asarray(indices)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        sample = np.asarray(self.x[index], dtype=np.float32)
        if sample.shape[0] != 2 and sample.shape[-1] == 2:
            sample = sample.T
        sample = np.array(sample[:, : self.signal_length], copy=True)
        if sample.shape != (2, self.signal_length):
            raise ValueError(
                f"Expected IQ sample [2, {self.signal_length}], got {sample.shape}."
            )
        peak_power = np.max(sample[0] ** 2 + sample[1] ** 2)
        sample /= max(float(np.sqrt(peak_power)), 1e-12)
        return torch.from_numpy(sample), torch.tensor(int(self.y[index]))


def make_loader(dataset, split, batch_size, shuffle, num_workers=0):
    num_classes = dataset_path_dict[dataset]["pt_class"]
    x_path, y_path = split_paths(dataset, split, num_classes)
    ds = IQClassificationDataset(x_path, y_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_train_val_loaders(
    dataset, batch_size, seed, num_workers=0, val_ratio=0.2
):
    num_classes = dataset_path_dict[dataset]["pt_class"]
    train_x, train_y = split_paths(dataset, "train", num_classes)
    try:
        val_x, val_y = split_paths(dataset, "val", num_classes)
    except FileNotFoundError:
        labels = np.load(train_y, mmap_mode="r")
        train_indices, val_indices = train_test_split(
            np.arange(len(labels)),
            test_size=val_ratio,
            random_state=seed,
            stratify=np.asarray(labels),
        )
        train_dataset = IQClassificationDataset(
            train_x, train_y, indices=train_indices
        )
        val_dataset = IQClassificationDataset(
            train_x, train_y, indices=val_indices
        )
    else:
        train_dataset = IQClassificationDataset(train_x, train_y)
        val_dataset = IQClassificationDataset(val_x, val_y)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
    )


def add_awgn_torch(inputs, snr_db):
    if not torch.is_tensor(snr_db):
        snr_db = torch.full(
            (inputs.shape[0], 1, 1),
            float(snr_db),
            device=inputs.device,
            dtype=inputs.dtype,
        )
    else:
        snr_db = snr_db.to(device=inputs.device, dtype=inputs.dtype)
        if snr_db.ndim == 1:
            snr_db = snr_db[:, None, None]
    signal_power = inputs.square().mean(dim=-1, keepdim=True)
    target_noise_power = signal_power / torch.pow(10.0, snr_db / 10.0)
    noise = torch.randn_like(inputs)
    current_noise_power = noise.square().mean(dim=-1, keepdim=True)
    return inputs + noise * torch.sqrt(
        target_noise_power / current_noise_power.clamp_min(1e-12)
    )


def build_model(baseline, num_classes, device):
    spec = BASELINE_SPECS[baseline]
    encoder_name = spec["encoder"]
    if encoder_name == "WiSigCNN":
        model_root = os.path.join(PROJECT_ROOT, "models", "WiSigCNNFeature.py")
    else:
        model_root = model_path_dict[encoder_name]
    encoder = create_model(
        model_root,
        feature_dim=spec["feature_dim"],
        dtype="iq",
        seq_len=256,
        patch_size=16,
        num_channels=2,
        emb_dim=128,
        depth=3,
        dropout_rate=0.3 if encoder_name != "WiSigCNN" else 0.5,
    )
    return PooledEncoderClassifier(
        encoder,
        num_classes,
        map_channels=spec.get("map_channels"),
    ).to(device)


def run_root(baseline, dataset, seed):
    return os.path.join(
        PROJECT_ROOT,
        "runs",
        "LowSNR_Baselines",
        baseline,
        dataset,
        f"seed_{seed}",
    )


def save_checkpoint(path, model, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    metadata = checkpoint["metadata"]
    model = build_model(
        metadata["baseline"], metadata["num_classes"], device
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, metadata


def evaluate(model, loader, device, snr=None):
    predictions = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            if snr is not None:
                inputs = add_awgn_torch(inputs, float(snr))
            logits = model(inputs)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(targets.numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return {
        "acc": accuracy_score(labels, predictions) * 100.0,
        "macro_f1": f1_score(labels, predictions, average="macro") * 100.0,
    }


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
