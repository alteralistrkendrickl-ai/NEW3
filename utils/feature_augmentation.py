import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.get_dataset import (
    default_normalize_fn,
    entire_max_min,
    load_data,
    power_normalize_fn,
    sample_max_min,
)


def _normalizer(name):
    if name == "sample":
        return sample_max_min
    if name == "dataset":
        return entire_max_min
    if name == "power":
        return power_normalize_fn
    return default_normalize_fn


def extract_encoder_features(encoder, x, y, device, batch_size=128, desc="features"):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    labels = []
    encoder.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc):
            inputs = inputs.to(device)
            features.append(encoder(inputs).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def class_statistics(features, labels):
    stats = {}
    for label in sorted(np.unique(labels)):
        class_features = features[labels == label]
        mean = class_features.mean(axis=0)
        if len(class_features) > 1:
            var = class_features.var(axis=0, ddof=1)
        else:
            var = np.zeros_like(mean)
        stats[int(label)] = {"mean": mean, "var": var}
    return stats


def _l2_normalize(array, eps=1e-12):
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, eps)


def build_auxiliary_statistics(encoder, config, device):
    dcfa = config["dcfa"]
    normalize_fn = _normalizer(config["dataset"]["normalize"])
    x_aux, y_aux = load_data(dcfa["aux_root"], dcfa["aux_num_classes"], "train")
    x_aux = normalize_fn(x_aux)
    features, labels = extract_encoder_features(
        encoder,
        x_aux,
        y_aux,
        device,
        batch_size=config["dataset"]["test_batch_size"],
        desc="Extracting auxiliary statistics",
    )
    return class_statistics(features, labels)


def distribution_calibrated_augmentation(features, labels, aux_stats, config):
    dcfa = config["dcfa"]
    rng = np.random.default_rng(config["random_seed"])
    real_stats = class_statistics(features, labels)
    aux_labels = np.array(sorted(aux_stats))
    aux_means = np.stack([aux_stats[int(label)]["mean"] for label in aux_labels], axis=0)
    aux_means_for_search = _l2_normalize(aux_means)

    aug_features = [features]
    aug_labels = [labels]
    for label, target_stat in real_stats.items():
        target_mean_for_search = _l2_normalize(target_stat["mean"][None, :])
        distances = np.linalg.norm(aux_means_for_search - target_mean_for_search, axis=1)
        top_count = min(dcfa["top_m"], len(aux_labels))
        neighbor_labels = aux_labels[np.argsort(distances)[:top_count]]
        neighbor_means = np.stack([aux_stats[int(item)]["mean"] for item in neighbor_labels], axis=0)
        neighbor_vars = np.stack([aux_stats[int(item)]["var"] for item in neighbor_labels], axis=0)

        calibrated_mean = (
            dcfa["alpha"] * target_stat["mean"]
            + (1.0 - dcfa["alpha"]) * neighbor_means.mean(axis=0)
        )
        calibrated_var = (
            dcfa["beta"] * target_stat["var"]
            + (1.0 - dcfa["beta"]) * neighbor_vars.mean(axis=0)
            + dcfa["epsilon"]
        )
        samples = rng.normal(
            loc=calibrated_mean,
            scale=np.sqrt(np.maximum(calibrated_var, dcfa["epsilon"])),
            size=(dcfa["aug_per_class"], features.shape[1]),
        ).astype(features.dtype, copy=False)
        aug_features.append(samples)
        aug_labels.append(np.full(dcfa["aug_per_class"], label, dtype=labels.dtype))

    return np.concatenate(aug_features, axis=0), np.concatenate(aug_labels, axis=0)
