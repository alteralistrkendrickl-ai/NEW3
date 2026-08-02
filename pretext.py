import os
import shutil
from copy import deepcopy
from math import exp

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from models.LocalFingerprintMNet import LocalFingerprintMNet
from models.lfdb import LightweightLFDB
from utils.channel_aug import (
    add_random_awgn,
    random_awgn_level_view,
    random_channel_view,
    random_joint_interference_view,
)
from utils.config import (
    is_joint_interference_method,
    is_local_fingerprint_method,
    pretrain_config,
    uses_temporal_encoder_config,
)
from utils.get_dataset import get_pretrain_dataloader
from utils.utils import (
    ListApply,
    RecordTime,
    accuracy,
    create_model,
    get_logger_and_writer,
    load_encoder_weights,
    set_seed,
)


def _forward_features(encoder, inputs, labels=None, mix_lambda=None):
    if labels is None:
        return encoder(inputs)
    return encoder(inputs, labels, mix_lambda)


def _forward_feature_map(encoder, inputs):
    if hasattr(encoder, "forward_map"):
        return encoder.forward_map(inputs)
    return encoder(inputs)


def _ramp_weight(config, epoch, weight_name):
    base_weight = config["lfdb"][weight_name]
    warmup_epochs = config["lfdb"].get("warmup_epochs", 0)
    if epoch < warmup_epochs:
        return 0.0
    if config["epoch"] <= warmup_epochs:
        return base_weight
    progress = (epoch + 1 - warmup_epochs) / max(config["epoch"] - warmup_epochs, 1)
    ramp = 2.0 / (1.0 + exp(-10.0 * max(0.0, min(1.0, progress)))) - 1.0
    return base_weight * ramp


def _stage_weight(config, epoch, stage_name):
    lfdb_conf = config["lfdb"]
    if stage_name == "con":
        return lfdb_conf["con_weight"] if epoch >= lfdb_conf.get("warmup_epochs", 0) else 0.0
    if stage_name == "orth":
        return lfdb_conf.get("orth_weight", 0.0) if epoch >= lfdb_conf.get("warmup_epochs", 0) else 0.0
    if stage_name == "rest_uniform":
        return lfdb_conf.get("rest_uniform_weight", 0.0) if epoch >= lfdb_conf.get("warmup_epochs", 0) else 0.0
    if stage_name == "rest_probe":
        return lfdb_conf.get("rest_probe_weight", 0.0)
    if stage_name == "res":
        return lfdb_conf.get("rest_uniform_weight", 0.0) if epoch >= lfdb_conf.get("warmup_epochs", 0) else 0.0
    if stage_name == "adv":
        start = lfdb_conf.get("warmup_epochs", 0) + lfdb_conf.get("stage2_epochs", 0)
        if epoch < start:
            return 0.0
        remaining = max(config["epoch"] - start, 1)
        progress = (epoch + 1 - start) / remaining
        ramp = 2.0 / (1.0 + exp(-10.0 * max(0.0, min(1.0, progress)))) - 1.0
        return lfdb_conf["adv_weight"] * ramp
    if stage_name == "rest_adv":
        if epoch < lfdb_conf.get("warmup_epochs", 0):
            return 0.0
        return lfdb_conf.get("rest_adv_weight", 0.0)
    raise ValueError(f"Unknown staged loss: {stage_name}")


def _orthogonal_loss(outputs):
    fp = F.normalize(outputs["fingerprint"], dim=1)
    rest = F.normalize(outputs["z_rest"], dim=1)
    cosine_loss = F.cosine_similarity(fp, rest, dim=1).pow(2).mean()

    fp_centered = fp - fp.mean(dim=0, keepdim=True)
    rest_centered = rest - rest.mean(dim=0, keepdim=True)
    covariance = fp_centered.t().matmul(rest_centered) / max(fp.shape[0] - 1, 1)
    covariance_loss = covariance.pow(2).mean()
    return cosine_loss + covariance_loss


def _uniform_prediction_loss(logits):
    log_probs = F.log_softmax(logits, dim=1)
    return -log_probs.mean(dim=1).mean()


def _supervised_contrastive_loss(features, labels, temperature=0.2):
    features = F.normalize(features, dim=1)
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.t()).float()
    logits = features.matmul(features.t()) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(positive_mask) - torch.eye(
        positive_mask.shape[0], device=positive_mask.device
    )
    positive_mask = positive_mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.new_tensor(0.0)
    loss = -(positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
    return loss[valid].mean()


def _normalized_map_distance(student_map, teacher_map):
    student_map = F.normalize(student_map, dim=1)
    teacher_map = F.normalize(teacher_map.detach(), dim=1)
    return (1.0 - (student_map * teacher_map).sum(dim=1)).mean()


def _multilevel_curriculum_levels(config, epoch, training):
    if not training:
        return (-10.0, -5.0, 0.0)
    low_start = config["lfdb"].get("low_snr_start_epoch", 10)
    very_low_start = config["lfdb"].get("very_low_snr_start_epoch", 30)
    if epoch < low_start:
        return (0.0, -5.0)
    if epoch < very_low_start:
        return (-5.0,) * 7 + (-10.0,) * 3
    extreme_weight, low_weight, clean_weight = config["lfdb"].get(
        "multilevel_snr_weights", (6, 3, 1)
    )
    return (
        (-10.0,) * extreme_weight
        + (-5.0,) * low_weight
        + (0.0,) * clean_weight
    )


def _prepare_teacher(module):
    if module is None:
        return
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _get_encoder_adaptation_modules(encoder):
    if not all(hasattr(encoder, name) for name in ("fusion", "encoder", "norm")):
        raise ValueError(
            "Selective encoder fine-tuning currently requires MSFTFNet."
        )
    layers = getattr(encoder.encoder, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("MSFTFNet does not expose a trainable Transformer layer.")
    return [encoder.fusion, layers[-1], encoder.norm]


def _configure_restoration_only(encoder, lfdb, selective_encoder_finetune=False):
    if lfdb is None or getattr(lfdb, "feature_restorer", None) is None:
        raise ValueError("Restoration-only training requires a feature restorer.")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in lfdb.parameters():
        parameter.requires_grad_(False)
    for parameter in lfdb.feature_restorer.parameters():
        parameter.requires_grad_(True)
    adaptation_modules = []
    if selective_encoder_finetune:
        adaptation_modules = _get_encoder_adaptation_modules(encoder)
        for module in adaptation_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return adaptation_modules


def _configure_multilevel_restoration(
    encoder,
    lfdb,
    selective_encoder_finetune=False,
):
    if not hasattr(encoder, "tf_enhancer") or not hasattr(
        encoder, "forward_stages"
    ):
        raise ValueError(
            "Multi-level restoration currently requires MSFTFNet."
        )
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in lfdb.parameters():
        parameter.requires_grad_(False)
    for parameter in encoder.tf_enhancer.parameters():
        parameter.requires_grad_(True)
    adaptation_modules = []
    if selective_encoder_finetune:
        adaptation_modules = _get_encoder_adaptation_modules(encoder)
        for module in adaptation_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return adaptation_modules


def _configure_triview_no_restore(
    encoder,
    lfdb,
    selective_encoder_finetune=False,
):
    if not hasattr(encoder, "tf_enhancer"):
        raise ValueError(
            "Tri-view no-restore training currently requires MSFTFNet."
        )
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in lfdb.parameters():
        parameter.requires_grad_(False)
    for parameter in encoder.tf_enhancer.parameters():
        parameter.requires_grad_(True)
    adaptation_modules = []
    if selective_encoder_finetune:
        adaptation_modules = _get_encoder_adaptation_modules(encoder)
        for module in adaptation_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return adaptation_modules


def _load_fixed_teacher_source(config, encoder, lfdb, device, logger):
    run_root = os.path.expanduser(config["lfdb"].get("teacher_run_root", ""))
    checkpoint_name = config["lfdb"].get("teacher_checkpoint", "best")
    if not run_root:
        raise ValueError("Fixed-teacher restoration requires teacher_run_root.")

    encoder_path = os.path.join(run_root, f"{checkpoint_name}_encoder.pth")
    lfdb_path = os.path.join(run_root, f"{checkpoint_name}_lfdb.pth")
    if not os.path.isfile(encoder_path) or not os.path.isfile(lfdb_path):
        raise FileNotFoundError(
            f"Fixed teacher weights not found: {encoder_path} / {lfdb_path}"
        )

    load_encoder_weights(encoder, encoder_path, device)
    lfdb_state = torch.load(lfdb_path, map_location=device)
    if isinstance(lfdb_state, dict) and "state_dict" in lfdb_state:
        lfdb_state = lfdb_state["state_dict"]
    incompatible = lfdb.load_state_dict(lfdb_state, strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("feature_restorer.")
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Teacher LFDB is incompatible with the restoration model. "
            f"Missing: {invalid_missing}; unexpected: "
            f"{incompatible.unexpected_keys}"
        )
    logger.info(f"==> Loaded pretrained encoder and LFDB initialization from: {run_root}")


@torch.no_grad()
def _ema_update_module(teacher, student, decay):
    if teacher is None or student is None:
        return
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters()
    ):
        teacher_parameter.mul_(decay).add_(
            student_parameter.detach(), alpha=1.0 - decay
        )
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        if teacher_buffer.is_floating_point():
            teacher_buffer.mul_(decay).add_(
                student_buffer.detach(), alpha=1.0 - decay
            )
        else:
            teacher_buffer.copy_(student_buffer)


def run_step(config, inputs, device, encoder, rot_classifier, mixed_classifier,
             cls, mml, mtl, lfdb=None, teacher_encoder=None, teacher_lfdb=None,
             training=False, epoch=0):
    """Run all configured pretext tasks for one batch."""
    signals, rot_labels, device_labels = inputs
    batch_size, rot_num = signals.shape[:2]

    rot_inputs = signals.reshape(batch_size * rot_num, *signals.shape[2:]).to(device)
    if training and config["augmentation"]["awgn_enable"]:
        rot_inputs = add_random_awgn(rot_inputs, config["augmentation"]["awgn_snr_range"])

    rot_labels = rot_labels.reshape(-1).long().to(device)
    device_labels = device_labels.long().to(device)
    expanded_device_labels = device_labels.repeat_interleave(rot_num)
    mixed_inputs = signals[:, 0].to(device)

    base_features = None
    lfdb_outputs = None
    channel_outputs = None
    clean_output = None
    teacher_clean_output = None
    teacher_clean_stages = None
    losses = {}
    metrics = {"rot_acc": 0.0, "sei_acc": 0.0, "mixed_acc": 0.0}
    local_method = is_local_fingerprint_method(config.get("method_name"))

    def get_base_features():
        nonlocal base_features
        if base_features is None:
            base_features = _forward_features(encoder, rot_inputs)
        return base_features

    def get_fingerprint_features():
        nonlocal lfdb_outputs
        features = get_base_features()
        if lfdb is None:
            return features
        if lfdb_outputs is None:
            lfdb_outputs = lfdb(features, return_all=True)
        return lfdb_outputs["fingerprint"]

    def get_channel_outputs():
        nonlocal channel_outputs
        if channel_outputs is not None:
            return channel_outputs
        if is_joint_interference_method(config.get("method_name")):
            snr_levels = config["augmentation"].get("snr_levels")
            view_1_levels = snr_levels
            view_2_levels = snr_levels
            uses_multilevel_curriculum = (
                config["lfdb"].get("use_multilevel_restoration", False)
                or config["lfdb"].get(
                    "use_triview_curriculum_no_restore", False
                )
            )
            if uses_multilevel_curriculum:
                curriculum_levels = _multilevel_curriculum_levels(
                    config, epoch, training
                )
                view_1_levels = curriculum_levels
                view_2_levels = curriculum_levels
            elif config["lfdb"].get("is_clean_anchor_v2", False):
                low_start = config["lfdb"].get("low_snr_start_epoch", 20)
                very_low_start = config["lfdb"].get("very_low_snr_start_epoch", 60)
                view_1_levels = tuple(level for level in snr_levels if level >= 0.0)
                if training and epoch < low_start:
                    view_2_levels = tuple(level for level in snr_levels if 0.0 <= level <= 10.0)
                elif training and epoch < very_low_start:
                    view_2_levels = tuple(level for level in snr_levels if -5.0 <= level <= 5.0)
                else:
                    if config["lfdb"].get("is_clean_anchor_v3", False):
                        extreme_levels = tuple(level for level in snr_levels if level <= -10.0)
                        transition_levels = tuple(level for level in snr_levels if -5.0 <= level <= 0.0)
                        view_2_levels = extreme_levels + extreme_levels + transition_levels
                    else:
                        view_2_levels = tuple(level for level in snr_levels if level <= 5.0)
            elif config["lfdb"].get("is_clean_anchor", False) and training:
                minimum_snr = (
                    -10.0 if epoch >= config["lfdb"].get("very_low_snr_start_epoch", 60)
                    else -5.0 if epoch >= config["lfdb"].get("low_snr_start_epoch", 20)
                    else 0.0
                )
                view_1_levels = tuple(level for level in snr_levels if level >= minimum_snr)
                view_2_levels = view_1_levels
            if not view_1_levels or not view_2_levels:
                raise ValueError("The configured SNR levels do not support the active curriculum stage.")
            if uses_multilevel_curriculum:
                view_1, snr_1, fading_1 = random_awgn_level_view(
                    mixed_inputs,
                    view_1_levels,
                )
                view_2, snr_2, fading_2 = random_awgn_level_view(
                    mixed_inputs,
                    view_2_levels,
                )
            else:
                view_1, snr_1, fading_1 = random_joint_interference_view(
                    mixed_inputs,
                    view_1_levels,
                    enable_awgn=config["augmentation"]["awgn_enable"],
                    low_snr_prob=config["lfdb"].get(
                        "snrboost_low_prob", 0.0
                    )
                    if config["lfdb"].get("is_snrboost", False) else 0.0,
                    low_snr_max=config["lfdb"].get(
                        "snrboost_low_max", 0.0
                    ),
                )
                view_2, snr_2, fading_2 = random_joint_interference_view(
                    mixed_inputs,
                    view_2_levels,
                    enable_awgn=config["augmentation"]["awgn_enable"],
                    low_snr_prob=config["lfdb"].get(
                        "snrboost_low_prob", 0.0
                    )
                    if config["lfdb"].get("is_snrboost", False) else 0.0,
                    low_snr_max=config["lfdb"].get(
                        "snrboost_low_max", 0.0
                    ),
                )
        else:
            view_1, snr_1, fading_1 = random_channel_view(
                mixed_inputs, config["augmentation"]["awgn_snr_range"],
                enable_awgn=config["augmentation"]["awgn_enable"]
            )
            view_2, snr_2, fading_2 = random_channel_view(
                mixed_inputs, config["augmentation"]["awgn_snr_range"],
                enable_awgn=config["augmentation"]["awgn_enable"]
            )
        stages_1 = None
        stages_2 = None
        if local_method:
            if config["lfdb"].get("use_multilevel_restoration", False):
                stages_1 = encoder.forward_stages(view_1)
                stages_2 = encoder.forward_stages(view_2)
                out_1 = lfdb(stages_1["feature_map"], return_all=True)
                out_2 = lfdb(stages_2["feature_map"], return_all=True)
            else:
                out_1 = lfdb(
                    _forward_feature_map(encoder, view_1), return_all=True
                )
                out_2 = lfdb(
                    _forward_feature_map(encoder, view_2), return_all=True
                )
        else:
            out_1 = lfdb(_forward_features(encoder, view_1), return_all=True)
            out_2 = lfdb(_forward_features(encoder, view_2), return_all=True)
        env_1 = fading_1.long() * int(config["lfdb"]["snr_classes"]) + snr_1.long()
        env_2 = fading_2.long() * int(config["lfdb"]["snr_classes"]) + snr_2.long()
        channel_outputs = {
            "out_1": out_1,
            "out_2": out_2,
            "snr": torch.cat([snr_1, snr_2]).long(),
            "fading": torch.cat([fading_1, fading_2]).long(),
            "env": torch.cat([env_1, env_2]).long(),
            "device_labels": torch.cat([device_labels, device_labels]).long(),
            "stages_1": stages_1,
            "stages_2": stages_2,
        }
        return channel_outputs

    def get_clean_output():
        nonlocal clean_output
        if clean_output is None:
            clean_features = _forward_feature_map(encoder, mixed_inputs)
            clean_output = lfdb(clean_features, return_all=True)
        return clean_output

    def get_teacher_clean_output():
        nonlocal teacher_clean_output
        teacher_mode = config["lfdb"].get("teacher_mode", "none")
        teacher_start_epoch = (
            0
            if teacher_mode == "fixed"
            else config["lfdb"].get("ema_start_epoch", 1)
        )
        if (
            teacher_encoder is None
            or teacher_lfdb is None
            or epoch < teacher_start_epoch
        ):
            return get_clean_output()
        if teacher_clean_output is None:
            with torch.no_grad():
                teacher_features = _forward_feature_map(
                    teacher_encoder, mixed_inputs
                )
                teacher_clean_output = teacher_lfdb(
                    teacher_features, return_all=True
                )
        return teacher_clean_output

    def get_teacher_clean_stages():
        nonlocal teacher_clean_stages
        if teacher_encoder is None or not hasattr(
            teacher_encoder, "forward_stages"
        ):
            raise ValueError(
                "Multi-level restoration requires a fixed MSFTFNet teacher."
            )
        if teacher_clean_stages is None:
            with torch.no_grad():
                teacher_clean_stages = teacher_encoder.forward_stages(
                    mixed_inputs
                )
        return teacher_clean_stages

    for loss_name in config["mtl"]["item"]:
        if loss_name == "id":
            channel = get_channel_outputs()
            if local_method:
                predictions = torch.cat([
                    channel["out_1"]["id_logits"],
                    channel["out_2"]["id_logits"],
                ], dim=0)
                if (
                    config["lfdb"].get("is_amlf", False)
                    and config["lfdb"].get("global_id_weight", 0.0) > 0
                    and channel["out_1"].get("global_id_logits") is not None
                ):
                    global_predictions = torch.cat([
                        channel["out_1"]["global_id_logits"],
                        channel["out_2"]["global_id_logits"],
                    ], dim=0)
                else:
                    global_predictions = None
            else:
                fingerprints = torch.cat([
                    channel["out_1"]["fingerprint"],
                    channel["out_2"]["fingerprint"],
                ], dim=0)
                predictions = mixed_classifier(fingerprints)
                global_predictions = None
            noisy_id_loss = cls(predictions, channel["device_labels"])
            if config["lfdb"].get("is_clean_anchor", False):
                clean = get_clean_output()
                id_loss = (
                    config["lfdb"].get("clean_id_weight", 1.0)
                    * cls(clean["id_logits"], device_labels)
                    + config["lfdb"].get("noisy_id_weight", 0.5) * noisy_id_loss
                )
            else:
                id_loss = noisy_id_loss
            if global_predictions is not None:
                id_loss = id_loss + config["lfdb"]["global_id_weight"] * cls(
                    global_predictions, channel["device_labels"]
                )
            if config["lfdb"].get("is_cifd", False) and config["lfdb"].get("clean_id_weight", 0.0) > 0:
                clean_output = get_clean_output()
                id_loss = id_loss + config["lfdb"]["clean_id_weight"] * cls(
                    clean_output["id_logits"], device_labels
                )
            losses[loss_name] = id_loss
            metrics["sei_acc"] = accuracy(predictions, channel["device_labels"])

        elif loss_name == "clean_cons":
            channel = get_channel_outputs()
            clean_features = get_teacher_clean_output()["id_features"].detach()
            consistency = 0.5 * (
                1.0 - F.cosine_similarity(
                    channel["out_1"]["id_features"], clean_features, dim=1
                ).mean()
                + 1.0 - F.cosine_similarity(
                    channel["out_2"]["id_features"], clean_features, dim=1
                ).mean()
            )
            losses[loss_name] = config["lfdb"].get("clean_cons_weight", 0.2) * consistency

        elif loss_name == "multi_restore":
            channel = get_channel_outputs()
            teacher_stages = get_teacher_clean_stages()
            stage_names = ("time_map", "freq_map", "fused_map")
            view_losses = []
            for student_stages in (
                channel["stages_1"],
                channel["stages_2"],
            ):
                stage_loss = sum(
                    _normalized_map_distance(
                        student_stages[name],
                        teacher_stages[name],
                    )
                    for name in stage_names
                ) / len(stage_names)
                view_losses.append(stage_loss)
            restoration_loss = sum(view_losses) / len(view_losses)
            losses[loss_name] = (
                config["lfdb"].get("multilevel_restore_weight", 0.2)
                * restoration_loss
            )

        elif loss_name == "rest_adv":
            channel = get_channel_outputs()
            rest_logits = torch.cat([
                channel["out_1"]["rest_id_logits"],
                channel["out_2"]["rest_id_logits"],
            ], dim=0)
            losses[loss_name] = _stage_weight(config, epoch, "rest_adv") * cls(
                rest_logits, channel["device_labels"]
            )

        elif loss_name == "supcon":
            channel = get_channel_outputs()
            features = torch.cat([
                channel["out_1"]["id_features"],
                channel["out_2"]["id_features"],
            ], dim=0)
            losses[loss_name] = config["lfdb"].get("supcon_weight", 0.0) * _supervised_contrastive_loss(
                features,
                channel["device_labels"],
                config["lfdb"].get("supcon_temp", 0.2),
            )

        elif loss_name == "orth":
            channel = get_channel_outputs()
            orth_loss = 0.5 * (
                _orthogonal_loss(channel["out_1"]) + _orthogonal_loss(channel["out_2"])
            )
            if config["lfdb"].get("is_cifd", False):
                mask_loss = 0.5 * (
                    channel["out_1"]["mask_loss"] + channel["out_2"]["mask_loss"]
                )
                orth_loss = orth_loss + config["lfdb"].get("mask_weight", 0.0) * mask_loss
            losses[loss_name] = _stage_weight(config, epoch, "orth") * orth_loss

        elif loss_name == "res":
            channel = get_channel_outputs()
            rest_probe_logits = torch.cat([
                channel["out_1"]["rest_probe_logits"],
                channel["out_2"]["rest_probe_logits"],
            ], dim=0)
            rest_probe_loss = config["lfdb"].get("rest_probe_weight", 0.0) * cls(
                rest_probe_logits, channel["device_labels"]
            )
            rest_1 = channel["out_1"].get("rest_uniform_logits")
            rest_2 = channel["out_2"].get("rest_uniform_logits")
            rest_uniform_logits = torch.cat([
                rest_1 if rest_1 is not None else channel["out_1"]["rest_id_logits"],
                rest_2 if rest_2 is not None else channel["out_2"]["rest_id_logits"],
            ], dim=0)
            rest_uniform_loss = _stage_weight(config, epoch, "res") * _uniform_prediction_loss(rest_uniform_logits)
            losses[loss_name] = rest_probe_loss + rest_uniform_loss

        elif loss_name == "rest_probe":
            channel = get_channel_outputs()
            rest_probe_logits = torch.cat([
                channel["out_1"]["rest_probe_logits"],
                channel["out_2"]["rest_probe_logits"],
            ], dim=0)
            losses[loss_name] = _stage_weight(config, epoch, "rest_probe") * cls(
                rest_probe_logits, channel["device_labels"]
            )

        elif loss_name == "rest_uniform":
            channel = get_channel_outputs()
            rest_1 = channel["out_1"].get("rest_uniform_logits")
            rest_2 = channel["out_2"].get("rest_uniform_logits")
            rest_logits = torch.cat([
                rest_1 if rest_1 is not None else channel["out_1"]["rest_id_logits"],
                rest_2 if rest_2 is not None else channel["out_2"]["rest_id_logits"],
            ], dim=0)
            losses[loss_name] = _stage_weight(config, epoch, "rest_uniform") * _uniform_prediction_loss(rest_logits)

        elif loss_name == "inv":
            channel = get_channel_outputs()
            inv_loss = 1.0 - F.cosine_similarity(
                channel["out_1"]["fingerprint"],
                channel["out_2"]["fingerprint"],
                dim=1,
            ).mean()
            losses[loss_name] = _ramp_weight(config, epoch, "inv_weight") * inv_loss

        elif loss_name == "int":
            channel = get_channel_outputs()
            snr_logits = torch.cat([
                channel["out_1"]["snr_logits"],
                channel["out_2"]["snr_logits"],
            ], dim=0)
            fading_logits = torch.cat([
                channel["out_1"]["fading_logits"],
                channel["out_2"]["fading_logits"],
            ], dim=0)
            losses[loss_name] = config["lfdb"]["int_weight"] * (
                cls(snr_logits, channel["snr"]) + cls(fading_logits, channel["fading"])
            )

        elif loss_name == "mask" and is_joint_interference_method(config.get("method_name")) and not local_method:
            channel = get_channel_outputs()
            mask_mean = torch.cat([
                channel["out_1"]["mask"],
                channel["out_2"]["mask"],
            ], dim=0).mean()
            target = mask_mean.new_tensor(config["lfdb"]["mask_ratio"])
            losses[loss_name] = config["lfdb"]["mask_weight"] * torch.abs(mask_mean - target)

        elif loss_name == "rot_cls":
            predictions = rot_classifier(get_base_features())
            losses[loss_name] = cls(predictions, rot_labels)
            metrics["rot_acc"] = accuracy(predictions, rot_labels)

        elif loss_name == "sei_cls":
            predictions = mixed_classifier(get_fingerprint_features())
            losses[loss_name] = cls(predictions, expanded_device_labels)
            metrics["sei_acc"] = accuracy(predictions, expanded_device_labels)

        elif loss_name == "mml":
            mix_lambda = mml.get_lamda()
            mixed_features, labels_a, labels_b = _forward_features(
                encoder, mixed_inputs, device_labels, mix_lambda
            )
            if lfdb is not None:
                mixed_features, _, _ = lfdb(mixed_features)
            predictions = mixed_classifier(mixed_features)
            losses[loss_name] = mml(predictions, labels_a, labels_b)
            metrics["mixed_acc"] = (
                mix_lambda * accuracy(predictions, labels_a)
                + (1.0 - mix_lambda) * accuracy(predictions, labels_b)
            )

        elif loss_name == "con":
            channel = get_channel_outputs()
            con_loss = 1.0 - F.cosine_similarity(
                channel["out_1"]["fingerprint"],
                channel["out_2"]["fingerprint"],
                dim=1,
            ).mean()
            weight = _stage_weight(config, epoch, "con") if local_method else _ramp_weight(config, epoch, "con_weight")
            losses[loss_name] = weight * con_loss

        elif loss_name == "adv":
            channel = get_channel_outputs()
            if local_method:
                adv_logits = torch.cat([
                    channel["out_1"]["env_logits"],
                    channel["out_2"]["env_logits"],
                ], dim=0)
                losses[loss_name] = _stage_weight(config, epoch, "adv") * cls(
                    adv_logits, channel["env"]
                )
            else:
                adv_logits = torch.cat([
                    channel["out_1"]["adv_logits"],
                    channel["out_2"]["adv_logits"],
                ], dim=0)
                losses[loss_name] = _ramp_weight(config, epoch, "adv_weight") * cls(
                    adv_logits, channel["device_labels"]
                )

        elif loss_name == "ch":
            channel = get_channel_outputs()
            snr_logits = torch.cat([
                channel["out_1"]["snr_logits"],
                channel["out_2"]["snr_logits"],
            ], dim=0)
            fading_logits = torch.cat([
                channel["out_1"]["fading_logits"],
                channel["out_2"]["fading_logits"],
            ], dim=0)
            losses[loss_name] = config["lfdb"]["ch_weight"] * (
                cls(snr_logits, channel["snr"]) + cls(fading_logits, channel["fading"])
            )

        elif loss_name == "mask":
            if local_method:
                channel = get_channel_outputs()
                mask_loss = 0.5 * (
                    channel["out_1"]["mask_loss"] + channel["out_2"]["mask_loss"]
                )
                losses[loss_name] = config["lfdb"]["mask_weight"] * mask_loss
            else:
                get_fingerprint_features()
                mask_mean = lfdb_outputs["mask"].mean()
                target = mask_mean.new_tensor(config["lfdb"]["mask_ratio"])
                losses[loss_name] = config["lfdb"]["mask_weight"] * torch.abs(mask_mean - target)
        else:
            raise ValueError(f"Unknown pretext loss: {loss_name}")

    ordered_losses = [losses[name] for name in config["mtl"]["item"]]
    if local_method and config["lfdb"].get("manual_local_loss", False):
        total_loss = sum(ordered_losses)
    else:
        total_loss = mtl(*ordered_losses)
    return ordered_losses + [total_loss], metrics


def _run_epoch(logger, writer, config, epoch, dataloader, device, encoder,
               rot_classifier, mixed_classifier, cls, mml, mtl, lfdb=None,
               optimizers=None, schedulers=None, teacher_encoder=None,
               teacher_lfdb=None):
    training = optimizers is not None
    modules = [encoder, rot_classifier, mixed_classifier, mtl]
    if lfdb is not None:
        modules.append(lfdb)
    for module in modules:
        module.train(training)
    if training and config["lfdb"].get("restoration_only", False):
        encoder.eval()
        if lfdb is not None:
            lfdb.eval()
            if getattr(lfdb, "feature_restorer", None) is not None:
                lfdb.feature_restorer.train()
        if (
            config["lfdb"].get("use_multilevel_restoration", False)
            or config["lfdb"].get(
                "use_triview_curriculum_no_restore", False
            )
        ):
            encoder.tf_enhancer.train()
        if config["lfdb"].get("selective_encoder_finetune", False):
            for module in _get_encoder_adaptation_modules(encoder):
                module.train()
    _prepare_teacher(teacher_encoder)
    _prepare_teacher(teacher_lfdb)

    metric_sums = {"rot_acc": 0.0, "sei_acc": 0.0, "mixed_acc": 0.0}
    loss_sums = [0.0] * (config["mtl"]["num"] + 1)
    split_name = "Train" if training else "Val"

    if training:
        learning_rates = [
            group["lr"]
            for optimizer in optimizers
            for group in optimizer.param_groups
        ]
        logger.info(
            "==> lr = " + ", ".join(str(value) for value in learning_rates)
        )

    progress = tqdm(dataloader, desc=f"{split_name} epoch {epoch + 1}/{config['epoch']}")
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for inputs in progress:
            loss_items, metrics = run_step(
                config, inputs, device, encoder, rot_classifier, mixed_classifier,
                cls, mml, mtl, lfdb, teacher_encoder, teacher_lfdb,
                training=training, epoch=epoch
            )
            if training:
                optimizers.zero_grad()
                loss_items[-1].backward()
                if config["lfdb"].get("grad_clip", 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for optimizer in optimizers for group in optimizer.param_groups for parameter in group["params"]],
                        config["lfdb"]["grad_clip"],
                    )
                optimizers.step()
                if (
                    teacher_encoder is not None
                    and teacher_lfdb is not None
                    and config["lfdb"].get("teacher_mode") == "ema"
                ):
                    decay = config["lfdb"].get("ema_decay", 0.996)
                    _ema_update_module(teacher_encoder, encoder, decay)
                    _ema_update_module(teacher_lfdb, lfdb, decay)

            for name, value in metrics.items():
                metric_sums[name] += value
            for index, loss in enumerate(loss_items):
                loss_sums[index] += loss.item()

    if training:
        schedulers.step()

    count = max(len(dataloader), 1)
    metrics = {name: value / count for name, value in metric_sums.items()}
    losses = [value / count for value in loss_sums]
    loss_names = [name.upper() for name in config["mtl"]["item"]] + ["TOTAL"]
    loss_text = ", ".join(f"{name}: {value:.6f}" for name, value in zip(loss_names, losses))
    logger.info(
        f"==> {split_name} Set: Rot-Acc: {metrics['rot_acc']:.2f}%, "
        f"SEI-Acc: {metrics['sei_acc']:.2f}%, Mixed-Acc: {metrics['mixed_acc']:.2f}%, "
        f"{loss_text}"
    )

    for name, value in metrics.items():
        writer.add_scalar(f"{split_name}/{name}", value, epoch)
    for name, value in zip(loss_names, losses):
        writer.add_scalar(f"{split_name}/loss_{name.lower()}", value, epoch)
    return metrics, losses


def _save_checkpoint(path, epoch, config, best_record, encoder, rot_classifier,
                     mixed_classifier, mtl, lfdb, optimizers, schedulers,
                     teacher_encoder=None, teacher_lfdb=None):
    torch.save({
        "epoch": epoch,
        "config": config,
        "best_record": best_record,
        "encoder": encoder.state_dict(),
        "rot_classifier": rot_classifier.state_dict(),
        "mixed_classifier": mixed_classifier.state_dict(),
        "mtl": mtl.state_dict(),
        "lfdb": lfdb.state_dict() if lfdb is not None else None,
        "teacher_encoder": (
            teacher_encoder.state_dict() if teacher_encoder is not None else None
        ),
        "teacher_lfdb": (
            teacher_lfdb.state_dict() if teacher_lfdb is not None else None
        ),
        "optimizers": optimizers.state_dict(),
        "schedulers": schedulers.state_dict(),
    }, path)


def _load_checkpoint(path, device, encoder, rot_classifier, mixed_classifier,
                     mtl, lfdb, optimizers, schedulers, teacher_encoder=None,
                     teacher_lfdb=None):
    checkpoint = torch.load(path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    rot_classifier.load_state_dict(checkpoint["rot_classifier"])
    mixed_classifier.load_state_dict(checkpoint["mixed_classifier"])
    mtl.load_state_dict(checkpoint["mtl"])
    if lfdb is not None and checkpoint.get("lfdb") is not None:
        lfdb.load_state_dict(checkpoint["lfdb"])
    if teacher_encoder is not None:
        teacher_encoder.load_state_dict(
            checkpoint.get("teacher_encoder") or checkpoint["encoder"]
        )
    if teacher_lfdb is not None:
        teacher_lfdb.load_state_dict(
            checkpoint.get("teacher_lfdb") or checkpoint["lfdb"]
        )
    if checkpoint.get("optimizers") is not None:
        optimizers.load_state_dict(checkpoint["optimizers"])
    if checkpoint.get("schedulers") is not None:
        schedulers.load_state_dict(checkpoint["schedulers"])
    return checkpoint


def train_and_val(record_time, logger, writer, config, train_dl, val_dl, device,
                  encoder, rot_classifier, mixed_classifier, cls, mml, mtl,
                  lfdb, optimizers, schedulers, checkpoint=None,
                  teacher_encoder=None, teacher_lfdb=None):
    best_record = (
        deepcopy(checkpoint["best_record"])
        if checkpoint is not None and "best_record" in checkpoint
        else {"epoch": -1, "metrics": {}, "loss": [float("inf")] * (config["mtl"]["num"] + 1)}
    )

    for epoch in range(config["start_epoch"], config["epoch"]):
        logger.info("--------------------------------------------")
        logger.info(f"Epoch {epoch + 1}/{config['epoch']}")
        record_time.start()

        _run_epoch(
            logger, writer, config, epoch, train_dl, device, encoder,
            rot_classifier, mixed_classifier, cls, mml, mtl, lfdb,
            optimizers, schedulers, teacher_encoder, teacher_lfdb
        )
        metrics, losses = _run_epoch(
            logger, writer, config, epoch, val_dl, device, encoder,
            rot_classifier, mixed_classifier, cls, mml, mtl, lfdb,
            teacher_encoder=teacher_encoder, teacher_lfdb=teacher_lfdb
        )

        current_score = metrics.get("sei_acc", 0.0)
        best_score = best_record.get("metrics", {}).get("sei_acc", float("-inf"))
        if current_score > best_score or (
            current_score == best_score and sum(losses[:-1]) < sum(best_record["loss"][:-1])
        ):
            best_record = {"epoch": epoch, "metrics": metrics, "loss": losses}
            torch.save(encoder.state_dict(), os.path.join(config["exp_path"], "best_encoder.pth"))
            torch.save(mixed_classifier.state_dict(), os.path.join(config["exp_path"], "best_id_classifier.pth"))
            if lfdb is not None:
                torch.save(lfdb.state_dict(), os.path.join(config["exp_path"], "best_lfdb.pth"))
            logger.info(f"==> Best encoder saved at epoch {epoch + 1}.")

        torch.save(encoder.state_dict(), os.path.join(config["exp_path"], "final_encoder.pth"))
        torch.save(mixed_classifier.state_dict(), os.path.join(config["exp_path"], "final_id_classifier.pth"))
        if lfdb is not None:
            torch.save(lfdb.state_dict(), os.path.join(config["exp_path"], "final_lfdb.pth"))
        if epoch % config["save_freq"] == 0 or epoch == config["epoch"] - 1:
            checkpoint_path = os.path.join(config["exp_path"], "checkpoint.pth")
            _save_checkpoint(
                checkpoint_path, epoch, config, best_record, encoder,
                rot_classifier, mixed_classifier, mtl, lfdb, optimizers, schedulers,
                teacher_encoder, teacher_lfdb
            )
            shutil.copy2(checkpoint_path, os.path.join(config["save_path"], "checkpoint.pth"))

        logger.info(
            "==> Time spent (current/mean/total/remain): {}/{}/{}/{}"
            .format(*record_time.step())
        )

    for filename in (
        "best_encoder.pth",
        "final_encoder.pth",
        "best_id_classifier.pth",
        "final_id_classifier.pth",
        "best_lfdb.pth",
        "final_lfdb.pth",
    ):
        source = os.path.join(config["exp_path"], filename)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(config["save_path"], filename))
    logger.info("--------------------------------------------")
    logger.info(f"End. Best Record: {best_record}")


def pretext(config=None):
    config = pretrain_config() if config is None else config
    set_seed(config["random_seed"])
    config["exp_type"] += "_random_rot"
    run_root = os.path.join("runs", config["exp_type"], config["exp_name"])
    logger, writer, exp_path, save_path = get_logger_and_writer(run_root)
    config["exp_path"] = exp_path
    config["save_path"] = save_path

    logger.info(f"==> Running file: {os.path.abspath(__file__)}")
    logger.info(f"==> Config: {config}")
    train_dataloader, val_dataloader = get_pretrain_dataloader(config)
    device = torch.device(config["device"])

    encoder_config = config["encoder"]
    encoder_kwargs = {
        "feature_dim": encoder_config["feature_dim"],
        "dtype": config["dataset"]["type"],
    }
    if uses_temporal_encoder_config(encoder_config["name"]):
        encoder_kwargs.update(encoder_config["TSLA_config"])
    encoder = create_model(encoder_config["root"], **encoder_kwargs).to(device)
    rot_classifier = create_model(
        config["rot_classifier"]["root"],
        in_dim=config["rot_classifier"]["in_dim"],
        num_classes=config["rot_classifier"]["num_classes"],
    ).to(device)
    mixed_classifier = create_model(
        config["mixed_classifier"]["root"],
        in_dim=config["mixed_classifier"]["in_dim"],
        num_classes=config["mixed_classifier"]["num_classes"],
    ).to(device)

    lfdb = None
    if config["lfdb"]["enabled"]:
        if is_local_fingerprint_method(config.get("method_name")):
            local_channels = (
                config["encoder"]["TSLA_config"]["emb_dim"]
                if uses_temporal_encoder_config(config["encoder"]["name"])
                else config["encoder"]["feature_dim"]
            )
            lfdb = LocalFingerprintMNet(
                in_channels=local_channels,
                num_classes=config["lfdb"]["num_classes"],
                env_classes=config["lfdb"]["snr_classes"] * config["lfdb"]["fading_classes"],
                mask_min=config["lfdb"]["mask_min"],
                mask_max=config["lfdb"]["mask_max"],
                tv_weight=config["lfdb"]["mask_tv_weight"],
                fusion_mode=config["lfdb"].get("fusion_mode", "fingerprint"),
                use_rest_adv=config["lfdb"].get("use_rest_adv", False),
                use_rest_probe=config["lfdb"].get("use_orth", False),
                use_rest_projector=config["lfdb"].get("use_rest_projector", False),
                use_multiscale=config["lfdb"].get("use_multiscale", False),
                use_global_head=config["lfdb"].get("use_global_head", False),
                use_cosine_head=config["lfdb"].get("use_cosine_head", False),
                cosine_scale=config["lfdb"].get("cosine_scale", 16.0),
                use_feature_restorer=config["lfdb"].get(
                    "use_feature_restorer", False
                ),
            ).to(device)
        else:
            lfdb = LightweightLFDB(
                feat_dim=config["encoder"]["feature_dim"],
                num_classes=config["lfdb"]["num_classes"],
                snr_classes=config["lfdb"]["snr_classes"],
                fading_classes=config["lfdb"]["fading_classes"],
            ).to(device)

    teacher_encoder = None
    teacher_lfdb = None
    encoder_adaptation_modules = []
    triview_no_restore = config["lfdb"].get(
        "use_triview_curriculum_no_restore", False
    )
    if triview_no_restore:
        if lfdb is None:
            raise ValueError("Tri-view no-restore training requires LFDB.")
        _load_fixed_teacher_source(config, encoder, lfdb, device, logger)
        if config["lfdb"].get("restoration_only", False):
            encoder_adaptation_modules = _configure_triview_no_restore(
                encoder,
                lfdb,
                config["lfdb"].get("selective_encoder_finetune", False),
            )
    if (
        config["lfdb"].get("use_feature_restorer", False)
        or config["lfdb"].get("use_multilevel_restoration", False)
    ):
        if lfdb is None:
            raise ValueError("Feature restoration requires the local fingerprint module.")
        if config["lfdb"].get("teacher_mode") == "fixed":
            _load_fixed_teacher_source(
                config, encoder, lfdb, device, logger
            )
        teacher_encoder = deepcopy(encoder)
        teacher_lfdb = deepcopy(lfdb)
        _prepare_teacher(teacher_encoder)
        _prepare_teacher(teacher_lfdb)
        if config["lfdb"].get("teacher_mode") == "fixed":
            logger.info("==> Created frozen teacher snapshot from initialization.")
        if config["lfdb"].get("restoration_only", False):
            if config["lfdb"].get("use_multilevel_restoration", False):
                encoder_adaptation_modules = (
                    _configure_multilevel_restoration(
                        encoder,
                        lfdb,
                        config["lfdb"].get(
                            "selective_encoder_finetune", False
                        ),
                    )
                )
            else:
                encoder_adaptation_modules = _configure_restoration_only(
                    encoder,
                    lfdb,
                    config["lfdb"].get(
                        "selective_encoder_finetune", False
                    ),
                )

    cls = torch.nn.CrossEntropyLoss()
    mml = create_model(config["mml"]["root"], beta=config["mml"]["beta"])
    mtl = create_model(config["mtl"]["root"], num=config["mtl"]["num"]).to(device)

    optimizer_config = config["optimizer"]
    if config["lfdb"].get("restoration_only", False):
        if (
            config["lfdb"].get("use_multilevel_restoration", False)
            or triview_no_restore
        ):
            restoration_parameters = list(
                encoder.tf_enhancer.parameters()
            )
            restoration_name = "tf_enhancer"
        else:
            restoration_parameters = list(
                lfdb.feature_restorer.parameters()
            )
            restoration_name = "restorer"
        parameter_groups = [{
            "params": restoration_parameters,
            "lr": optimizer_config["lr"],
        }]
        if encoder_adaptation_modules:
            adaptation_parameters = [
                parameter
                for module in encoder_adaptation_modules
                for parameter in module.parameters()
            ]
            parameter_groups.append({
                "params": adaptation_parameters,
                "lr": config["lfdb"]["encoder_adapt_lr"],
            })
            logger.info(
                "==> Selective encoder fine-tuning: fusion gate, final "
                "Transformer layer, and output normalization."
            )
            logger.info(
                "==> Trainable parameters: "
                f"{restoration_name}="
                f"{sum(parameter.numel() for parameter in parameter_groups[0]['params'])}, "
                "encoder_adapter="
                f"{sum(parameter.numel() for parameter in adaptation_parameters)}."
            )
        optimizers = ListApply([
            AdamW(
                parameter_groups,
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config["weight_decay"],
            )
        ])
    else:
        trainable_modules = [encoder, rot_classifier, mixed_classifier, mtl]
        if lfdb is not None:
            trainable_modules.append(lfdb)
        optimizers = ListApply([
            AdamW(
                module.parameters(),
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config["weight_decay"],
            )
            for module in trainable_modules
        ])
    schedulers = ListApply([
        StepLR(
            optimizer,
            step_size=optimizer_config["step_size"],
            gamma=optimizer_config["gamma"],
        )
        for optimizer in optimizers
    ])

    checkpoint = None
    resume_path = config.get("resume", "")
    if resume_path:
        if os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, "checkpoint.pth")
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        checkpoint = _load_checkpoint(
            resume_path, device, encoder, rot_classifier, mixed_classifier,
            mtl, lfdb, optimizers, schedulers, teacher_encoder, teacher_lfdb
        )
        config["start_epoch"] = checkpoint["epoch"] + 1
        logger.info(f"==> Resumed checkpoint: {resume_path}")
    else:
        config["start_epoch"] = 0

    record_time = RecordTime(max(config["epoch"] - config["start_epoch"], 0))
    train_and_val(
        record_time, logger, writer, config, train_dataloader, val_dataloader,
        device, encoder, rot_classifier, mixed_classifier, cls, mml, mtl,
        lfdb, optimizers, schedulers, checkpoint, teacher_encoder, teacher_lfdb
    )
    writer.close()


if __name__ == "__main__":
    pretext(pretrain_config(
        encoder_name="CVTSLANet",
        classifiar_name="Linear",
        dataset_name="ads-b",
        input_type="iq",
        rot_num=8,
        feature_dim=1024,
        max_epoch=5,
        save_freq=1,
    ))
