import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lfdb import GradientReversal


class CosineClassifier(nn.Module):
    """Normalized classifier for compact fine-grained emitter features."""

    def __init__(self, in_dim, num_classes, scale=16.0, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        self.scale = float(scale)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = self.dropout(x)
        x = F.normalize(x, dim=1)
        weight = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x, weight)


class FeatureRestorationBlock(nn.Module):
    """Recover weak local structure with an identity-initialized residual path."""

    def __init__(self, channels, hidden_channels):
        super().__init__()
        hidden_channels = max(int(hidden_channels), 16)
        self.pre_norm = nn.GroupNorm(1, channels)
        self.residual = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        final_projection = self.residual[-1]
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def forward(self, feature_map):
        residual = self.residual(self.pre_norm(feature_map))
        gate = self.gate(feature_map)
        return feature_map + gate * residual, residual, gate


class LocalFingerprintMNet(nn.Module):
    """Local channel-time fingerprint selection with environment adversarial head."""

    def __init__(
        self,
        in_channels=128,
        hidden_channels=64,
        num_classes=90,
        env_classes=21,
        grl_alpha=1.0,
        mask_min=0.10,
        mask_max=0.40,
        tv_weight=0.1,
        fusion_mode="fingerprint",
        use_rest_adv=False,
        use_rest_probe=False,
        use_rest_projector=False,
        use_multiscale=False,
        use_global_head=False,
        use_cosine_head=False,
        cosine_scale=16.0,
        use_feature_restorer=False,
    ):
        super().__init__()
        if fusion_mode not in {"fingerprint", "concat"}:
            raise ValueError("fusion_mode must be 'fingerprint' or 'concat'.")
        self.grl_alpha = grl_alpha
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.tv_weight = tv_weight
        self.fusion_mode = fusion_mode
        self.use_rest_adv = use_rest_adv
        self.use_rest_probe = use_rest_probe
        self.use_rest_projector = use_rest_projector
        self.use_multiscale = use_multiscale
        self.use_global_head = use_global_head
        self.use_cosine_head = use_cosine_head
        self.use_feature_restorer = use_feature_restorer
        id_dim = in_channels * 2 if fusion_mode == "concat" else in_channels
        mid_channels = max(hidden_channels // 2, 16)
        self.feature_restorer = (
            FeatureRestorationBlock(in_channels, hidden_channels)
            if use_feature_restorer else None
        )
        self.mnet = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.id_head = (
            CosineClassifier(id_dim, num_classes, scale=cosine_scale)
            if use_cosine_head else
            nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(id_dim, num_classes),
            )
        )
        self.global_id_head = (
            (
                CosineClassifier(in_channels, num_classes, scale=cosine_scale)
                if use_cosine_head else
                nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(in_channels, num_classes),
                )
            )
            if use_global_head else None
        )
        self.env_head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, env_classes),
        )
        if use_rest_projector:
            self.rest_projector = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_channels, in_channels),
                nn.LayerNorm(in_channels),
            )
        else:
            self.rest_projector = nn.Identity()
        if use_rest_adv or use_rest_probe:
            self.rest_id_head = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, num_classes),
            )
        else:
            self.rest_id_head = None
        if use_multiscale:
            self.scale_gate = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, 3),
            )
        else:
            self.scale_gate = None

    @staticmethod
    def _ensure_map(features):
        if features.ndim == 2:
            return features.unsqueeze(-1)
        if features.ndim != 3:
            raise ValueError("LocalFingerprintMNet expects [B, C] or [B, C, T] features.")
        return features

    def weighted_pool(self, feature_map, mask):
        weighted = feature_map * mask
        return weighted.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1e-6)

    def multiscale_fingerprint(self, feature_map, mask):
        if not self.use_multiscale:
            return self.weighted_pool(feature_map, mask), None

        smooth_3 = F.avg_pool1d(feature_map, kernel_size=3, stride=1, padding=1)
        smooth_5 = F.avg_pool1d(feature_map, kernel_size=5, stride=1, padding=2)
        scale_features = torch.stack([
            self.weighted_pool(feature_map, mask),
            self.weighted_pool(smooth_3, mask),
            self.weighted_pool(smooth_5, mask),
        ], dim=1)
        gate = F.softmax(self.scale_gate(feature_map.mean(dim=-1)), dim=1)
        fingerprint = (scale_features * gate.unsqueeze(-1)).sum(dim=1)
        return fingerprint, gate

    def rest_pool(self, feature_map, mask):
        rest_mask = 1.0 - mask
        weighted = feature_map * rest_mask
        return weighted.sum(dim=-1) / rest_mask.sum(dim=-1).clamp_min(1e-6)

    def mask_regularization(self, mask):
        mean_mask = mask.mean()
        area_loss = torch.relu(mask.new_tensor(self.mask_min) - mean_mask).pow(2)
        area_loss = area_loss + torch.relu(mean_mask - mask.new_tensor(self.mask_max)).pow(2)
        if mask.shape[-1] > 1:
            tv_loss = torch.abs(mask[..., 1:] - mask[..., :-1]).mean()
        else:
            tv_loss = mask.new_tensor(0.0)
        return area_loss + self.tv_weight * tv_loss

    def frozen_rest_head(self, features):
        if self.rest_id_head is None:
            return None
        output = features
        for module in self.rest_id_head:
            if isinstance(module, nn.Linear):
                bias = None if module.bias is None else module.bias.detach()
                output = F.linear(output, module.weight.detach(), bias)
            else:
                output = module(output)
        return output

    def forward(self, features, return_all=False):
        raw_feature_map = self._ensure_map(features)
        restoration_delta = None
        restoration_gate = None
        if self.feature_restorer is not None:
            feature_map, restoration_delta, restoration_gate = self.feature_restorer(
                raw_feature_map
            )
        else:
            feature_map = raw_feature_map
        mask = self.mnet(feature_map)
        h_avg = feature_map.mean(dim=-1)
        fingerprint, scale_gate = self.multiscale_fingerprint(feature_map, mask)
        rest_raw = self.rest_pool(feature_map, mask)
        rest = self.rest_projector(rest_raw)
        if self.fusion_mode == "concat":
            id_features = torch.cat([h_avg, fingerprint], dim=1)
        else:
            id_features = fingerprint
        id_logits = self.id_head(id_features)
        global_id_logits = self.global_id_head(h_avg) if self.global_id_head is not None else None
        adv_features = GradientReversal.apply(fingerprint, self.grl_alpha)
        env_logits = self.env_head(adv_features)
        rest_id_logits = None
        rest_probe_logits = None
        rest_uniform_logits = None
        if self.rest_id_head is not None:
            rest_features = GradientReversal.apply(rest, self.grl_alpha) if self.use_rest_adv else rest
            rest_id_logits = self.rest_id_head(rest_features)
            if self.use_rest_probe:
                rest_probe_logits = self.rest_id_head(rest.detach())
                rest_uniform_logits = self.frozen_rest_head(rest)
        if not return_all:
            return fingerprint, mask, env_logits
        return {
            "raw_feature_map": raw_feature_map,
            "feature_map": feature_map,
            "restoration_delta": restoration_delta,
            "restoration_gate": restoration_gate,
            "h_avg": h_avg,
            "fingerprint": fingerprint,
            "z_rest": rest,
            "z_rest_raw": rest_raw,
            "id_features": id_features,
            "scale_gate": scale_gate,
            "mask": mask,
            "id_logits": id_logits,
            "global_id_logits": global_id_logits,
            "env_logits": env_logits,
            "rest_id_logits": rest_id_logits,
            "rest_probe_logits": rest_probe_logits,
            "rest_uniform_logits": rest_uniform_logits,
            "mask_loss": self.mask_regularization(mask),
        }
