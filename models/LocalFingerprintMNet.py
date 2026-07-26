import torch
import torch.nn as nn

from models.lfdb import GradientReversal


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
        id_dim = in_channels * 2 if fusion_mode == "concat" else in_channels
        mid_channels = max(hidden_channels // 2, 16)
        self.mnet = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.id_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(id_dim, num_classes),
        )
        self.env_head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, env_classes),
        )
        if use_rest_adv:
            self.rest_id_head = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, num_classes),
            )
        else:
            self.rest_id_head = None

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

    def forward(self, features, return_all=False):
        feature_map = self._ensure_map(features)
        mask = self.mnet(feature_map)
        h_avg = feature_map.mean(dim=-1)
        fingerprint = self.weighted_pool(feature_map, mask)
        rest = self.rest_pool(feature_map, mask)
        if self.fusion_mode == "concat":
            id_features = torch.cat([h_avg, fingerprint], dim=1)
        else:
            id_features = fingerprint
        id_logits = self.id_head(id_features)
        adv_features = GradientReversal.apply(fingerprint, self.grl_alpha)
        env_logits = self.env_head(adv_features)
        rest_id_logits = None
        if self.rest_id_head is not None:
            rest_adv_features = GradientReversal.apply(rest, self.grl_alpha)
            rest_id_logits = self.rest_id_head(rest_adv_features)
        if not return_all:
            return fingerprint, mask, env_logits
        return {
            "feature_map": feature_map,
            "h_avg": h_avg,
            "fingerprint": fingerprint,
            "z_rest": rest,
            "id_features": id_features,
            "mask": mask,
            "id_logits": id_logits,
            "env_logits": env_logits,
            "rest_id_logits": rest_id_logits,
            "mask_loss": self.mask_regularization(mask),
        }
