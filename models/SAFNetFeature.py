import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if not (sys.path[0] == "." or sys.path[0] == "models" or sys.path[0] == os.path.abspath("models")):
    from pathlib import Path
    sys.path.append(str(Path(__file__).absolute().parent))

from models.CVTSLANetFeature import PatchEmbed, TSLANet_layer, trunc_normal


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleEnhance(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        branch_channels = channels // 4
        self.branches = nn.ModuleList([
            ConvBNAct(channels, branch_channels, 3),
            ConvBNAct(channels, branch_channels, 5),
            ConvBNAct(channels, branch_channels, 9),
            ConvBNAct(channels, branch_channels, 3, dilation=2),
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_channels * 4, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class SNRQualityGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels * 2 + 3, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def quality_stats(signal):
        signal = signal.float()
        magnitude = torch.sqrt(signal[:, 0].square() + signal[:, 1].square() + 1e-8)
        mean = magnitude.mean(dim=-1, keepdim=True)
        std = magnitude.std(dim=-1, keepdim=True)
        smooth = F.avg_pool1d(magnitude.unsqueeze(1), kernel_size=9, stride=1, padding=4).squeeze(1)
        residual = (magnitude - smooth).square().mean(dim=-1, keepdim=True)
        residual_ratio = residual / magnitude.square().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.cat([mean, std, residual_ratio], dim=1)

    def forward(self, signal, temporal_map, enhancement_map):
        temporal_summary = temporal_map.mean(dim=-1)
        enhancement_summary = enhancement_map.mean(dim=-1)
        stats = self.quality_stats(signal).to(temporal_summary.dtype)
        return self.gate(torch.cat([temporal_summary, enhancement_summary, stats], dim=1))


class SAFNet(nn.Module):
    """SNR-aware temporal and time-frequency fingerprint encoder."""

    def __init__(
        self,
        seq_len=256,
        patch_size=16,
        num_channels=2,
        emb_dim=128,
        depth=3,
        num_classes=1024,
        dropout_rate=0.3,
    ):
        super().__init__()
        if num_channels != 2:
            raise ValueError("SAFNet expects IQ input with 2 channels.")
        stride = max(patch_size // 2, 1)
        self.num_patches = int((seq_len - patch_size) / stride + 1)
        self.depth = depth

        self.patch_embed = PatchEmbed(
            seq_len=seq_len,
            patch_size=patch_size,
            in_chans=num_channels,
            embed_dim=emb_dim,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        self.pos_drop = nn.Dropout(dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]
        self.temporal_blocks = nn.ModuleList([
            TSLANet_layer(dim=emb_dim, num_patch=self.num_patches, drop=dropout_rate, drop_path=dpr[i])
            for i in range(depth)
        ])

        self.freq_stem = nn.Sequential(
            ConvBNAct(2, emb_dim, 5),
            MultiScaleEnhance(emb_dim, dropout=dropout_rate * 0.5),
            nn.AdaptiveAvgPool1d(self.num_patches),
        )
        self.enhance_proj = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, 1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
        )
        self.gate = SNRQualityGate(emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim * 4, num_classes),
        )
        trunc_normal(self.pos_embed, std=0.02)

    @staticmethod
    def _complex_spectrum(x):
        complex_x = torch.complex(x[:, 0].float(), x[:, 1].float())
        spectrum = torch.fft.fft(complex_x, dim=-1, norm="ortho")
        spectrum = spectrum[..., : spectrum.shape[-1] // 2 + 1]
        magnitude = torch.log1p(torch.abs(spectrum))
        phase_cos = torch.cos(torch.angle(spectrum))
        return torch.stack([magnitude, phase_cos], dim=1).to(dtype=x.dtype)

    @staticmethod
    def cutmix_data(x, y, lam):
        batch_size, _, x_len = x.shape
        cut_len = min(max(int(x_len * lam), 1), x_len)
        start_index = 0 if cut_len == x_len else torch.randint(0, x_len - cut_len + 1, (1,)).item()
        end_index = start_index + cut_len
        batch_index = torch.randperm(batch_size).to(x.device)
        mixed_x = x.clone()
        mixed_x[:, :, start_index:end_index] = x[batch_index, :, start_index:end_index].clone()
        return mixed_x, y, y[batch_index]

    def temporal_forward_map(self, x):
        tokens = self.patch_embed(x.float())
        tokens = self.pos_drop(tokens + self.pos_embed)
        for block in self.temporal_blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2)

    def forward_map(self, x):
        temporal_map = self.temporal_forward_map(x)
        enhancement_map = self.enhance_proj(self.freq_stem(self._complex_spectrum(x)))
        alpha = self.gate(x, temporal_map, enhancement_map).unsqueeze(-1)
        fused_map = (1.0 - alpha) * temporal_map + alpha * enhancement_map
        tokens = self.norm(fused_map.transpose(1, 2))
        return tokens.transpose(1, 2)

    def _set_forward(self, x):
        feature_map = self.forward_map(x).transpose(1, 2)
        mean_pool = feature_map.mean(dim=1)
        max_pool = feature_map.max(dim=1).values
        return self.head(torch.cat([mean_pool, max_pool], dim=1))

    def forward(self, x, y=None, mixed_lambda=None):
        if y is not None or mixed_lambda is not None:
            assert y is not None and mixed_lambda is not None
            mixed_x, y_a, y_b = self.cutmix_data(x, y, mixed_lambda)
            return self._set_forward(mixed_x), y_a, y_b
        return self._set_forward(x)


def create_model(feature_dim=1024, dtype="iq", **kwargs):
    if "iq" not in dtype:
        raise ValueError("SAFNet is only used for IQ input.")
    return SAFNet(
        seq_len=kwargs.get("seq_len", 256),
        patch_size=kwargs.get("patch_size", 16),
        num_channels=kwargs.get("num_channels", 2),
        emb_dim=kwargs.get("emb_dim", 128),
        depth=kwargs.get("depth", 3),
        num_classes=feature_dim,
        dropout_rate=kwargs.get("dropout_rate", 0.3),
    )
