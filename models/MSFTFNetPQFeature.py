import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MSFTFNetFeature import MSFTFNet


class PhysicalQualityDescriptor(nn.Module):
    """Extract compact signal-quality priors without using ground-truth SNR."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = x.float()
        complex_x = torch.complex(x[:, 0], x[:, 1])

        spectrum = torch.fft.fft(complex_x, dim=-1, norm="ortho")
        power = spectrum.abs().square().clamp_min(self.eps)
        spectral_flatness = (
            power.log().mean(dim=-1).exp() / power.mean(dim=-1)
        ).clamp(0.0, 1.0)

        probability = power / power.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        spectral_entropy = -(
            probability * probability.clamp_min(self.eps).log()
        ).sum(dim=-1)
        spectral_entropy = (
            spectral_entropy / math.log(max(power.shape[-1], 2))
        ).clamp(0.0, 1.0)

        amplitude = complex_x.abs()
        centered_amplitude = amplitude - amplitude.mean(dim=-1, keepdim=True)
        amplitude_variance = centered_amplitude.square().mean(dim=-1)
        amplitude_kurtosis = centered_amplitude.pow(4).mean(dim=-1) / (
            amplitude_variance.square() + self.eps
        )

        second_moment = complex_x.square().mean(dim=-1)
        fourth_moment = complex_x.pow(4).mean(dim=-1)
        average_power = complex_x.abs().square().mean(dim=-1)
        normalized_cumulant = (
            fourth_moment - 3.0 * second_moment.square()
        ).abs() / (average_power.square() + self.eps)

        return torch.stack(
            [
                spectral_flatness,
                spectral_entropy,
                amplitude_kurtosis.clamp_max(100.0).log1p(),
                normalized_cumulant.clamp_max(100.0).log1p(),
            ],
            dim=1,
        )


class PhysicalQualityFusion(nn.Module):
    """Condition time-frequency modulation and fusion on physical quality."""

    def __init__(self, channels, quality_dim=4):
        super().__init__()
        hidden_dim = max(channels // 2, 32)
        self.quality_encoder = nn.Sequential(
            nn.LayerNorm(quality_dim),
            nn.Linear(quality_dim, hidden_dim),
            nn.GELU(),
        )
        self.modulator = nn.Linear(hidden_dim, channels * 4)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2 + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * 2),
        )
        self.channels = channels

        nn.init.zeros_(self.modulator.weight)
        nn.init.zeros_(self.modulator.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _modulate(feature_map, scale, shift):
        scale = 0.1 * torch.tanh(scale).unsqueeze(-1)
        shift = 0.1 * torch.tanh(shift).unsqueeze(-1)
        return feature_map * (1.0 + scale) + shift

    def forward(self, time_map, freq_map, quality):
        quality_embedding = self.quality_encoder(quality)
        time_scale, time_shift, freq_scale, freq_shift = self.modulator(
            quality_embedding
        ).chunk(4, dim=1)

        time_map = self._modulate(time_map, time_scale, time_shift)
        freq_map = self._modulate(freq_map, freq_scale, freq_shift)
        summary = torch.cat(
            [
                time_map.mean(dim=-1),
                freq_map.mean(dim=-1),
                quality_embedding,
            ],
            dim=1,
        )
        weights = self.gate(summary).view(-1, 2, self.channels)
        weights = F.softmax(weights, dim=1)
        return (
            weights[:, 0].unsqueeze(-1) * time_map
            + weights[:, 1].unsqueeze(-1) * freq_map
        )


class MSFTFNetPQ(MSFTFNet):
    """MSFTFNet with physical-quality-conditioned time-frequency fusion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_descriptor = PhysicalQualityDescriptor()
        self.fusion = PhysicalQualityFusion(self.emb_dim)

    def forward_map(self, x):
        x = x.float()
        time_map = self.time_stem(x)
        freq_map = self.freq_stem(self._complex_spectrum(x))
        quality = self.quality_descriptor(x)
        feature_map = self.fusion(time_map, freq_map, quality)
        tokens = feature_map.transpose(1, 2) + self.pos_embed
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2)


def create_model(feature_dim=1024, dtype="iq", **kwargs):
    if "iq" not in dtype:
        raise ValueError("MSFTFNet-PQ is only used for IQ input.")
    return MSFTFNetPQ(
        seq_len=kwargs.get("seq_len", 256),
        patch_size=kwargs.get("patch_size", 16),
        num_channels=kwargs.get("num_channels", 2),
        emb_dim=kwargs.get("emb_dim", 128),
        depth=kwargs.get("depth", 3),
        num_classes=feature_dim,
        dropout_rate=kwargs.get("dropout_rate", 0.3),
    )
