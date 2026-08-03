import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MSFTFNetFeature import MSFTFNet


class QualityCalibratedScaleRouter(nn.Module):
    """Route scale/domain candidates using learned signal quality statistics."""

    def __init__(self, channels, kernel_sizes=(3, 7, 11), temperature=1.0):
        super().__init__()
        self.kernel_sizes = tuple(int(value) for value in kernel_sizes)
        if any(value < 1 or value % 2 == 0 for value in self.kernel_sizes):
            raise ValueError("Router kernel sizes must be positive odd integers.")
        if temperature <= 0:
            raise ValueError("Router temperature must be positive.")

        self.num_candidates = len(self.kernel_sizes) + 2
        self.temperature = float(temperature)
        hidden_dim = max(channels // 2, 32)
        statistic_dim = channels * 2

        self.quality_encoder = nn.Sequential(
            nn.LayerNorm(statistic_dim * self.num_candidates),
            nn.Linear(statistic_dim * self.num_candidates, channels),
            nn.GELU(),
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
        )
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.route_head = nn.Sequential(
            nn.Linear(statistic_dim + hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)
        nn.init.zeros_(self.route_head[-1].weight)
        nn.init.zeros_(self.route_head[-1].bias)

    @staticmethod
    def _statistics(feature_map):
        return torch.cat(
            [
                feature_map.mean(dim=-1),
                feature_map.std(dim=-1, unbiased=False),
            ],
            dim=1,
        )

    def _candidates(self, time_map, freq_map):
        candidates = [time_map]
        candidates.extend(
            F.avg_pool1d(
                time_map,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
            for kernel_size in self.kernel_sizes
        )
        candidates.append(freq_map)
        return candidates

    def forward(self, time_map, freq_map):
        candidates = self._candidates(time_map, freq_map)
        statistics = [self._statistics(candidate) for candidate in candidates]
        quality_context = self.quality_encoder(torch.cat(statistics, dim=1))
        quality = torch.sigmoid(self.quality_head(quality_context)).squeeze(1)

        route_logits = torch.cat(
            [
                self.route_head(
                    torch.cat(
                        [
                            statistic,
                            quality_context,
                            quality.unsqueeze(1),
                        ],
                        dim=1,
                    )
                )
                for statistic in statistics
            ],
            dim=1,
        )
        weights = F.softmax(route_logits / self.temperature, dim=1)
        fused = sum(
            weight.unsqueeze(1).unsqueeze(2) * candidate
            for weight, candidate in zip(weights.unbind(dim=1), candidates)
        )
        entropy = -(
            weights * weights.clamp_min(1e-8).log()
        ).sum(dim=1)
        return fused, quality, weights, entropy


class MSFTFNetQCRouter(MSFTFNet):
    """MSFTFNet with quality-calibrated scale and domain routing."""

    def __init__(self, *args, router_temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.qc_router = QualityCalibratedScaleRouter(
            self.emb_dim,
            temperature=router_temperature,
        )

    def forward_stages(self, x):
        time_map = self.time_stem(x.float())
        freq_map = self.freq_stem(self._complex_spectrum(x))
        time_map, freq_map, reliability = self.tf_enhancer(time_map, freq_map)
        feature_map, quality, route_weights, route_entropy = self.qc_router(
            time_map,
            freq_map,
        )
        tokens = feature_map.transpose(1, 2) + self.pos_embed
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return {
            "time_map": time_map,
            "freq_map": freq_map,
            "fused_map": feature_map,
            "feature_map": tokens.transpose(1, 2),
            "reliability": reliability,
            "quality": quality,
            "route_weights": route_weights,
            "route_entropy": route_entropy,
        }


def create_model(feature_dim=1024, dtype="iq", **kwargs):
    if "iq" not in dtype:
        raise ValueError("MSFTFNet-QCRouter is only used for IQ input.")
    return MSFTFNetQCRouter(
        seq_len=kwargs.get("seq_len", 256),
        patch_size=kwargs.get("patch_size", 16),
        num_channels=kwargs.get("num_channels", 2),
        emb_dim=kwargs.get("emb_dim", 128),
        depth=kwargs.get("depth", 3),
        num_classes=feature_dim,
        dropout_rate=kwargs.get("dropout_rate", 0.3),
        router_temperature=kwargs.get("router_temperature", 1.0),
    )
