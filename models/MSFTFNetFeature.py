import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleTemporalBlock(nn.Module):
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
            nn.Conv1d(branch_channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return residual + self.fuse(x)


class TimeFrequencyFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 2),
        )

    def forward(self, time_map, freq_map):
        summary = torch.cat([time_map.mean(dim=-1), freq_map.mean(dim=-1)], dim=1)
        weights = F.softmax(self.gate(summary), dim=1)
        fused = weights[:, 0:1].unsqueeze(-1) * time_map + weights[:, 1:2].unsqueeze(-1) * freq_map
        return fused


class MSFTFNet(nn.Module):
    """Multi-scale time-frequency fingerprint encoder for IQ SEI signals."""

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
            raise ValueError("MSFTFNet expects IQ input with 2 channels.")
        stride = max(patch_size // 2, 1)
        self.num_patches = int((seq_len - patch_size) / stride + 1)
        self.emb_dim = emb_dim

        self.time_stem = nn.Sequential(
            ConvBNAct(num_channels, emb_dim, patch_size, stride=stride),
            MultiScaleTemporalBlock(emb_dim, dropout=dropout_rate * 0.5),
            MultiScaleTemporalBlock(emb_dim, dropout=dropout_rate * 0.5),
            nn.AdaptiveAvgPool1d(self.num_patches),
        )
        self.freq_stem = nn.Sequential(
            ConvBNAct(2, emb_dim, 5),
            MultiScaleTemporalBlock(emb_dim, dropout=dropout_rate * 0.5),
            nn.AdaptiveAvgPool1d(self.num_patches),
        )
        self.fusion = TimeFrequencyFusion(emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=4,
            dim_feedforward=emb_dim * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim * 4, num_classes),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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

    def forward_map(self, x):
        time_map = self.time_stem(x.float())
        freq_map = self.freq_stem(self._complex_spectrum(x))
        feature_map = self.fusion(time_map, freq_map)
        tokens = feature_map.transpose(1, 2) + self.pos_embed
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
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
        raise ValueError("MSFTFNet is only used for IQ input.")
    return MSFTFNet(
        seq_len=kwargs.get("seq_len", 256),
        patch_size=kwargs.get("patch_size", 16),
        num_channels=kwargs.get("num_channels", 2),
        emb_dim=kwargs.get("emb_dim", 128),
        depth=kwargs.get("depth", 3),
        num_classes=feature_dim,
        dropout_rate=kwargs.get("dropout_rate", 0.3),
    )
