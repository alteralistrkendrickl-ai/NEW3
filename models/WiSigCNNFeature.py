import torch
from torch import nn


class WiSigCNN(nn.Module):
    """PyTorch feature extractor matching the official WiSig ManyTx CNN."""

    feature_dim = 80

    def __init__(self, seq_len=256, num_channels=2, dropout_rate=0.5, **_):
        super().__init__()
        if seq_len != 256 or num_channels != 2:
            raise ValueError("WiSigCNN expects 256-sample, two-channel IQ input.")
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 2), padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(8, 16, kernel_size=(3, 2), padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 16, kernel_size=(3, 2), padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 16, kernel_size=(3, 1), padding="same"),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, 256):
            raise ValueError(
                f"WiSigCNN expects [batch, 2, 256], received {tuple(x.shape)}."
            )
        # Official Keras input is [batch, 256, 2, 1].
        x = x.transpose(1, 2).unsqueeze(1)
        return self.projection(self.features(x))


def create_model(feature_dim=80, dtype="iq", **kwargs):
    if "iq" not in dtype:
        raise ValueError("WiSigCNN is only used for IQ input.")
    if feature_dim not in (80, None):
        raise ValueError("The official WiSigCNN feature dimension is fixed at 80.")
    return WiSigCNN(
        seq_len=kwargs.get("seq_len", 256),
        num_channels=kwargs.get("num_channels", 2),
        dropout_rate=kwargs.get("dropout_rate", 0.5),
    )
