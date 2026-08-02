import torch

from models.WiSigCNNFeature import WiSigCNN
from utils.low_snr_baseline import (
    BASELINE_SPECS,
    PooledEncoderClassifier,
    add_awgn_torch,
)


def test_wisig_cnn_feature_shape():
    model = WiSigCNN()
    output = model(torch.randn(4, 2, 256))
    assert output.shape == (4, 80)


def test_awgn_matches_requested_per_channel_snr():
    torch.manual_seed(7)
    inputs = torch.randn(64, 2, 4096)
    noisy = add_awgn_torch(inputs, -5.0)
    noise = noisy - inputs
    signal_power = inputs.square().mean(dim=-1)
    noise_power = noise.square().mean(dim=-1)
    measured = 10.0 * torch.log10(signal_power / noise_power)
    assert torch.allclose(measured.mean(), torch.tensor(-5.0), atol=0.05)


def test_main_online_awgn_baselines_keep_clean_pairs():
    assert (
        BASELINE_SPECS["MSFTFNet-OnlineAWGN-Paired"]["augmentation"]
        == "paired_online_awgn"
    )


def test_pooled_classifier_bypasses_projection_head():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection_called = False

        def forward_map(self, inputs):
            return inputs

        def forward(self, inputs):
            self.projection_called = True
            return inputs.mean(dim=-1)

    encoder = Encoder()
    model = PooledEncoderClassifier(encoder, num_classes=3, map_channels=2)
    output = model(torch.randn(4, 2, 16))
    assert output.shape == (4, 3)
    assert not encoder.projection_called
    assert (
        BASELINE_SPECS["WiSigCNN-OnlineAWGN"]["augmentation"]
        == "paired_online_awgn"
    )
