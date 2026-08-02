import torch

from models.WiSigCNNFeature import WiSigCNN
from utils.low_snr_baseline import add_awgn_torch


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
