import torch


SNR_LEVELS = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
SNR_BINS = (10.0, 20.0)
FADING_NONE = 0
FADING_RAYLEIGH = 1
FADING_RICIAN = 2


def add_random_awgn(signal, snr_range=(0.0, 30.0)):
    """Add sample-wise AWGN with SNR drawn uniformly from ``snr_range``."""
    min_snr, max_snr = map(float, snr_range)
    if min_snr > max_snr:
        raise ValueError("The minimum SNR cannot exceed the maximum SNR.")
    snr = torch.empty(signal.shape[0], device=signal.device, dtype=signal.dtype)
    snr.uniform_(min_snr, max_snr)
    return add_awgn(signal, snr)


def add_awgn(signal, snr):
    """Add AWGN to each sample while preserving its independently measured power."""
    if signal.ndim < 2:
        raise ValueError("signal must include a batch dimension and at least one feature dimension")

    reduce_dims = tuple(range(1, signal.ndim))
    signal_power = torch.mean(signal.square(), dim=reduce_dims, keepdim=True)
    snr = torch.as_tensor(snr, device=signal.device, dtype=signal.dtype)
    if snr.ndim == 0:
        snr = snr.repeat(signal.shape[0])
    if snr.numel() != signal.shape[0]:
        raise ValueError("snr must be a scalar or contain one value per sample")
    snr = snr.reshape(-1, *([1] * (signal.ndim - 1)))
    noise_power = signal_power / torch.pow(signal.new_tensor(10.0), snr / 10.0)
    return signal + torch.randn_like(signal) * torch.sqrt(noise_power.clamp_min(1e-12))


def snr_to_class(snr):
    """Map continuous SNR values to coarse channel labels."""
    labels = torch.zeros_like(snr, dtype=torch.long)
    labels = labels + (snr >= SNR_BINS[0]).long()
    labels = labels + (snr >= SNR_BINS[1]).long()
    return labels


def snr_level_to_class(snr, levels=SNR_LEVELS):
    """Map SNR values to the nearest discrete CoDiFA SNR level index."""
    level_tensor = torch.as_tensor(levels, device=snr.device, dtype=snr.dtype)
    distances = torch.abs(snr.reshape(-1, 1) - level_tensor.reshape(1, -1))
    return torch.argmin(distances, dim=1).long()


def _complex_scale(signal, real, imag):
    i_part = signal[:, 0]
    q_part = signal[:, 1]
    scaled = torch.empty_like(signal)
    scaled[:, 0] = real * i_part - imag * q_part
    scaled[:, 1] = imag * i_part + real * q_part
    return scaled


def _apply_rayleigh(signal):
    batch = signal.shape[0]
    real = torch.randn(batch, 1, device=signal.device, dtype=signal.dtype)
    imag = torch.randn(batch, 1, device=signal.device, dtype=signal.dtype)
    scale = signal.new_tensor(0.5).sqrt()
    return _complex_scale(signal, real * scale, imag * scale)


def _apply_rician(signal, k_factor=3.0):
    batch = signal.shape[0]
    k = signal.new_tensor(float(k_factor))
    los = torch.sqrt(k / (k + 1.0))
    scatter = torch.sqrt(1.0 / (2.0 * (k + 1.0)))
    real = los + torch.randn(batch, 1, device=signal.device, dtype=signal.dtype) * scatter
    imag = torch.randn(batch, 1, device=signal.device, dtype=signal.dtype) * scatter
    return _complex_scale(signal, real, imag)


def _apply_multipath(signal, max_delay=8):
    batch, _, length = signal.shape
    delay = torch.randint(1, max_delay + 1, (batch,), device=signal.device)
    gain = torch.empty(batch, 1, 1, device=signal.device, dtype=signal.dtype).uniform_(0.05, 0.25)
    delayed = torch.zeros_like(signal)
    for item in range(batch):
        d = int(delay[item].item())
        delayed[item, :, d:] = signal[item, :, : length - d]
    return signal + gain * delayed


def _apply_random_phase_and_amplitude(signal):
    batch = signal.shape[0]
    phase = torch.empty(batch, 1, device=signal.device, dtype=signal.dtype).uniform_(-0.25, 0.25)
    amplitude = torch.empty(batch, 1, 1, device=signal.device, dtype=signal.dtype).uniform_(0.8, 1.2)
    rotated = _complex_scale(signal, torch.cos(phase), torch.sin(phase))
    return rotated * amplitude


def random_joint_interference_view(signal, snr_levels=SNR_LEVELS, enable_awgn=True):
    """Create a CoDiFA channel-noise joint interference view.

    The returned labels supervise the interference branch: fading type and
    discrete SNR level. Fingerprint supervision still uses the device label.
    """
    if signal.ndim != 3 or signal.shape[1] != 2:
        raise ValueError("joint interference views expect IQ tensors shaped [batch, 2, length]")

    batch = signal.shape[0]
    output = _apply_random_phase_and_amplitude(signal)
    fading_labels = torch.randint(0, 3, (batch,), device=signal.device)

    if (fading_labels == FADING_RAYLEIGH).any():
        mask = fading_labels == FADING_RAYLEIGH
        output = output.clone()
        output[mask] = _apply_rayleigh(output[mask])
    if (fading_labels == FADING_RICIAN).any():
        mask = fading_labels == FADING_RICIAN
        output = output.clone()
        output[mask] = _apply_rician(output[mask])

    if batch > 0:
        multipath_mask = torch.rand(batch, device=signal.device) < 0.5
        if multipath_mask.any():
            output = output.clone()
            output[multipath_mask] = _apply_multipath(output[multipath_mask])

    level_tensor = torch.as_tensor(snr_levels, device=signal.device, dtype=signal.dtype)
    snr_indices = torch.randint(0, len(snr_levels), (batch,), device=signal.device)
    snr = level_tensor[snr_indices]
    if enable_awgn:
        output = add_awgn(output, snr)
    return output, snr_indices.long(), fading_labels.long()


def random_channel_view(signal, snr_range=(0.0, 30.0), enable_awgn=True):
    """Create one stochastic channel view and return free channel labels."""
    if signal.ndim != 3 or signal.shape[1] != 2:
        raise ValueError("channel views expect IQ tensors shaped [batch, 2, length]")

    batch = signal.shape[0]
    output = signal
    fading_labels = torch.randint(0, 3, (batch,), device=signal.device)

    if (fading_labels == FADING_RAYLEIGH).any():
        mask = fading_labels == FADING_RAYLEIGH
        output = output.clone()
        output[mask] = _apply_rayleigh(output[mask])
    if (fading_labels == FADING_RICIAN).any():
        mask = fading_labels == FADING_RICIAN
        output = output.clone()
        output[mask] = _apply_rician(output[mask])

    if batch > 0:
        multipath_mask = torch.rand(batch, device=signal.device) < 0.5
        if multipath_mask.any():
            output = output.clone()
            output[multipath_mask] = _apply_multipath(output[multipath_mask])

    min_snr, max_snr = map(float, snr_range)
    snr = torch.empty(batch, device=signal.device, dtype=signal.dtype).uniform_(min_snr, max_snr)
    if enable_awgn:
        output = add_awgn(output, snr)
    return output, snr_to_class(snr), fading_labels.long()
