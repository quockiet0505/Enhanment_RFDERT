"""FFT-bin-correct UFFR; Gaussian is the DQK assignment variant."""
import torch
from torch import nn


class UFFR(nn.Module):
    def __init__(self, channels, freq_radius=0.6, use_learnable=True, mask_kind="gaussian"):
        super().__init__()
        if freq_radius <= 0 or mask_kind not in {"gaussian", "binary"}:
            raise ValueError("Invalid UFFR mask configuration")
        self.channels, self.freq_radius, self.mask_kind = channels, freq_radius, mask_kind
        for name, value in [("alpha_low", 0.95), ("alpha_high", 1.55)]:
            if use_learnable:
                self.register_parameter(name, nn.Parameter(torch.tensor(value)))
            else:
                self.register_buffer(name, torch.tensor(value))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("UFFR expects BCHW with its configured channel count")
        with torch.autocast(device_type=x.device.type, enabled=False):
            h, w = x.shape[-2:]
            # Normalized Nyquist coordinates, using actual bins (even/odd safe).
            u = 2 * torch.fft.fftshift(torch.fft.fftfreq(h, device=x.device))
            v = 2 * torch.fft.fftshift(torch.fft.fftfreq(w, device=x.device))
            radius = torch.sqrt(u[:, None]**2 + v[None, :]**2)
            low = (radius < self.freq_radius).float() if self.mask_kind == "binary" else torch.exp(-0.5 * (radius / self.freq_radius)**2)
            spectrum = torch.fft.fftshift(torch.fft.fft2(x.float(), norm="ortho"), dim=(-2, -1))
            weight = self.alpha_low * low + self.alpha_high * (1-low)
            result = torch.fft.ifft2(torch.fft.ifftshift(spectrum * weight, dim=(-2, -1)), norm="ortho")
        return result.real.to(x.dtype)
