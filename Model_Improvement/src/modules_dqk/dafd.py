"""DAFD equations 1-10: overlapping Gaussian bands, FiLM gates and spatial fusion."""
import torch
from torch import nn


class DAFD(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.centers = nn.Parameter(torch.logit((torch.tensor([0.10, 0.35, 0.60])-0.05)/0.6))
        self.widths = nn.Parameter(torch.logit((torch.tensor([0.12, 0.18, 0.25])-0.05)/0.3))
        self.scene_encoders = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels//4), 1), nn.ReLU(),
            nn.Conv2d(max(1, channels//4), 2*channels, 1)) for _ in range(3)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, channels, 1), nn.ReLU(),
            nn.Conv2d(channels, channels, 1)) for _ in range(3)])
        for encoder, gate in zip(self.scene_encoders, self.gates):
            nn.init.zeros_(encoder[-1].weight)
            nn.init.zeros_(encoder[-1].bias)
            with torch.no_grad():
                encoder[-1].bias[:channels].fill_(1.)
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, torch.logit(torch.tensor(0.8)).item())
        self.fusion = nn.Conv2d(3*channels, channels, 1)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        self.alpha = 0.15

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("DAFD expects BCHW; channel repetition/truncation is forbidden")
        with torch.autocast(device_type=x.device.type, enabled=False):
            h, w = x.shape[-2:]
            spectrum = torch.fft.rfft2(x.float(), norm="ortho")
            u = torch.fft.fftfreq(h, device=x.device)
            v = torch.fft.rfftfreq(w, device=x.device)
            radius = torch.sqrt(u[:, None]**2 + v[None, :]**2)
            centers = 0.6*self.centers.sigmoid()+0.05
            widths = 0.3*self.widths.sigmoid()+0.05
            bands, statistics = [], []
            for k in range(3):
                filtered = spectrum * torch.exp(-0.5*((radius-centers[k])/(widths[k]+1e-7))**2)
                amplitude = torch.log1p(filtered.abs())
                gamma, beta = self.scene_encoders[k](amplitude).chunk(2, dim=1)
                gate = torch.sigmoid(self.gates[k](amplitude)*gamma+beta)
                statistics.append(gate.mean())
                bands.append(torch.fft.irfft2(filtered*gate, s=(h, w), norm="ortho"))
            result = x.float() + self.alpha*self.fusion(torch.cat(bands, dim=1))
        return result.to(x.dtype), statistics
