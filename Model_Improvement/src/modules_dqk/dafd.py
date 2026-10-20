import torch
import torch.nn as nn
import torch.nn.functional as F

class DAFD(nn.Module):
    """
    Degradation-Aware Frequency Decomposition (DAFD)
    Tách 3 dải tần (Thấp, Trung, Cao) qua 2D Real FFT kết hợp FiLM và Gate.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.centers = nn.Parameter(torch.tensor([-2.19, -0.61, 0.40]))
        self.widths = nn.Parameter(torch.tensor([-1.99, -1.25, -0.69]))
        
        self.scene_encoders = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(),
                nn.Conv2d(channels // 4, channels * 2, 1)
            ) for _ in range(3)
        ])
        
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 1)
            ) for _ in range(3)
        ])
        
        for gate in self.gates:
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, 1.386)
            
        # Dùng Conv2d cố định chuẩn xác, KHÔNG dùng LazyConv2d để tránh lỗi numel()
        self.fusion = nn.Conv2d(channels * 3, channels, 1)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        self.alpha = 0.15

    def forward(self, x):
        if x is None:
            return x, [torch.tensor(0.0)]
            
        if isinstance(x, (list, tuple)):
            if len(x) == 0 or x[0] is None:
                return x, [torch.tensor(0.0)]
            x_tensor = x[0]
            is_list = True
            is_nested = False
        elif hasattr(x, "tensors") and hasattr(x, "mask"):
            x_tensor = x.tensors
            mask = x.mask
            is_list = False
            is_nested = True
        else:
            x_tensor = x
            is_list = False
            is_nested = False

        if x_tensor is None or not hasattr(x_tensor, "shape") or len(x_tensor.shape) != 4:
            return x, [torch.tensor(0.0)]

        # Lớp bảo vệ: Ép kênh bằng đúng self.channels nếu lỡ bị lệch
        orig_C = x_tensor.shape[1]
        if orig_C != self.channels:
            repeats = (self.channels // orig_C) + 1
            x_tensor = x_tensor.repeat(1, repeats, 1, 1)[:, :self.channels, :, :]

        B, C, H, W = x_tensor.shape
        F_x = torch.fft.rfft2(x_tensor, norm="ortho")
        
        u = torch.fft.fftfreq(H, device=x_tensor.device)
        v = torch.fft.rfftfreq(W, device=x_tensor.device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
        radius = torch.sqrt(grid_u**2 + grid_v**2)
        
        c_k = 0.6 * torch.sigmoid(self.centers) + 0.05
        w_k = 0.3 * torch.sigmoid(self.widths) + 0.05
        
        spatial_bands = []
        gate_stats = []
        
        for k in range(3):
            M_k = torch.exp(-0.5 * ((radius - c_k[k]) / (w_k[k] + 1e-7))**2)
            F_k = F_x * M_k.unsqueeze(0).unsqueeze(0)
            
            A_k = torch.log(1 + torch.abs(F_k))
            
            film_params = self.scene_encoders[k](A_k.real)
            gate_raw = self.gates[k](A_k.real)

            gamma, beta = film_params.chunk(2, dim=1)
            G_k = torch.sigmoid(gate_raw * gamma + beta)
            gate_stats.append(G_k.mean())
            
            F_k_gated = F_k * G_k
            f_k = torch.fft.irfft2(F_k_gated, s=(H, W), norm="ortho")
            spatial_bands.append(f_k)
            
        fused = self.fusion(torch.cat(spatial_bands, dim=1))
        out_tensor = x_tensor + self.alpha * fused
        
        if is_list:
            return out_tensor, gate_stats
        elif is_nested:
            class StandardNestedTensor:
                def __init__(self, tensors, mask):
                    self.tensors = tensors
                    self.mask = mask
                    self.no_padding = None
                def decompose(self):
                    return self.tensors, self.mask
            return StandardNestedTensor(out_tensor, mask), gate_stats
        else:
            return out_tensor, gate_stats