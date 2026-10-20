import torch
import torch.nn as nn

class UFFR(nn.Module):
    """
    Underwater Frequency-aware Feature Reweighting (UFFR)
    Đánh trọng số dải tần số thấp/cao trên miền Fourier[cite: 36, 41].
    """
    def __init__(self, channels, freq_radius=0.6, use_learnable=True):
        super().__init__()
        self.freq_radius = freq_radius
        if use_learnable:
            self.alpha_low = nn.Parameter(torch.tensor(0.95))
            self.alpha_high = nn.Parameter(torch.tensor(1.55))
        else:
            self.alpha_low = 0.95
            self.alpha_high = 1.55

    def forward(self, x):
        # Bóc tách an toàn nếu x là NestedTensor hoặc tuple/list
        if hasattr(x, "tensors") and hasattr(x, "mask"):
            x_tensor = x.tensors
            mask = x.mask
            is_nested = True
            is_list = False
        elif isinstance(x, (list, tuple)):
            x_tensor = x[0]
            is_nested = False
            is_list = True
        else:
            x_tensor = x
            is_nested = False
            is_list = False

        B, C, H, W = x_tensor.shape
        fft_x = torch.fft.fft2(x_tensor, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))
        
        u = torch.linspace(-1, 1, H, device=x_tensor.device)
        v = torch.linspace(-1, 1, W, device=x_tensor.device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
        radius = torch.sqrt(grid_u**2 + grid_v**2)
        
        low_mask = (radius < self.freq_radius).float()
        high_mask = 1.0 - low_mask
        mask_weight = self.alpha_low * low_mask + self.alpha_high * high_mask
        
        fft_shifted_weighted = fft_shifted * mask_weight.unsqueeze(0).unsqueeze(0)
        ifft_shifted = torch.fft.ifftshift(fft_shifted_weighted, dim=(-2, -1))
        out_tensor = torch.fft.ifft2(ifft_shifted, norm="ortho").real

        if is_nested:
            class StandardNestedTensor:
                def __init__(self, tensors, mask):
                    self.tensors = tensors
                    self.mask = mask
                    self.no_padding = None
                def decompose(self):
                    return self.tensors, self.mask
            return StandardNestedTensor(out_tensor, mask)
        elif is_list:
            return (out_tensor,) + tuple(x[1:])
        else:
            return out_tensor