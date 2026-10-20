import torch
import torch.nn as nn

class MFA(nn.Module):
    """
    Multi-Scale Frequency-Aware Attention (MFA)
    Tinh chỉnh đa tỷ lệ không gian kết hợp xấp xỉ tần số[cite: 36, 42, 43].
    """
    def __init__(self, channels, alpha_freq=0.75, dropout=0.15):
        super().__init__()
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.conv9 = nn.Conv2d(channels, channels, kernel_size=9, padding=4, groups=channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.alpha_freq = alpha_freq
        self.gate_conv = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout2d(dropout)
        self.gamma = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x):
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

        f1 = self.conv5(x_tensor)
        f2 = self.conv7(x_tensor)
        f3 = self.conv9(x_tensor)
        x_ms = (f1 + f2 + f3) / 3.0
        
        x_low = self.avg_pool(x_ms)
        x_high = x_ms - x_low
        x_freq = x_low + self.alpha_freq * x_high
        
        gate = torch.sigmoid(self.gate_conv(x_freq))
        x_gated = x_freq * gate
        x_out = self.dropout(x_gated)
        out_tensor = x_tensor + self.gamma * x_out

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