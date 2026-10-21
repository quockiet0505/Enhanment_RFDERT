import torch
import torch.nn as nn

class DARM(nn.Module):
    """
    Degradation-Aware Restoration Module (DARM)
    Khôi phục chất lượng ảnh đầu vào bằng bù dư nhẹ (residual compensation)[cite: 25].
    """
    def __init__(self, in_channels=3, alpha_init=0.1):
        super(DARM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        # Start as identity when transferring a detector checkpoint.
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.act1(residual)
        residual = self.conv2(residual)
        residual = self.act2(residual)
        R = self.conv3(residual)
        out = x + self.alpha * R
        return out, R

def compute_tv_loss(R):
    """Tổng biến thiên (Total Variation) Regularization Loss áp dụng lên bản đồ R[cite: 27]."""
    batch_size, c, h, w = R.shape
    tv_h = torch.abs(R[:, :, 1:, :] - R[:, :, :-1, :]).sum()
    tv_w = torch.abs(R[:, :, :, 1:] - R[:, :, :, :-1]).sum()
    return (tv_h + tv_w) / (batch_size * c * h * w)