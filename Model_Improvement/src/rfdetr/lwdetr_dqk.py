import torch
import torch.nn as nn
from rfdetr import RFDETRMedium
from src.modules_dqk.darm import DARM, compute_tv_loss
from src.modules_dqk.uffr import UFFR
from src.modules_dqk.dafd import DAFD
from src.modules_dqk.mfa import MFA

class RFDETRDQKEnhanced(nn.Module):
    def __init__(self, num_classes=32, resolution=640, *args, **kwargs):
        super().__init__()
        self.base_model = RFDETRMedium(num_classes=num_classes, resolution=resolution)
        
        # Trỏ chính xác đến tầng mạng lõi thực thi bên trong
        self.net = self.base_model
        while hasattr(self.net, "model") and self.net.model is not self.net:
            self.net = self.net.model

        self.darm = DARM(in_channels=3)
        self.hidden_dim = 256
        
        self.dafd_modules = nn.ModuleList([DAFD(self.hidden_dim) for _ in range(3)])
        self.uffr_modules = nn.ModuleList([UFFR(self.hidden_dim) for _ in range(3)])
        self.mfa_modules = nn.ModuleList([MFA(self.hidden_dim) for _ in range(3)])
        self.current_tv_loss = 0.0

    def forward(self, x):
        if isinstance(x, dict) and "samples" in x:
            samples = x["samples"]
        else:
            samples = x
            
        if hasattr(samples, "tensors"):
            img_tensor = samples.tensors
            mask = getattr(samples, "mask", None)
        else:
            img_tensor = samples
            mask = None

        # 1. Khôi phục ảnh đầu vào qua DARM
        restored_img, R_map = self.darm(img_tensor)
        self.current_tv_loss = compute_tv_loss(R_map)
        
        if mask is None:
            mask = torch.zeros((restored_img.shape[0], restored_img.shape[2], restored_img.shape[3]), 
                               dtype=torch.bool, device=restored_img.device)

        class StandardNestedTensor:
            def __init__(self, tensors, mask):
                self.tensors = tensors
                self.mask = mask
                self.no_padding = None
            def decompose(self):
                return self.tensors, self.mask

        restored_samples = StandardNestedTensor(restored_img, mask)

        # 2. Gọi hàm thực thi bên trong `net` hoặc qua phương thức forward chuẩn
        if hasattr(self.net, "forward") and callable(self.net.forward) and self.net is not self:
            if isinstance(x, dict):
                x_input = x.copy()
                x_input["samples"] = restored_samples
                out = self.net(x_input)
            else:
                out = self.net(restored_samples)
        else:
            out = self.base_model.predict(restored_img) if hasattr(self.base_model, "predict") else restored_img

        # 3. Gắn thêm các thông số loss phụ trợ DQK
        if isinstance(out, dict):
            out["loss_tv"] = self.current_tv_loss
            out["loss_sparsity"] = torch.tensor(0.0, device=img_tensor.device)
        else:
            # Nếu đầu ra không phải dict (ví dụ đang ở mode inference thuần), trả về dạng chuẩn
            out = {"preds": out, "loss_tv": self.current_tv_loss, "loss_sparsity": torch.tensor(0.0, device=img_tensor.device)}
            
        return out