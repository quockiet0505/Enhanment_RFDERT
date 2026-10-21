"""DQK v1 core: DARM -> backbone taps (DAFD -> UFFR) -> projector -> MFA.

The combined order is a DQK design, not a verbatim architecture from one paper.
RF-DETR 1.10.1 owns the decoder, matcher, optimizer groups and EMA lifecycle.
"""
import copy
import torch
from torch import nn
from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list
from src.modules_dqk import DARM, compute_tv_loss, DAFD, UFFR, MFA

ARCHITECTURE = "dqk-four-modules-v1"
DEFAULT_DQK = {"architecture": ARCHITECTURE, "uffr_mask": "gaussian", "lambda_tv": 1e-4,
               "lambda_sparsity": 1e-5, "frequency_order": "DAFD_then_UFFR"}


class DQKBackbone(nn.Module):
    def __init__(self, original, channels, hidden_dim, levels, dqk):
        super().__init__()
        self.original = original
        self.dafd_modules = nn.ModuleList([DAFD(c) for c in channels])
        self.uffr_modules = nn.ModuleList([UFFR(c, mask_kind=dqk["uffr_mask"]) for c in channels])
        self.mfa_modules = nn.ModuleList([MFA(hidden_dim) for _ in range(levels)])
        self.gate_statistics = None

    def forward(self, samples):
        raw = self.original.encoder(samples.tensors)
        if len(raw) != len(self.dafd_modules):
            raise ValueError("Backbone tap count changed")
        refined, statistics = [], []
        for tensor, dafd, uffr in zip(raw, self.dafd_modules, self.uffr_modules):
            tensor, gates = dafd(tensor)
            refined.append(uffr(tensor))
            statistics.extend(gates)
        self.gate_statistics = torch.stack(statistics)
        projected = self.original.projector(refined)
        if len(projected) != len(self.mfa_modules):
            raise ValueError("Projector level count changed")
        features = [NestedTensor(mfa(tensor), self.original._level_mask(samples, tensor), samples.no_padding)
                    for mfa, tensor in zip(self.mfa_modules, projected)]
        if self.original.cross_attn_projector is not None:
            raise ValueError("DQK v1 supports the single detection projector only")
        return features, None

    def get_named_param_lr_pairs(self, args, prefix="backbone.0"):
        # Preserve native DINO layer-wise LR/decay and match renamed state keys.
        pairs = self.original.get_named_param_lr_pairs(args, prefix=prefix)
        return {name.replace(prefix+".", prefix+".original.", 1): value for name, value in pairs.items()}


class RFDETRDQKEnhanced(LWDETR):
    @classmethod
    def from_base(cls, base, model_config, dqk=None):
        settings = dict(DEFAULT_DQK if dqk is None else dqk)
        if settings.get("architecture") != ARCHITECTURE:
            raise ValueError("Unsupported DQK architecture")
        if settings.get("frequency_order") != "DAFD_then_UFFR":
            raise ValueError("Unknown DQK frequency order")
        # Adopt a fully constructed native LWDETR, retaining native public methods
        # and head/decoder parameter names. No hooks, class monkey-patches or globals.
        core = cls.__new__(cls)
        core.__dict__ = base.__dict__.copy()
        core._modules = base._modules.copy()
        core._parameters = base._parameters.copy()
        core._buffers = base._buffers.copy()
        core.dqk_config = settings
        backbone = core.backbone[0]
        channels = list(backbone.encoder._out_feature_channels)
        core.backbone[0] = DQKBackbone(backbone, channels, model_config.hidden_dim,
                                       len(model_config.projector_scale), settings)
        core.darm = DARM()
        core.register_buffer("dqk_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        core.register_buffer("dqk_std", torch.tensor([.229, .224, .225])[None, :, None, None])
        return core

    def forward(self, samples, targets=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        image = samples.tensors
        rgb = image*self.dqk_std+self.dqk_mean
        restored, residual = self.darm(rgb)
        if samples.mask is not None:
            valid = (~samples.mask).unsqueeze(1)
            residual = residual*valid
            restored = torch.where(valid, restored, rgb)
        normalized = (restored-self.dqk_mean)/self.dqk_std
        output = super().forward(NestedTensor(normalized, samples.mask, samples.no_padding), targets)
        output["loss_tv"] = compute_tv_loss(residual) if self.training else image.new_zeros(())
        statistics = self.backbone[0].gate_statistics
        output["loss_sparsity"] = ((statistics-.8)**2).mean() if self.training else image.new_zeros(())
        # Do not retain a graph on the module (EMA deepcopy must remain valid).
        self.backbone[0].gate_statistics = None
        return output


def build_enhanced_facade(checkpoint_path, device="cpu", resolution=640):
    from rfdetr import RFDETRMedium
    from rfdetr.config import RFDETRMediumConfig
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    notes = checkpoint.get("args", {}).get("notes") or {}
    if checkpoint.get("model_name") != "RFDETRDQKEnhanced" or "dqk" not in notes:
        raise ValueError("Checkpoint is not a verified DQK enhanced model")
    names = checkpoint["args"]["class_names"]
    config = dict(checkpoint["model_config"])
    config.update(pretrain_weights=None, device=device, resolution=resolution)
    mc = RFDETRMediumConfig(**config)
    facade = RFDETRMedium(**mc.model_dump())
    facade.model.model = RFDETRDQKEnhanced.from_base(facade.model.model, mc, notes["dqk"])
    facade.model.model.load_state_dict(checkpoint["model"], strict=True)
    facade.model.class_names = list(names)
    facade.model.model.eval()
    return facade, checkpoint
