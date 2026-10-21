"""DQK enhanced checkpoint adapter; shares the verified native inference path."""
import importlib.metadata
import torch

from src.testing.rfdetr_legacy_adapter import RFDETRAdapter
from src.rfdetr.lwdetr_dqk import build_enhanced_facade


class RFDETREnhancedAdapter(RFDETRAdapter):
    def __init__(self, model_cfg, weights_path, class_names, name_to_cat_id,
                 device="cuda:0", conf_thresh=0.0, imgsz=640, max_detections=100):
        self.model, checkpoint = build_enhanced_facade(weights_path, device, int(imgsz))
        if importlib.metadata.version("rfdetr") != checkpoint.get("rfdetr_version"):
            raise ValueError("Enhanced checkpoint/runtime RF-DETR versions differ")
        names = checkpoint["args"]["class_names"]
        if len(names) != 32 or len(set(names)) != 32 or set(names) != set(class_names):
            raise ValueError("Enhanced checkpoint class schema differs from the dataset")
        if model_cfg.get("checkpoint_classes", names) != names or model_cfg.get("label_offset", 0) != 0:
            raise ValueError("Configured enhanced label schema is incorrect")
        state = self.model.model.model.state_dict()
        if any(not torch.equal(state[k].cpu(), tensor.cpu()) for k, tensor in checkpoint["model"].items()):
            raise RuntimeError("Enhanced checkpoint tensors were not restored exactly")
        for token in ("darm.", "dafd_modules.", "uffr_modules.", "mfa_modules."):
            if not any(token in key for key in state):
                raise ValueError(f"Enhanced checkpoint lacks {token}")
        if any(v.shape[0] != 33 for k, v in state.items() if "class_embed" in k):
            raise ValueError("Enhanced classification heads must have 33 slots")
        self.checkpoint_classes = list(names)
        self.class_idx_to_cat_id = {i: int(name_to_cat_id[name]) for i, name in enumerate(names)}
        self.ignored_class_indices = set()
        self.background_class_id = 32
        self.label_offset = 0
        self.conf_thresh = float(conf_thresh)
        self.max_detections = int(max_detections)
        self.backend_max_detections = int(self.model.model.args.num_select)
        self.imgsz = int(self.model.model_config.resolution)
        self.load_validation = {"strict": True, "missing_keys": [], "unexpected_keys": [],
            "all_checkpoint_tensors_equal": True, "state_entries": len(state),
            "architecture": checkpoint["args"]["notes"]["dqk"],
            "checkpoint_epoch": checkpoint["epoch"], "model_name": checkpoint["model_name"],
            "rfdetr_version": checkpoint["rfdetr_version"]}
