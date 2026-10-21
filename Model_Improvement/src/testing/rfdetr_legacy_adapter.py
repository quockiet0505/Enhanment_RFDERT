"""Strict, checkpoint-backed custom-class RF-DETR inference using its native API."""
import importlib.metadata

import cv2
import numpy as np


class RFDETRAdapter:
    def __init__(self, model_cfg, weights_path, class_names, name_to_cat_id,
                 device="cuda:0", conf_thresh=0.0, imgsz=640, max_detections=100):
        import torch
        from rfdetr import RFDETRMedium

        if model_cfg.get("model_class") != "RFDETRMedium":
            raise ValueError("This verified adapter supports RFDETRMedium baseline/legacy only; enhanced requires its own verified architecture.")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_name") != "RFDETRMedium":
            raise ValueError("Checkpoint architecture is not RFDETRMedium")
        version = importlib.metadata.version("rfdetr")
        if version != checkpoint.get("rfdetr_version"):
            raise ValueError(f"RF-DETR version mismatch: runtime={version}, checkpoint={checkpoint.get('rfdetr_version')}")
        state = checkpoint["model"]
        if any(any(part in key.lower() for part in ("darm", "dafd", "uffr", "mfa")) for key in state):
            raise ValueError("A checkpoint containing DQK modules cannot be evaluated as baseline")
        args = checkpoint.get("args", {})
        names = args.get("class_names") if isinstance(args, dict) else getattr(args, "class_names", None)
        if not names or len(names) != len(set(names)):
            raise ValueError("Checkpoint must contain unique training class_names")
        self.checkpoint_classes = list(names)
        if set(names) != set(class_names) or set(names) != set(name_to_cat_id):
            raise ValueError("Checkpoint and evaluated dataset classes differ")
        configured_names = model_cfg.get("checkpoint_classes")
        if configured_names is not None and list(configured_names) != list(names):
            raise ValueError("Configured checkpoint class order differs from saved training order")
        if int(model_cfg.get("label_offset", 0)) != 0:
            raise ValueError("Custom-class RF-DETR uses contiguous zero-based labels")
        expected_slots = len(names) + 1
        head_shapes = {key: list(value.shape) for key, value in state.items() if "class_embed" in key}
        if not head_shapes or any(shape[0] != expected_slots for shape in head_shapes.values()):
            raise ValueError(f"All classification heads must have {expected_slots} output slots")
        saved_config = checkpoint.get("model_config", {})
        if int(saved_config.get("num_classes", -1)) != len(names):
            raise ValueError("Checkpoint config/head/class names disagree")
        self.model = RFDETRMedium.from_checkpoint(weights_path, device=device, resolution=int(imgsz))
        core = self.model.model.model
        # The native constructor loads compatibly; this second, strict load is deliberate:
        # missing/unexpected keys or shape changes must fail before any metric is emitted.
        core.load_state_dict(state, strict=True)
        loaded = core.state_dict()
        if any(not torch.equal(loaded[key].cpu(), value.cpu()) for key, value in state.items()):
            raise RuntimeError("Loaded tensors differ from the requested checkpoint")
        if list(self.model.class_names) != list(names):
            raise ValueError("Native API resolved a different training class order")
        core.eval()
        self.class_idx_to_cat_id = {index: int(name_to_cat_id[name]) for index, name in enumerate(names)}
        self.ignored_class_indices = set()
        self.label_offset = 0
        self.background_class_id = len(names)
        self.conf_thresh = float(conf_thresh)
        self.max_detections = int(max_detections)
        self.backend_max_detections = int(self.model.model.args.num_select)
        self.imgsz = int(self.model.model_config.resolution)
        if self.imgsz != int(imgsz) or core.class_embed.out_features != expected_slots:
            raise ValueError("Effective model resolution/head differs from request")
        self.load_validation = {
            "strict": True, "missing_keys": [], "unexpected_keys": [],
            "all_checkpoint_tensors_equal": True, "state_entries": len(state),
            "classification_head_shapes": head_shapes, "checkpoint_epoch": checkpoint.get("epoch"),
            "rfdetr_version": version, "model_name": checkpoint["model_name"],
            "preprocessing": "native RFDETR.predict: RGB, resize, saved model mean/std",
            "postprocessing": "native sigmoid top-k across query-class pairs; filter background; global cap",
            "means": list(self.model.means), "stds": list(self.model.stds),
        }

    def native_detections(self, image_bgr):
        from PIL import Image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.model.predict(Image.fromarray(image_rgb), threshold=self.conf_thresh,
                                  include_source_image=False)

    def predict_image(self, image_bgr, orig_w, orig_h):
        if image_bgr is None or image_bgr.shape[:2] != (orig_h, orig_w):
            raise ValueError("Image dimensions do not match the decoded image")
        output = self.native_detections(image_bgr)
        boxes = np.asarray(output.xyxy)
        scores = np.asarray(output.confidence)
        labels = np.asarray(output.class_id).astype(int)
        if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
            raise ValueError("Native prediction contains non-finite values")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Native confidence is outside [0,1]")
        unknown = set(labels.tolist()) - set(self.class_idx_to_cat_id) - {self.background_class_id}
        if unknown:
            raise ValueError(f"Unmapped RF-DETR output labels: {sorted(unknown)}")
        candidates = np.flatnonzero((labels != self.background_class_id) & (scores >= self.conf_thresh))
        order = candidates[np.argsort(-scores[candidates], kind="stable")][:self.max_detections]
        detections = []
        for index in order:
            x1, y1, x2, y2 = boxes[index].astype(float)
            x1, x2 = np.clip([x1, x2], 0, orig_w)
            y1, y2 = np.clip([y1, y2], 0, orig_h)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({"bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                               "score": float(scores[index]),
                               "category_id": self.class_idx_to_cat_id[int(labels[index])]})
        return detections

    def get_torch_module(self):
        return self.model.model.model

    def verify_pipeline(self, image_bgr):
        """Check native RGB/resize/normalization and postprocess on a validation image."""
        import torch
        from PIL import Image
        from torchvision.transforms import functional as F
        captured = {}
        core = self.get_torch_module()
        def before(module, args):
            value = args[0]
            captured["input"] = (value.tensors if hasattr(value, "tensors") else value).detach().clone()
        def after(module, args, output):
            captured["output"] = output
        handles = [core.register_forward_pre_hook(before), core.register_forward_hook(after)]
        try:
            native = self.native_detections(image_bgr)
        finally:
            for handle in handles:
                handle.remove()
        actual = captured["input"]
        rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        expected = F.to_tensor(rgb).to(actual.device)
        expected = F.normalize(F.resize(expected, [self.imgsz, self.imgsz], antialias=False),
                               self.model.means, self.model.stds).unsqueeze(0)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        h, w = image_bgr.shape[:2]
        with torch.no_grad():
            post = self.model.model.postprocess(captured["output"],
                       target_sizes=torch.tensor([[h, w]], device=actual.device),
                       score_threshold=self.conf_thresh)[0]
        keep = post["scores"] > self.conf_thresh
        np.testing.assert_array_equal(np.asarray(native.class_id), post["labels"][keep].cpu().numpy())
        np.testing.assert_allclose(native.confidence, post["scores"][keep].float().cpu().numpy(), rtol=1e-6)
        np.testing.assert_allclose(native.xyxy, post["boxes"][keep].float().cpu().numpy(), rtol=1e-6, atol=1e-4)
        return {"rgb_resize_normalize_parity": True, "native_postprocess_parity": True,
                "input_shape": list(actual.shape), "native_detections": len(native),
                "logits_shape": list(captured["output"]["pred_logits"].shape)}
