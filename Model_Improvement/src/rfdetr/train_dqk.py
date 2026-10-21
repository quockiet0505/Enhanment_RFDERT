"""RF-DETR 1.10.1 Lightning training with the actual four-module DQK core."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from torch import nn
import yaml
from pytorch_lightning import Callback, seed_everything
from rfdetr.config import RFDETRMediumConfig, TrainConfig
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.trainer import build_trainer
from rfdetr.datasets.coco import filter_parent_categories, annotated_category_ids
from src.rfdetr.lwdetr_dqk import RFDETRDQKEnhanced, DEFAULT_DQK
from src.common.dqk_checkpoints import export_best


class DQKCriterion(nn.Module):
    def __init__(self, base, settings, detection_weights):
        super().__init__()
        self.base = base
        self.weight_dict = dict(base.weight_dict, loss_tv=settings["lambda_tv"],
                                loss_sparsity=settings["lambda_sparsity"])
        for key in self.weight_dict:
            if key.startswith("loss_bbox"):
                self.weight_dict[key] = float(detection_weights["loss_bbox"])
            elif key.startswith("loss_giou"):
                self.weight_dict[key] = float(detection_weights["loss_giou"])

    def forward(self, outputs, targets, **kwargs):
        losses = self.base(outputs, targets, **kwargs)
        return dict(losses, loss_tv=outputs["loss_tv"], loss_sparsity=outputs["loss_sparsity"])


class DQKModelModule(RFDETRModelModule):
    def __init__(self, model_config, train_config, settings, detection_weights):
        super().__init__(model_config, train_config)
        self.model = RFDETRDQKEnhanced.from_base(self.model, model_config, settings)
        self.criterion = DQKCriterion(self.criterion, settings, detection_weights)
        self.strict_loading = True


class CommitEpoch(Callback):
    def __init__(self, commit_fn):
        self.commit_fn = commit_fn

    def on_train_epoch_end(self, trainer, pl_module):
        if self.commit_fn and trainer.is_global_zero:
            self.commit_fn()

    def on_fit_end(self, trainer, pl_module):
        if self.commit_fn and trainer.is_global_zero:
            self.commit_fn()


class AuditTraining(Callback):
    def __init__(self, output):
        self.output = Path(output)
        self.report = {"optimizer_steps": 0}

    def on_fit_start(self, trainer, pl_module):
        parameters = [p for opt in trainer.optimizers for group in opt.param_groups for p in group["params"]]
        actual = {id(p) for p in parameters}
        expected = {id(p) for p in pl_module.model.parameters() if p.requires_grad}
        if actual != expected or len(parameters) != len(actual):
            raise RuntimeError("Optimizer parameter coverage missing/duplicated")
        self.report.update(optimizer_covers_all_trainable=True, no_duplicate_parameters=True)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        relevant = [(n, p) for n, p in pl_module.model.named_parameters()
                    if any(token in n for token in ("darm.", "dafd_modules.", "uffr_modules.", "mfa_modules."))]
        missing = [n for n, p in relevant if p.grad is None]
        if missing or any(not torch.isfinite(p.grad).all() for _, p in relevant):
            raise RuntimeError(f"DQK missing/nonfinite gradients: {missing}")
        self.report.update(optimizer_steps=self.report["optimizer_steps"]+1,
                           dqk_parameters_with_grad=len(relevant), dqk_gradients_finite=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            (self.output/"training_audit.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")


def dataset_schema(dataset_dir):
    names, hashes = None, {}
    for split in ("train", "valid", "test"):
        path = Path(dataset_dir)/split/"_annotations.coco.json"
        raw = path.read_bytes()
        annotations = json.loads(raw)
        kept = filter_parent_categories(annotations["categories"], annotated_category_ids(annotations))
        current = [category["name"] for category in kept]
        if names is None:
            names = current
        if current != names or len(current) != 32 or len(set(current)) != 32:
            raise ValueError("Train/valid/test must share the same 32-class schema")
        hashes[split] = hashlib.sha256(raw).hexdigest()
    return names, hashes


def train_rfdetr_dqk(dataset_dir, config_path, output_dir, commit_fn=None, *, smoke=False,
                     resume=None, epochs=None):
    if importlib.metadata.version("rfdetr") != "1.10.1":
        raise RuntimeError("DQK v1 is pinned to rfdetr==1.10.1")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Reusing a populated output is only allowed with an explicit full-state resume.
    if any(output.iterdir()) and not resume:
        raise FileExistsError(f"Output already populated: {output}")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    training = config["training"]
    names, hashes = dataset_schema(dataset_dir)
    settings = dict(DEFAULT_DQK, **config.get("dqk", {}))
    if config["model"]["variant"] != "RFDETRDQKEnhanced" or int(training["device"]) != 0:
        raise ValueError("DQK pipeline expects RFDETRDQKEnhanced on a single GPU (device 0)")
    seed_everything(int(training["seed"]), workers=True)
    mc = RFDETRMediumConfig(num_classes=len(names), resolution=int(training["imgsz"]),
        model_name="RFDETRDQKEnhanced", compile=False,
        pretrain_weights=None if resume or not config["model"].get("pretrained", True) else "rf-detr-medium.pth")
    if mc.pretrain_weights:
        # Direct Lightning module construction bypasses facade cache preparation.
        Path(mc.pretrain_weights).parent.mkdir(parents=True, exist_ok=True)
    options = dict(dataset_dir=str(dataset_dir), output_dir=str(output),
        epochs=1 if smoke else int(epochs or training["epochs"]),
        batch_size=1 if smoke else int(training["batch_size"]),
        grad_accum_steps=1 if smoke else int(training["grad_accum_steps"]),
        num_workers=0 if smoke else int(training["workers"]), lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]), seed=int(training["seed"]),
        amp_dtype=training["amp_dtype"], early_stopping=not smoke,
        early_stopping_patience=int(training["patience"]), early_stopping_use_ema=True,
        eval_interval=1 if smoke else int(training["eval_interval"]), eval_max_dets=100, use_ema=True, eval_base_model=False,
        tensorboard=True, run_test=False, class_names=names, compact_train_metrics=False,
        multi_scale=bool(training.get("multi_scale", False)),
        expanded_scales=bool(training.get("multi_scale", False)), compute_val_loss=True,
        cls_loss_coef=float(config["loss_weights"]["loss_cls"]),
        notes={"dqk": settings, "annotation_sha256": hashes,
               "detection_loss_weights": config["loss_weights"],
               "purpose": "smoke_not_scientific_result" if smoke else "DQK_enhanced_training"})
    tc = TrainConfig(**options)
    module = DQKModelModule(mc, tc, settings, config["loss_weights"])
    datamodule = RFDETRDataModule(mc, tc)
    trainer_options = {"accelerator": "gpu", "devices": 1}
    if smoke:
        trainer_options.update(limit_train_batches=3, limit_val_batches=1, num_sanity_val_steps=0)
    trainer = build_trainer(tc, mc, **trainer_options)
    trainer.callbacks.append(AuditTraining(output))
    trainer.callbacks.append(CommitEpoch(commit_fn))
    effective = {"model_config": mc.model_dump(), "train_config": tc.model_dump(), "dqk": settings,
                 "class_names": names, "annotation_sha256": hashes}
    (output/"dqk_effective_config.json").write_text(json.dumps(effective, indent=2, default=str), encoding="utf-8")
    try:
        trainer.fit(module, datamodule, ckpt_path=resume)
        winner = output/"checkpoint_best_total.pth"
        if not winner.is_file():
            raise RuntimeError("Trainer did not produce checkpoint_best_total.pth")
        checkpoint = torch.load(winner, map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        for token in ("darm.", "dafd_modules.", "uffr_modules.", "mfa_modules."):
            if not any(token in key for key in state):
                raise RuntimeError(f"Best checkpoint lacks {token}")
        if checkpoint.get("model_name") != "RFDETRDQKEnhanced":
            raise RuntimeError("Best checkpoint has incorrect architecture metadata")
        # RF-DETR 1.10.1 strips model_config from best_total; the native facade
        # normally restores it after fit. We own the custom facade, so export the
        # complete inference schema ourselves, without changing winner weights.
        return export_best(output, mc.model_dump(), names, options["notes"])
    finally:
        if commit_fn:
            commit_fn()
