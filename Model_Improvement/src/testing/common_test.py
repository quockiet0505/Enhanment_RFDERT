"""Config-driven COCOeval benchmark for eight object-detection models."""
from pathlib import Path
import os
import time
import json
import argparse
import gzip
import gc
import hashlib
import importlib.metadata
import platform
import uuid
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import modal

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(CURRENT_DIR, "..", "..", "configs", "models_config.json")
PROTOCOL_NAME = "common-coco-v7-640-b1"
REQUIRED_MODEL_KEYS = {
    "YOLOv26X",
    "FasterRCNN_ResNet50",
    "FasterRCNN_MobileNetv3Large",
    "FCOS_ResNet50_FPN",
    "YOLOv11",
    "RetinaNet",
    "RFDETR_Medium",
    "RTDETR_L",
    "RFDETR_DQK_Enhanced",
}
REQUIRED_MODEL_ROUTES = {
    "YOLOv26X": ("ultralytics", {"yolo"}),
    "FasterRCNN_ResNet50": ("torchvision", {"fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2"}),
    "FasterRCNN_MobileNetv3Large": ("torchvision", {"fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_mobilenet_v3_large_320_fpn"}),
    "FCOS_ResNet50_FPN": ("torchvision", {"fcos_resnet50_fpn"}),
    "YOLOv11": ("ultralytics", {"yolo"}),
    "RetinaNet": ("torchvision", {"retinanet_resnet50_fpn", "retinanet_resnet50_fpn_v2"}),
    "RFDETR_Medium": ("rfdetr", {"rfdetr_medium"}),
    "RTDETR_L": ("ultralytics", {"rtdetr"}),
    "RFDETR_DQK_Enhanced": ("rfdetr", {"rfdetr_medium"}),
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or CONFIG_FILE
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Không thể đọc file cấu hình tại {path}: {e}")
    return {}


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source_file(path: str) -> str:
    with open(path, "rb") as source:
        normalized = source.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def installed_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def describe_runtime_device(device: str) -> str:
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = torch.device(device).index or 0
            return torch.cuda.get_device_name(index)
    except Exception:
        pass
    return platform.processor() or "cpu"


def default_compute_device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def compute_protocol_id(protocol: Dict[str, Any]) -> str:
    payload = {key: value for key, value in protocol.items() if key != "protocol_id"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_payload_sha256(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    def escape_cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    headers = [escape_cell(column) for column in dataframe.columns]
    rows = [[escape_cell(value) for value in row] for row in dataframe.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_requested_inference_config(model_cfg: Dict[str, Any], bench_cfg: Dict[str, Any], evaluated_classes: List[str]) -> Dict[str, Any]:
    return {
        "family": model_cfg.get("family"),
        "model_type": model_cfg.get("model_type"),
        "model_class": model_cfg.get("model_class"),
        "imgsz": int(model_cfg.get("imgsz", bench_cfg["imgsz"])),
        "batch_size": int(bench_cfg["batch_size"]),
        "prediction_conf_threshold": float(bench_cfg["conf_threshold"]),
        "max_detections": int(bench_cfg["max_detections"]),
        "label_offset": int(model_cfg["label_offset"]) if "label_offset" in model_cfg else None,
        "nms_iou_threshold": model_cfg.get("nms_iou_threshold"),
        "constructor_kwargs": dict(model_cfg.get("constructor_kwargs", {})),
        "checkpoint_classes": list(model_cfg.get("checkpoint_classes") or evaluated_classes),
        "class_name_map": dict(model_cfg.get("class_name_map", {})),
        "ignored_checkpoint_classes": list(model_cfg.get("ignored_checkpoint_classes", [])),
    }


def build_resolved_inference_config(adapter: Any, requested_config: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint_classes = getattr(adapter, "checkpoint_classes", {})
    if isinstance(checkpoint_classes, dict):
        indexed_classes = checkpoint_classes
    else:
        indexed_classes = {index: name for index, name in enumerate(checkpoint_classes or [])}
    resolved = {
        "requested": requested_config,
        "effective": {
            "imgsz": int(getattr(adapter, "imgsz")),
            "prediction_conf_threshold": float(getattr(adapter, "conf_thresh")),
            "max_detections": int(getattr(adapter, "max_detections")),
            "backend_max_detections": int(getattr(adapter, "backend_max_detections", getattr(adapter, "max_detections"))),
            "nms_iou_threshold": getattr(adapter, "nms_iou_threshold", None),
            "constructor_kwargs": dict(getattr(adapter, "effective_constructor_kwargs", {})),
            "background_class_id": getattr(adapter, "background_class_id", None),
            "ignored_output_labels": sorted(int(label) for label in getattr(adapter, "ignored_output_labels", set())),
        },
        "checkpoint_classes": [{"index": int(index), "name": str(name)} for index, name in sorted(indexed_classes.items())],
        "class_idx_to_category_id": {str(int(index)): int(category_id) for index, category_id in sorted(getattr(adapter, "class_idx_to_cat_id", {}).items())},
        "ignored_class_indices": sorted(int(index) for index in getattr(adapter, "ignored_class_indices", set())),
    }
    return resolved


def build_protocol_metadata(bench_cfg: Dict[str, Any], annotation_sha256: str, image_ids: List[int], valid_categories: List[Dict[str, Any]], runtime_device: str) -> Dict[str, Any]:
    protocol = {
        "name": bench_cfg.get("protocol_name", PROTOCOL_NAME),
        "evaluator": "pycocotools.cocoeval.COCOeval",
        "evaluator_source_sha256": sha256_source_file(os.path.abspath(__file__)),
        "iou_type": "bbox",
        "annotation_sha256": annotation_sha256,
        "image_ids_sha256": hashlib.sha256(json.dumps(sorted(int(i) for i in image_ids), separators=(",", ":")).encode("utf-8")).hexdigest(),
        "num_test_images": len(image_ids),
        "categories": [{"id": int(c["id"]), "name": c["name"]} for c in valid_categories],
        "settings": {
            "inference_imgsz": int(bench_cfg.get("imgsz", 640)),
            "inference_batch_size": int(bench_cfg.get("batch_size", 1)),
            "prediction_conf_threshold": float(bench_cfg.get("conf_threshold", 0.0)),
            "operating_conf_threshold": float(bench_cfg.get("operating_conf_threshold", 0.55)),
            "operating_iou_threshold": float(bench_cfg.get("iou_threshold", 0.50)),
            "operating_metric_source": "pycocotools.COCOeval.evalImgs",
            "operating_averaging": "micro_over_all_categories",
            "max_detections": int(bench_cfg.get("max_detections", 100)),
            "latency_batch_size": int(bench_cfg.get("batch_size", 1)),
            "latency_warmup_runs": int(bench_cfg.get("warmup_runs", 10)),
            "runtime_device": runtime_device,
            "coco_iou_thresholds": [round(float(v), 2) for v in np.arange(0.50, 0.96, 0.05)],
            "coco_area_ranges": ["all", "small", "medium", "large"],
        },
        "software": {
            "python": platform.python_version(),
            "pycocotools": installed_version("pycocotools"),
            "torch": installed_version("torch"),
            "torchvision": installed_version("torchvision"),
            "ultralytics": installed_version("ultralytics"),
            "rfdetr": installed_version("rfdetr"),
        },
    }
    protocol["protocol_id"] = compute_protocol_id(protocol)
    return protocol


def validate_configuration(config_data: Dict[str, Any]) -> None:
    classes = config_data.get("classes", [])
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("config.classes must be a non-empty list of unique class names")

    bench_cfg = config_data.get("benchmark_settings", {})
    required_imgsz = int(bench_cfg.get("imgsz", 0))
    required_batch_size = int(bench_cfg.get("batch_size", 0))
    if bench_cfg.get("protocol_name") != PROTOCOL_NAME:
        raise ValueError(f"benchmark_settings.protocol_name must be {PROTOCOL_NAME}")

    supported = {
        "ultralytics": {"yolo", "rtdetr"},
        "torchvision": {"fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2", "fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_mobilenet_v3_large_320_fpn", "fcos_resnet50_fpn", "retinanet_resnet50_fpn", "retinanet_resnet50_fpn_v2"},
        "rfdetr": {"rfdetr_medium"},
    }
    models = config_data.get("models", {})
    if not models:
        raise ValueError("config.models is empty")
    for model_key, model_cfg in models.items():
        family = model_cfg.get("family")
        model_type = model_cfg.get("model_type")
        if family not in supported or model_type not in supported[family]:
            raise ValueError(f"Unsupported configuration for {model_key}: family={family}, model_type={model_type}")
        required_family, allowed_model_types = REQUIRED_MODEL_ROUTES[model_key]
        if family != required_family or model_type not in allowed_model_types:
            raise ValueError(f"{model_key} must use family={required_family} and model_type in {sorted(allowed_model_types)}, got family={family}, model_type={model_type}")


def resolve_checkpoint_path(model_key: str, model_cfg: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    env_key = "MODEL_WEIGHTS_" + "".join(char if char.isalnum() else "_" for char in model_key.upper())
    candidates = []
    env_path = os.environ.get(env_key, "").strip()
    if env_path: candidates.append(os.path.abspath(os.path.expanduser(env_path)))
    cloud_path = str(model_cfg.get("weights", "")).strip()
    if cloud_path: candidates.append(os.path.abspath(os.path.expanduser(cloud_path)))
    local_spec = str(model_cfg.get("local_weights", "")).strip()
    if local_spec:
        local_path = local_spec if os.path.isabs(local_spec) else os.path.join(CURRENT_DIR, local_spec)
        candidates.append(os.path.abspath(os.path.expanduser(local_path)))

    unique_candidates = list(dict.fromkeys(candidates))
    for candidate in unique_candidates:
        if os.path.isfile(candidate):
            return candidate, unique_candidates
    return None, unique_candidates


def get_roboflow_api_key(config_data: Optional[Dict[str, Any]] = None) -> str:
    env_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if env_key: return env_key

    candidates = [os.path.join(CURRENT_DIR, ".env"), os.path.join(os.path.dirname(CURRENT_DIR), ".env"), ".env"]
    for env_file in candidates:
        if os.path.exists(env_file):
            try:
                with open(env_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == "ROBOFLOW_API_KEY":
                                val = v.strip().strip('"').strip("'")
                                if val and val != "your_roboflow_api_key_here":
                                    return val
            except Exception:
                pass

    if config_data:
        cfg_key = config_data.get("roboflow", {}).get("api_key", "").strip()
        if cfg_key: return cfg_key
    return ""


GLOBAL_CONFIG = load_config()
MODAL_SETTINGS = GLOBAL_CONFIG.get("modal_settings", {})

MODAL_VOLUME_NAME = os.environ.get(
    "MODAL_VOLUME_NAME",
    MODAL_SETTINGS.get(
        "volume_name",
        "volume-10-09-2026-fasterrcnn-640"
    )
)

APP_NAME = MODAL_SETTINGS.get(
    "app_name", "app-common-coco-evaluate"
)
MODAL_GPU = MODAL_SETTINGS.get("gpu", "A100")
MODAL_TIMEOUT = int(
    MODAL_SETTINGS.get("timeout_seconds", 7200)
)

app = modal.App(APP_NAME)

dataset_volume = modal.Volume.from_name(
    MODAL_VOLUME_NAME,
    create_if_missing=False
)

runs_volume = modal.Volume.from_name(
    "rfdetr-runs",
    create_if_missing=False
)

PROJECT_ROOT = Path.cwd()

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==5.1.0",
        "ultralytics==8.4.161",
        "rfdetr==1.11.0",
        "roboflow>=1.1.33",
        "opencv-python-headless>=4.10.0.84",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0",
        "pillow>=10.3.0",
        "thop>=0.1.1",
        "tabulate>=0.9.0",
        "pycocotools==2.0.8",
        "tqdm>=4.66.0",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(
        str(PROJECT_ROOT / "src"),
        remote_path="/root/src",
    )
)

class RoboflowCOCODatasetManager:
    def __init__(self, api_key: str = "", dataset_root: Optional[str] = None, workspace: str = "nckhcict2025", project: str = "completed-project", version: int = 5, dummy_cat_name: str = "19-8-BoSungAnhChoCacLopBiTh-v3zo"):
        self.api_key = api_key if api_key else get_roboflow_api_key()
        self.dataset_root = dataset_root
        self.workspace = workspace
        self.project = project
        self.version = version
        self.dummy_cat_name = dummy_cat_name
        self.coco_gt = None
        self.valid_categories = []
        self.valid_cat_ids = []
        self.class_names = []
        self.name_to_cat_id = {}
        self.test_images = {}
        self.test_image_paths = []
        self.test_annotation_sha256 = ""

    def ensure_dataset(self, target_dir: Optional[str] = None) -> str:
        candidate_paths = [target_dir if target_dir else None, self.dataset_root if self.dataset_root else None, "/data/dataset/Completed-Project-5", "./dataset/Completed-Project-5", os.path.abspath(os.path.join(CURRENT_DIR, "..", "Completed-Project-5"))]
        candidate_paths = [p for p in candidate_paths if p]

        for p in candidate_paths:
            if os.path.exists(p):
                ann = os.path.join(p, "test", "_annotations.coco.json")
                if os.path.exists(ann):
                    print(f"[DATASET] Tìm thấy tập dữ liệu COCO có sẵn tại: {os.path.abspath(p)}")
                    self.dataset_root = p
                    return p

        download_dir = target_dir or self.dataset_root or os.path.join(CURRENT_DIR, "dataset")
        os.makedirs(download_dir, exist_ok=True)
        if not self.api_key:
            raise ValueError("[ERROR] Thiếu Roboflow API Key!")

        from roboflow import Roboflow
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(self.workspace).project(self.project)
        version = project.version(self.version)

        curr_dir = os.getcwd()
        try:
            os.chdir(download_dir)
            dataset = version.download("coco")
            actual_location = getattr(dataset, "location", download_dir)
        finally:
            os.chdir(curr_dir)

        actual_path = actual_location
        if not os.path.exists(os.path.join(actual_path, "test", "_annotations.coco.json")):
            for root, _, files in os.walk(download_dir):
                if "_annotations.coco.json" in files and os.path.basename(root) == "test":
                    actual_path = os.path.dirname(root)
                    break

        print(f"[DATASET] Tải thành công tập dữ liệu COCO về: {actual_path}")
        self.dataset_root = actual_path
        return actual_path

    def load_test_split(self):
        from pycocotools.coco import COCO
        test_ann = os.path.join(self.dataset_root, "test", "_annotations.coco.json")
        if not os.path.exists(test_ann):
            for root, _, files in os.walk(self.dataset_root):
                if "_annotations.coco.json" in files and os.path.basename(root) == "test":
                    test_ann = os.path.join(root, "_annotations.coco.json")
                    break

        if not os.path.exists(test_ann):
            raise FileNotFoundError(f"[ERROR] Không tìm thấy test/_annotations.coco.json trong {self.dataset_root}")

        print(f"[DATASET] Nạp COCO Ground Truth từ: {test_ann}")
        self.test_annotation_sha256 = sha256_file(test_ann)
        self.coco_gt = COCO(test_ann)

        raw_cats = sorted(self.coco_gt.dataset.get("categories", []), key=lambda x: x["id"])
        self.valid_categories = [c for c in raw_cats if c["name"].strip() != self.dummy_cat_name]
        self.valid_cat_ids = [c["id"] for c in self.valid_categories]
        self.class_names = [c["name"] for c in self.valid_categories]
        self.name_to_cat_id = {c["name"]: c["id"] for c in self.valid_categories}

        test_dir = os.path.dirname(test_ann)
        self.test_images = {}
        self.test_image_paths = []

        for img_id in self.coco_gt.getImgIds():
            info = self.coco_gt.loadImgs(img_id)[0]
            fn = info["file_name"]
            candidates = [os.path.join(test_dir, fn), os.path.join(test_dir, "images", fn), os.path.join(self.dataset_root, "test", fn), os.path.join(self.dataset_root, "test", "images", fn)]
            actual_f = next((p for p in candidates if os.path.exists(p)), None)
            if actual_f:
                self.test_images[img_id] = {"path": actual_f, "width": info["width"], "height": info["height"]}
                self.test_image_paths.append(actual_f)

        missing_image_ids = sorted(set(self.coco_gt.getImgIds()) - set(self.test_images))
        if missing_image_ids:
            raise FileNotFoundError(f"Missing {len(missing_image_ids)} test images declared in the COCO JSON")

        return self.coco_gt, self.valid_categories, self.test_images


def build_class_mapping(indexed_names: Dict[int, str], name_to_cat_id: Dict[str, int], model_cfg: Dict[str, Any], backend_name: str) -> Tuple[Dict[int, int], set[int]]:
    aliases = model_cfg.get("class_name_map", {})
    ignored_names = set(model_cfg.get("ignored_checkpoint_classes", []))
    index_to_cat_id = {}
    ignored_indices = set()
    unknown = []

    for index, raw_name in indexed_names.items():
        if raw_name in ignored_names:
            ignored_indices.add(int(index))
            continue
        canonical_name = aliases.get(raw_name, raw_name)
        if canonical_name not in name_to_cat_id:
            unknown.append(raw_name)
            continue
        index_to_cat_id[int(index)] = int(name_to_cat_id[canonical_name])

    if unknown or len(index_to_cat_id) != len(name_to_cat_id) or set(index_to_cat_id.values()) != set(name_to_cat_id.values()):
        raise ValueError(f"{backend_name} checkpoint classes do not map one-to-one onto COCO ground truth.")
    return index_to_cat_id, ignored_indices


def normalize_coco_detection(box_xyxy: Any, score: float, category_id: int, image_width: int, image_height: int) -> Optional[Dict[str, Any]]:
    xmin = max(0.0, min(float(image_width), float(box_xyxy[0])))
    ymin = max(0.0, min(float(image_height), float(box_xyxy[1])))
    xmax = max(0.0, min(float(image_width), float(box_xyxy[2])))
    ymax = max(0.0, min(float(image_height), float(box_xyxy[3])))
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0.0 or height <= 0.0:
        return None
    return {"bbox": [xmin, ymin, width, height], "score": float(score), "category_id": int(category_id)}


class UltralyticsAdapter:
    def __init__(self, model_cfg: Dict[str, Any], weights_path: str, name_to_cat_id: Dict[str, int], device: str = "cuda:0", conf_thresh: float = 0.0, imgsz: int = 640, max_detections: int = 100):
        self.device = device
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.max_detections = max_detections
        self.nms_iou_threshold = model_cfg.get("nms_iou_threshold")

        model_type = str(model_cfg.get("model_type", "yolo")).lower()
        if model_type == "rtdetr":
            from ultralytics import RTDETR
            self.model = RTDETR(weights_path)
        elif model_type == "yolo":
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
        else:
            raise ValueError(f"model_type Ultralytics không được hỗ trợ: {model_type}")

        checkpoint_names = getattr(self.model, "names", {})
        if isinstance(checkpoint_names, list):
            checkpoint_names = {i: str(name) for i, name in enumerate(checkpoint_names)}
        else:
            checkpoint_names = {int(index): str(name) for index, name in checkpoint_names.items()}
        self.checkpoint_classes = checkpoint_names
        self.class_idx_to_cat_id, self.ignored_class_indices = build_class_mapping(checkpoint_names, name_to_cat_id, model_cfg, "Ultralytics")
        self.backend_max_detections = max(self.max_detections, 300) if self.ignored_class_indices else self.max_detections

    def predict_image(self, image_bgr: np.ndarray, orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
        predict_kwargs = {"source": image_bgr, "conf": self.conf_thresh, "imgsz": self.imgsz, "device": self.device, "verbose": False, "max_det": self.backend_max_detections}
        if self.nms_iou_threshold is not None:
            predict_kwargs["iou"] = float(self.nms_iou_threshold)
        res = self.model.predict(**predict_kwargs)[0]
        detections = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_idx in zip(boxes, scores, classes):
                if cls_idx in self.ignored_class_indices: continue
                if cls_idx not in self.class_idx_to_cat_id: raise ValueError(f"Không ánh xạ được class index {cls_idx}")
                detection = normalize_coco_detection(box, score, self.class_idx_to_cat_id[cls_idx], orig_w, orig_h)
                if detection is not None:
                    detections.append(detection)
        detections.sort(key=lambda detection: detection["score"], reverse=True)
        return detections[: self.max_detections]

    def get_torch_module(self):
        return self.model.model


class RFDETRAdapter:
    def __init__(self, model_cfg: Dict[str, Any], weights_path: str, class_names: List[str], name_to_cat_id: Dict[str, int], device: str = "cuda:0", conf_thresh: float = 0.0, imgsz: int = 640, max_detections: int = 100):
        import sys
        # Đảm bảo có thể import các module mở rộng
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        
        import rfdetr
        from src.rfdetr.lwdetr_dqk import RFDETRDQKEnhanced

        self.conf_thresh = conf_thresh
        self.max_detections = max_detections
        self.label_offset = int(model_cfg.get("label_offset", 0))
        self.checkpoint_classes = model_cfg.get("checkpoint_classes") or class_names
        indexed_names = {index: name for index, name in enumerate(self.checkpoint_classes)}
        self.class_idx_to_cat_id, self.ignored_class_indices = build_class_mapping(indexed_names, name_to_cat_id, model_cfg, "RF-DETR")
        
        model_class_name = model_cfg.get("model_class", "RFDETRMedium")
        if model_class_name == "RFDETRDQKEnhanced":
            import torch
            model_class = RFDETRDQKEnhanced
            # Khởi tạo mô hình với num_classes = 32
            self.model = model_class(num_classes=32)
            
            state_dict = torch.load(weights_path, map_location="cpu")
            if 'model' in state_dict:
                ckpt_weights = state_dict['model']
            elif 'state_dict' in state_dict:
                ckpt_weights = state_dict['state_dict']
            else:
                ckpt_weights = state_dict
                
            # Load trọng số vào model (Strict=False để xử lý các lớp thặng dư nhưng in cảnh báo nếu thiếu)
            missing, unexpected = self.model.load_state_dict(ckpt_weights, strict=False)
            print(f"[+] Loaded RFDETRDQKEnhanced weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            
            self.model.to(device)
            self.model.eval()
        else:
            model_class = getattr(rfdetr, model_class_name, None)
            if model_class is None: raise ValueError(f"Không tìm thấy class {model_class_name} trong package rfdetr")
            self.model = model_class.from_checkpoint(weights_path, device=device, resolution=int(imgsz))

        model_context = getattr(self.model, "model", getattr(self.model, "net", self.model))
        model_args = getattr(model_context, "args", None)
        self.background_class_id = int(getattr(model_args, "num_classes", len(self.checkpoint_classes)))
        self.backend_max_detections = int(getattr(model_args, "num_select", max(self.max_detections, 300)))
        self.imgsz = int(getattr(getattr(self.model, "model_config", None), "resolution", imgsz))

    def predict_image(self, image_bgr: np.ndarray, orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
        from PIL import Image
        import torch

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Nếu model tự custom thì chưa bọc hàm predict() giống như RFDETR API chuẩn, ta phải làm thủ công hoặc dùng wrapper
        if hasattr(self.model, "predict"):
            output = self.model.predict(Image.fromarray(image_rgb), threshold=self.conf_thresh)
            boxes = np.asarray(output.xyxy)
            scores = np.asarray(output.confidence)
            labels = np.asarray(output.class_id).astype(int)
        else:
            # Fallback manual inference
            import torchvision.transforms as T
            transforms = T.Compose([T.ToTensor(), T.Resize((self.imgsz, self.imgsz))])
            img_t = transforms(Image.fromarray(image_rgb)).unsqueeze(0).to(next(self.model.parameters()).device)
            with torch.no_grad():
                out = self.model(img_t)
            
            from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy
            scores_pred = out["pred_logits"].sigmoid()[0]
            boxes_pred = box_cxcywh_to_xyxy(out["pred_boxes"][0])
            topk_vals, topk_inds = torch.topk(scores_pred.max(-1)[0], self.backend_max_detections)
            boxes = boxes_pred[topk_inds].cpu().numpy()
            boxes[:, [0, 2]] *= orig_w
            boxes[:, [1, 3]] *= orig_h
            scores = topk_vals.cpu().numpy()
            labels = scores_pred[topk_inds].argmax(-1).cpu().numpy()

        foreground = labels != self.background_class_id
        class_indices = labels - self.label_offset
        ignored = np.isin(class_indices, list(self.ignored_class_indices))
        candidates = np.flatnonzero(foreground & ~ignored & (scores >= self.conf_thresh))
        order = candidates[np.argsort(-scores[candidates], kind="stable")][: self.max_detections]
        detections = []
        for box, score, raw_label in zip(boxes[order], scores[order], labels[order]):
            cls_idx = int(raw_label) - self.label_offset
            if cls_idx in self.ignored_class_indices: continue
            if cls_idx not in self.class_idx_to_cat_id: continue
            detection = normalize_coco_detection(box, score, self.class_idx_to_cat_id[cls_idx], orig_w, orig_h)
            if detection is not None:
                detections.append(detection)
        return detections

    def get_torch_module(self):
        import torch
        candidate = self.model
        for _ in range(3):
            if isinstance(candidate, torch.nn.Module):
                return candidate
            candidate = getattr(candidate, "model", None)
            if candidate is None:
                break
        raise TypeError("RF-DETR did not expose an underlying torch.nn.Module")


class BenchmarkEngine:
    @staticmethod
    def measure_complexity(adapter: Any, device: str = "cuda:0", imgsz: int = 640) -> Tuple[Optional[float], float]:
        import torch
        torch_model = adapter.get_torch_module()
        effective_device = torch.device(device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu")
        torch_model.eval().to(effective_device)

        gflops = None
        params_m = round(sum(p.numel() for p in torch_model.parameters()) / 1e6, 2)
        
        try:
            from thop import profile
            dummy = torch.randn(1, 3, imgsz, imgsz).to(effective_device)
            try:
                flops, _ = profile(torch_model, inputs=(dummy,), verbose=False)
            except Exception:
                flops, _ = profile(torch_model, inputs=([dummy[0]],), verbose=False)
            gflops = round(flops / 1e9, 2)
        except Exception as exc:
            print(f"  [WARNING] GFLOPs unavailable (THOP profiling failed): {exc}")

        return gflops, params_m

    @staticmethod
    def measure_latency_and_fps(adapter: Any, test_image_paths: List[str], device: str = "cuda:0", warmup: int = 10) -> Tuple[float, float]:
        import torch
        total_images = len(test_image_paths)
        if total_images == 0:
            return 0.0, 0.0
        use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

        for i in range(min(warmup, total_images)):
            img = cv2.imread(test_image_paths[i])
            adapter.predict_image(img, img.shape[1], img.shape[0])

        if use_cuda:
            torch.cuda.synchronize()

        elapsed = 0.0
        measured_images = 0
        for p in test_image_paths:
            img = cv2.imread(p)
            if use_cuda: torch.cuda.synchronize()
            start_time = time.perf_counter()
            adapter.predict_image(img, img.shape[1], img.shape[0])
            if use_cuda: torch.cuda.synchronize()
            elapsed += time.perf_counter() - start_time
            measured_images += 1

        latency_ms = round((elapsed / measured_images) * 1000, 2)
        fps = round(measured_images / elapsed, 2) if elapsed > 0 else 0.0
        return latency_ms, fps


class COCOBenchmarkEvaluator:
    @staticmethod
    def _operating_point_from_cocoeval(coco_eval, coco_gt, score_threshold: float, iou_threshold: float) -> Dict[str, Any]:
        iou_indexes = np.flatnonzero(np.isclose(coco_eval.params.iouThrs, iou_threshold))
        iou_index = int(iou_indexes[0])
        area_all = list(coco_eval.params.areaRng[0])
        max_det = int(coco_eval.params.maxDets[-1])
        counts = {int(cat_id): {"TP": 0, "FP": 0, "FN": 0} for cat_id in coco_eval.params.catIds}

        for eval_image in coco_eval.evalImgs:
            if eval_image is None: continue
            if list(eval_image["aRng"]) != area_all or int(eval_image["maxDet"]) != max_det: continue

            cat_id = int(eval_image["category_id"])
            if cat_id not in counts: continue
            scores = np.asarray(eval_image["dtScores"], dtype=np.float64)
            selected = scores >= score_threshold
            matches = np.asarray(eval_image["dtMatches"])[iou_index, selected]
            ignored_detections = np.asarray(eval_image["dtIgnore"])[iou_index, selected].astype(bool)
            valid_detections = ~ignored_detections
            true_positive_mask = (matches > 0) & valid_detections
            false_positive_mask = (matches == 0) & valid_detections

            valid_ground_truth = int(np.count_nonzero(~np.asarray(eval_image["gtIgnore"], dtype=bool)))
            matched_ground_truth_ids = {int(gt_id) for gt_id in matches[true_positive_mask] if int(gt_id) > 0}
            counts[cat_id]["TP"] += len(matched_ground_truth_ids)
            counts[cat_id]["FP"] += int(np.count_nonzero(false_positive_mask))
            counts[cat_id]["FN"] += max(0, valid_ground_truth - len(matched_ground_truth_ids))

        per_class = {}
        for cat_id, cls_counts in counts.items():
            tp, fp, fn = cls_counts["TP"], cls_counts["FP"], cls_counts["FN"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            cat_name = coco_gt.loadCats(cat_id)[0]["name"]
            per_class[cat_name] = {"Precision": round(precision, 4), "Recall": round(recall, 4), "F1": round(f1, 4), "TP": tp, "FP": fp, "FN": fn}

        tp = sum(item["TP"] for item in counts.values())
        fp = sum(item["FP"] for item in counts.values())
        fn = sum(item["FN"] for item in counts.values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "TP": tp, "FP": fp, "FN": fn,
            "score_threshold": score_threshold, "iou_threshold": iou_threshold, "source": "pycocotools.COCOeval.evalImgs",
            "per_class": per_class,
        }

    @staticmethod
    def evaluate(coco_gt, predictions_list: List[Dict[str, Any]], valid_cat_ids: List[int], image_ids: List[int], max_detections: int = 100, operating_score_threshold: float = 0.55, operating_iou_threshold: float = 0.50) -> Dict[str, Any]:
        from pycocotools.cocoeval import COCOeval
        if predictions_list:
            coco_dt = coco_gt.loadRes(predictions_list)
        else:
            from pycocotools.coco import COCO
            selected_image_ids = set(int(image_id) for image_id in image_ids)
            coco_dt = COCO()
            coco_dt.dataset = {
                "images": [image for image in coco_gt.dataset.get("images", []) if int(image["id"]) in selected_image_ids],
                "categories": [category for category in coco_gt.dataset.get("categories", []) if int(category["id"]) in set(int(cat_id) for cat_id in valid_cat_ids)],
                "annotations": [],
            }
            coco_dt.createIndex()
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.catIds = valid_cat_ids
        coco_eval.params.imgIds = sorted(image_ids)
        coco_eval.params.maxDets = [1, 10, int(max_detections)]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        operating_metrics = COCOBenchmarkEvaluator._operating_point_from_cocoeval(coco_eval, coco_gt, score_threshold=operating_score_threshold, iou_threshold=operating_iou_threshold)

        stats = coco_eval.stats
        map50_95 = float(stats[0])
        map50 = float(stats[1])
        map75 = float(stats[2])

        per_class_metrics = {}
        for idx_k, cat_id in enumerate(coco_eval.params.catIds):
            cat_name = coco_gt.loadCats(int(cat_id))[0]["name"]
            p_all = coco_eval.eval["precision"][:, :, idx_k, 0, 2]
            v_all = p_all[p_all > -1]
            cls_map50_95 = float(np.mean(v_all)) if len(v_all) > 0 else 0.0

            p_50 = coco_eval.eval["precision"][0, :, idx_k, 0, 2]
            v_50 = p_50[p_50 > -1]
            cls_map50 = float(np.mean(v_50)) if len(v_50) > 0 else 0.0

            r_cls = coco_eval.eval["recall"][:, idx_k, 0, 2]
            v_r = r_cls[r_cls > -1]
            cls_ar100 = float(np.mean(v_r)) if len(v_r) > 0 else 0.0

            per_class_metrics[cat_name] = {
                "Precision": operating_metrics["per_class"][cat_name]["Precision"],
                "Recall": operating_metrics["per_class"][cat_name]["Recall"],
                "F1": operating_metrics["per_class"][cat_name]["F1"],
                "TP": operating_metrics["per_class"][cat_name]["TP"],
                "FP": operating_metrics["per_class"][cat_name]["FP"],
                "FN": operating_metrics["per_class"][cat_name]["FN"],
                "mAP50": round(cls_map50, 4),
                "mAP50-95": round(cls_map50_95, 4),
                "AR100": round(cls_ar100, 4),
            }

        return {
            "Precision": operating_metrics["Precision"], "Recall": operating_metrics["Recall"], "F1": operating_metrics["F1"],
            "mAP50_95": round(map50_95, 4), "mAP50": round(map50, 4), "mAP75": round(map75, 4),
            "AP_small": round(float(stats[3]), 4), "AP_medium": round(float(stats[4]), 4), "AP_large": round(float(stats[5]), 4),
            "AR1": round(float(stats[6]), 4), "AR10": round(float(stats[7]), 4), "AR100": round(float(stats[8]), 4),
            "operating_point": {key: operating_metrics[key] for key in ("TP", "FP", "FN", "score_threshold", "iou_threshold", "source")},
            "per_class": per_class_metrics
        }


class ScientificReportVisualizer:
    @staticmethod
    def generate_all_reports(results_list: List[Dict[str, Any]], valid_categories: List[Dict[str, Any]], output_dir: str, chart_dpi: int = 300):
        os.makedirs(output_dir, exist_ok=True)
        class_names = [c["name"] for c in valid_categories]

        summary_rows = []
        for r in results_list:
            is_pending = "Chờ" in r.get("status", "")
            patience_val = r.get("Patience")
            patience_str = "-" if (patience_val is None or str(patience_val).strip() in ["", "null", "None"]) else str(patience_val)
            gflops_value = r.get("GFLOPs")

            summary_rows.append({
                "Mô hình": r["model_name"], "Precision": "-" if is_pending else f"{float(r['Precision']):.4f}", "Recall": "-" if is_pending else f"{float(r['Recall']):.4f}",
                "F1": "-" if is_pending else f"{float(r.get('F1', 0.0)):.4f}", "mAP@50": "-" if is_pending else f"{float(r['mAP50']):.4f}",
                "mAP@50-95": "-" if is_pending else f"{float(r['mAP50_95']):.4f}", "mAP@75": "-" if is_pending else f"{float(r.get('mAP75', 0.0)):.4f}",
                "AR@100": "-" if is_pending else f"{float(r.get('AR100', 0.0)):.4f}", "Patience": patience_str,
                "GFLOPs": "-" if is_pending or gflops_value is None else f"{float(gflops_value):.2f}",
                "Parameters": "-" if is_pending else f"{float(r['Parameters']):.2f}", "Latency": "-" if is_pending else f"{float(r['Latency']):.2f}",
                "FPS": "-" if is_pending else f"{float(r['FPS']):.2f}", "Trạng thái": r.get("status", "Hoàn tất")
            })

        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "summary_coco_comparison.csv")
        summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        md_path = os.path.join(output_dir, "summary_coco_comparison.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# BẢNG TỔNG HỢP SO SÁNH CÁC MÔ HÌNH OBJECT DETECTION\n\n")
            f.write(f"- **Đề tài**: Nghiên cứu khoa học - Nhận diện nguyên liệu nấu ăn\n")
            f.write(f"- **Thời gian**: {time.strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            f.write(dataframe_to_markdown(summary_df))

        cross_rows = []
        for cname in class_names:
            row_dict = {"Class Name": cname}
            for res in results_list:
                mname = res["model_name"]
                cdata = res.get("per_class", {}).get(cname, {})
                row_dict[f"{mname} (mAP50)"] = cdata.get("mAP50", "-")
                row_dict[f"{mname} (mAP50-95)"] = cdata.get("mAP50-95", "-")
            cross_rows.append(row_dict)

        cross_df = pd.DataFrame(cross_rows)
        cross_csv = os.path.join(output_dir, "all_models_per_class_comparison.csv")
        cross_df.to_csv(cross_csv, index=False, encoding="utf-8-sig")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            evaluated = [r for r in results_list if "Chờ" not in r.get("status", "")]
            if len(evaluated) > 0:
                m_names = [r["model_name"] for r in evaluated]
                m_50 = [float(r["mAP50"]) * 100 for r in evaluated]
                m_50_95 = [float(r["mAP50_95"]) * 100 for r in evaluated]

                x = np.arange(len(m_names))
                width = 0.35
                plt.figure(figsize=(max(10, len(m_names) * 2), 7), dpi=chart_dpi)
                sns.set_theme(style="whitegrid")
                plt.bar(x - width/2, m_50, width, label="mAP@50 (%)", color="#1f77b4")
                plt.bar(x + width/2, m_50_95, width, label="mAP@50-95 (%)", color="#ff7f0e")
                plt.ylabel("Độ chính xác (%)", fontsize=14, weight="bold")
                plt.title("Độ chính xác mAP@50 và mAP@50-95 chuẩn COCO Benchmark", fontsize=16, weight="bold", pad=15)
                plt.xticks(x, m_names, rotation=25, ha="right", fontsize=11, weight="bold")
                plt.legend(fontsize=12)
                plt.ylim(0, 105)

                for i in range(len(m_names)):
                    plt.text(x[i] - width/2, m_50[i] + 1.2, f"{m_50[i]:.1f}%", ha="center", fontsize=9, weight="bold")
                    plt.text(x[i] + width/2, m_50_95[i] + 1.2, f"{m_50_95[i]:.1f}%", ha="center", fontsize=9, weight="bold")

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "comparison_map_chart.png"), dpi=chart_dpi)
                plt.close()

        except Exception as e:
            print(f"  [VISUALIZER] Không thể xuất biểu đồ: {e}")

        print(f"\n[DONE] Đã lưu báo cáo khoa học tại: {os.path.abspath(output_dir)}")


def execute_common_coco_evaluation(dataset_dir: Optional[str] = None, output_dir: str = "Common_Evaluate_Results", target_models: Optional[List[str]] = None, config_data: Optional[Dict[str, Any]] = None, device: str = "cuda:0") -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    if config_data is None:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)

    rf_info = config_data.get("roboflow", {})
    api_key = rf_info.get("api_key", "") or get_roboflow_api_key(config_data)
    workspace_name = rf_info.get("workspace", "nckhcict2025")
    project_name = rf_info.get("project", "completed-project")
    version_num = rf_info.get("version", 5)
    dummy_cat = rf_info.get("dummy_category_name", "19-8-BoSungAnhChoCacLopBiTh-v3zo")

    bench_cfg = config_data.get("benchmark_settings", {})
    img_size = int(bench_cfg.get("imgsz", 640))
    conf_thresh = float(bench_cfg.get("conf_threshold", 0.0))
    operating_conf_thresh = float(bench_cfg.get("operating_conf_threshold", 0.55))
    operating_iou_thresh = float(bench_cfg.get("iou_threshold", 0.50))
    max_detections = int(bench_cfg.get("max_detections", 100))
    warmup_n = int(bench_cfg.get("warmup_runs", 10))
    chart_dpi = int(bench_cfg.get("chart_dpi", 300))

    all_models_dict = config_data.get("models", {})

    if target_models:
        eval_queue = {k: v for k, v in all_models_dict.items() if k in target_models}
    else:
        eval_queue = {model_key: model_cfg for model_key, model_cfg in all_models_dict.items() if model_cfg.get("enabled", False)}

    if not eval_queue:
        print("[NOTICE] Không có mô hình nào được bật. Đặt enabled=true cho ít nhất một mô hình trong models_config.json.")
        return []

    resolved_checkpoint_paths = {}
    for model_key, model_cfg in eval_queue.items():
        resolved_path, attempted_paths = resolve_checkpoint_path(model_key, model_cfg)
        if resolved_path:
            resolved_checkpoint_paths[model_key] = resolved_path
        else:
            print(f"[!] Warning: Missing checkpoint for {model_key}")

    print("\n" + "="*85)
    print(f"BẮT ĐẦU ĐÁNH GIÁ CHỌN LỌC {len(eval_queue)} MÔ HÌNH ĐƯỢC CHỈ ĐỊNH")
    print("="*85 + "\n")

    ds_manager = RoboflowCOCODatasetManager(api_key=api_key, dataset_root=dataset_dir, workspace=workspace_name, project=project_name, version=version_num, dummy_cat_name=dummy_cat)
    ds_manager.ensure_dataset(target_dir=dataset_dir)
    coco_gt, valid_categories, test_images = ds_manager.load_test_split()

    valid_cat_ids = ds_manager.valid_cat_ids
    class_names = ds_manager.class_names
    name_to_cat_id = ds_manager.name_to_cat_id
    test_image_paths = ds_manager.test_image_paths

    runtime_device = describe_runtime_device(device)
    protocol = build_protocol_metadata(bench_cfg, ds_manager.test_annotation_sha256, list(test_images.keys()), valid_categories, runtime_device)

    results_list = []

    for m_key, raw_model_cfg in eval_queue.items():
        m_cfg = dict(raw_model_cfg)
        disp_name = m_cfg.get("display_name", m_key)
        print("\n" + "-"*85)
        print(f"TIẾN HÀNH ĐÁNH GIÁ MÔ HÌNH: [{disp_name}]")
        print("-"*85)

        if m_key not in resolved_checkpoint_paths:
            continue
            
        actual_weights = resolved_checkpoint_paths[m_key]
        print(f"  [MODEL] Khởi tạo mô hình từ: {actual_weights}")
        model_img_size = int(m_cfg.get("imgsz", img_size))

        if m_cfg["family"] == "ultralytics":
            adapter = UltralyticsAdapter(m_cfg, actual_weights, name_to_cat_id, device=device, conf_thresh=conf_thresh, imgsz=model_img_size, max_detections=max_detections)
        elif m_cfg["family"] == "rfdetr":
            adapter = RFDETRAdapter(m_cfg, actual_weights, class_names, name_to_cat_id, device=device, conf_thresh=conf_thresh, imgsz=model_img_size, max_detections=max_detections)
        else:
            raise ValueError(f"Unsupported model family: {m_cfg['family']}")

        effective_img_size = int(getattr(adapter, "imgsz", model_img_size))
        
        print("  [BENCHMARK] Đo GFLOPs và Parameters...")
        gflops, params_m = BenchmarkEngine.measure_complexity(adapter, device=device, imgsz=effective_img_size)

        print("  [BENCHMARK] Đo Latency & FPS...")
        latency_ms, fps = BenchmarkEngine.measure_latency_and_fps(adapter, test_image_paths, device=device, warmup=warmup_n)

        print(f"  [INFERENCE] Đang suy luận trên {len(test_images)} ảnh test...")
        coco_predictions = []
        for img_id, info in test_images.items():
            img_mat = cv2.imread(info["path"])
            actual_h, actual_w = img_mat.shape[:2]
            preds = adapter.predict_image(img_mat, actual_w, actual_h)
            for p in preds:
                coco_predictions.append({"image_id": int(img_id), "category_id": int(p["category_id"]), "bbox": p["bbox"], "score": float(p["score"])})

        print("  [COCOEVAL] Tính toán metric COCO...")
        eval_metrics = COCOBenchmarkEvaluator.evaluate(coco_gt, coco_predictions, valid_cat_ids, list(test_images.keys()), max_detections=max_detections, operating_score_threshold=operating_conf_thresh, operating_iou_threshold=operating_iou_thresh)

        res_item = {
            "model_key": m_key, "model_name": disp_name, "Precision": eval_metrics["Precision"], "Recall": eval_metrics["Recall"], "F1": eval_metrics["F1"],
            "mAP50": eval_metrics["mAP50"], "mAP50_95": eval_metrics["mAP50_95"], "mAP75": eval_metrics["mAP75"],
            "Patience": m_cfg.get("patience", 10), "GFLOPs": gflops, "Parameters": params_m, "Latency": latency_ms, "FPS": fps,
            "status": "Hoàn tất đánh giá", "protocol": protocol, "operating_point": eval_metrics["operating_point"],
            "per_class": eval_metrics["per_class"],
        }
        results_list.append(res_item)

        res_json_file = os.path.join(output_dir, f"result_{m_key}.json")
        with open(res_json_file, "w", encoding="utf-8") as rf:
            json.dump(res_item, rf, ensure_ascii=False, indent=2)

        del adapter
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception:
            pass

    ScientificReportVisualizer.generate_all_reports(results_list, valid_categories, output_dir, chart_dpi=chart_dpi)
    return results_list


@app.function(
    image=image,
    gpu=MODAL_GPU,
    volumes={
        "/data": dataset_volume,
        "/runs": runs_volume,
    },
    timeout=MODAL_TIMEOUT,
)
def evaluate_models_cloud(model_names: Optional[List[str]] = None, config_dict: Optional[dict] = None):
    cfg = config_dict or load_config()
    m_cfg = cfg.get("modal_settings", {})
    cloud_dataset = m_cfg.get("cloud_dataset_dir", "/data/dataset/Completed-Project-5")
    cloud_output = os.path.join(m_cfg.get("cloud_output_dir", "/data/runs"), "Common_Eval_Results")

    execute_common_coco_evaluation(dataset_dir=cloud_dataset, output_dir=cloud_output, target_models=model_names, config_data=cfg, device="cuda:0")

    files_data = {}
    if os.path.exists(cloud_output):
        for fname in os.listdir(cloud_output):
            fp = os.path.join(cloud_output, fname)
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    files_data[fname] = f.read()

    dataset_volume.commit()
    return files_data


@app.local_entrypoint()
def evaluate_my_models():
    local_cfg = load_config()
    api_key = get_roboflow_api_key(local_cfg)
    if "roboflow" not in local_cfg: local_cfg["roboflow"] = {}
    local_cfg["roboflow"]["api_key"] = api_key
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print(f"[CLOUD] Bắt đầu đánh giá các mô hình đã bật trên Modal Cloud ({MODAL_GPU})...")
    files = evaluate_models_cloud.remote(config_dict=local_cfg)

    target_sub_dir = os.path.join(CURRENT_DIR, "..", "..", "Results", f"RESULT_{timestamp}")
    os.makedirs(target_sub_dir, exist_ok=True)
    
    for fname, data in files.items():
        out_p = os.path.join(target_sub_dir, fname)
        with open(out_p, "wb") as f:
            f.write(data)

    print(f"\n[SUCCESS] Hoàn tất! Đã lưu toàn bộ kết quả phân loại vào thư mục: {target_sub_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="")
    args = parser.parse_args()
    execute_common_coco_evaluation(dataset_dir=args.data_dir if args.data_dir else None)