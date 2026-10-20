"""RF-DETR Evaluation Module
Đánh giá mô hình RF-DETR trên tập test và xuất báo cáo:
- Precision, Recall, mAP50, mAP50-95, F1
- Sinh Ma trận nhầm lẫn (Confusion Matrix counts & normalized %)
- Đo đạc phần cứng: Parameters, GFLOPs, Latency (ms), FPS trên GPU A100
- Xuất báo cáo markdown & JSON
"""
import os
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.common.metrics_reporter import MetricsReporter
from src.common.benchmark import measure_hardware_benchmark
from src.rfdetr.confusion_matrix import compute_rfdetr_confusion_matrix


def evaluate_rfdetr(
    checkpoint_path: str = "/vol/checkpoints/rfdetr/best.pth",
    dataset_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_yaml_path: Optional[str] = None,
    config_path: str = "configs/rfdetr.yaml",
    output_dir: str = "/vol/outputs/rfdetr_eval",
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Tìm checkpoint hợp lệ
    if not os.path.exists(checkpoint_path):
        for alt in [
            "/vol/checkpoints/rfdetr/checkpoint_best_total.pth",
            "/vol/checkpoints/rfdetr/checkpoint_best_ema.pth",
            "/vol/checkpoints/rfdetr/checkpoint_best_regular.pth",
            "/vol/checkpoints/rfdetr/best.pth",
            "/vol/checkpoints/rfdetr/last.ckpt",
            "/vol/checkpoints/rfdetr/checkpoint.pth",
        ]:
            if os.path.exists(alt):
                checkpoint_path = alt
                break

    # 2. Chuẩn hóa dataset_dir từ các alias (dataset_dir, data_dir, data_yaml_path)
    if not dataset_dir:
        if data_dir:
            dataset_dir = data_dir
        elif data_yaml_path:
            if os.path.isfile(data_yaml_path):
                dataset_dir = os.path.dirname(data_yaml_path)
            else:
                dataset_dir = data_yaml_path
        else:
            candidates = [
                "/vol/datasets/completed-project-5/coco",
                "/vol/datasets/completed-project-5/yolo",
                "/vol/datasets/completed-project-1/coco",
                "/vol/datasets/completed-project-1/yolo",
            ]
            for c in candidates:
                if os.path.exists(c):
                    dataset_dir = c
                    break
            if not dataset_dir:
                dataset_dir = "/vol/datasets/completed-project-5/coco"

    # Auto-switch nếu truyền nhầm coco/yolo
    has_coco = os.path.exists(os.path.join(dataset_dir, "train", "_annotations.coco.json"))
    has_yolo = os.path.exists(os.path.join(dataset_dir, "data.yaml")) and os.path.exists(os.path.join(dataset_dir, "train", "images"))
    if not has_coco and not has_yolo:
        if "coco" in dataset_dir:
            yolo_alt = dataset_dir.replace("coco", "yolo")
            if os.path.exists(yolo_alt) and os.path.exists(os.path.join(yolo_alt, "data.yaml")):
                print(f"[*] [AUTO SWITCH] Chuyển tự động sang thư mục YOLO hợp lệ: {yolo_alt}")
                dataset_dir = yolo_alt
        elif "yolo" in dataset_dir:
            coco_alt = dataset_dir.replace("yolo", "coco")
            if os.path.exists(coco_alt) and os.path.exists(os.path.join(coco_alt, "train", "_annotations.coco.json")):
                print(f"[*] [AUTO SWITCH] Chuyển tự động sang thư mục COCO hợp lệ: {coco_alt}")
                dataset_dir = coco_alt

    # Đọc cấu hình
    eval_conf = 0.01
    cm_conf = 0.25
    imgsz = 640
    for cp in [config_path, "/root/configs/rfdetr.yaml", "configs/rfdetr.yaml"]:
        if os.path.exists(cp):
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                eval_conf = float(cfg.get("evaluation", {}).get("eval_conf", eval_conf))
                cm_conf = float(cfg.get("evaluation", {}).get("cm_conf", cm_conf))
                imgsz = int(cfg.get("training", {}).get("imgsz", imgsz))
                break
            except Exception:
                pass

    print(f"[*] Evaluating RF-DETR using checkpoint: {checkpoint_path} on dataset: {dataset_dir} (eval_conf={eval_conf}, cm_conf={cm_conf}, imgsz={imgsz})...")

    # 3. Phân biệt COCO format và YOLO format
    is_coco = False
    coco_test_json = os.path.join(dataset_dir, "test", "_annotations.coco.json")
    coco_train_json = os.path.join(dataset_dir, "train", "_annotations.coco.json")
    if os.path.exists(coco_test_json) or os.path.exists(coco_train_json):
        is_coco = True

    if is_coco:
        stray_yaml = os.path.join(dataset_dir, "data.yaml")
        if os.path.exists(stray_yaml):
            try:
                os.remove(stray_yaml)
                print(f"[+] Đã loại bỏ file {stray_yaml} để RF-DETR nạp đúng chuẩn COCO JSON format.")
            except Exception as e:
                print(f"[!] Warning removing stray data.yaml: {e}")

    from rfdetr import RFDETRMedium
    
    # 32 classes list
    class_names = [
        "beef", "bellpepper", "bittergourd", "bottlegourd", "broccoli",
        "cabbage", "carrot", "cauliflower", "chayote", "chicken",
        "chickenegg", "chickenleg", "chickenwin", "corn", "cucumber",
        "duckegg", "eggplant", "garlic", "ginger", "jicama",
        "okra", "onion", "pork", "potato", "pumpkin",
        "radish", "scallion", "shrimp", "spongegourd", "sweetpotato",
        "tofu", "tomato"
    ]

    model = None
    if hasattr(RFDETRMedium, "from_checkpoint"):
        try:
            model = RFDETRMedium.from_checkpoint(checkpoint_path)
            print(f"[+] Successfully loaded RFDETRMedium using from_checkpoint({checkpoint_path})")
        except Exception as e:
            print(f"[*] from_checkpoint fallback ({e})...")

    if model is None:
        try:
            model = RFDETRMedium(num_classes=32, pretrain_weights=checkpoint_path, resolution=imgsz)
        except Exception:
            try:
                model = RFDETRMedium(num_classes=32, pretrain_weights=checkpoint_path)
            except Exception:
                model = RFDETRMedium(num_classes=32)

    eval_kwargs = {
        "dataset_dir": dataset_dir,
        "split": "test",
        "output_dir": output_dir,
    }
    print(f"[*] Calling model.evaluate with: {eval_kwargs}")
    
    results = {}
    try:
        results = model.evaluate(**eval_kwargs)
    except Exception as e:
        print(f"[!] Warning model.evaluate failed ({e}), checking metrics.csv / checkpoint logs...")
        import traceback
        traceback.print_exc()

        # Fallback thử evaluate trên thư mục đối ứng (yolo <-> coco)
        alt_dataset_dir = None
        if "coco" in dataset_dir and os.path.exists(dataset_dir.replace("coco", "yolo")):
            alt_dataset_dir = dataset_dir.replace("coco", "yolo")
        elif "yolo" in dataset_dir and os.path.exists(dataset_dir.replace("yolo", "coco")):
            alt_dataset_dir = dataset_dir.replace("yolo", "coco")
            
        if alt_dataset_dir and alt_dataset_dir != dataset_dir:
            try:
                print(f"[*] Fallback: Thử evaluate trên thư mục đối ứng {alt_dataset_dir}...")
                results = model.evaluate(dataset_dir=alt_dataset_dir, split="test", output_dir=output_dir)
            except Exception as e2:
                print(f"[!] Fallback evaluate failed: {e2}")

    print(f"[*] Evaluation returned: {results}")

    # Nếu model.evaluate thất bại, thông báo lỗi chứ không lấy nhầm dữ liệu validation
    if not results:
        print("[!] model.evaluate không trả về kết quả, tiến hành tính toán trực tiếp từ Confusion Matrix và COCOeval...")

    # 4. Thực hiện suy luận toàn diện trên tập Test: sinh Confusion Matrix & tính toán COCOeval thực tế
    cm_paths = {}
    cm_res = {}
    try:
        cm_res = compute_rfdetr_confusion_matrix(
            model=model,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            class_names=class_names,
            conf_threshold=cm_conf,
            eval_conf=eval_conf,
            iou_threshold=0.5,
            max_test_samples=None,  # Đánh giá toàn bộ 1,531 ảnh của tập Test
        )
        if isinstance(cm_res, dict):
            cm_paths = {
                "confusion_matrix.png": cm_res.get("confusion_matrix", os.path.join(output_dir, "confusion_matrix.png")),
                "confusion_matrix_normalized.png": cm_res.get("confusion_matrix_normalized", os.path.join(output_dir, "confusion_matrix_normalized.png")),
                "confusion_matrix.csv": cm_res.get("confusion_matrix_csv", os.path.join(output_dir, "confusion_matrix.csv")),
                "confusion_matrix_normalized.csv": cm_res.get("confusion_matrix_normalized_csv", os.path.join(output_dir, "confusion_matrix_normalized.csv")),
                "per_class_metrics.csv": cm_res.get("per_class_csv", os.path.join(output_dir, "per_class_metrics.csv")),
            }
    except Exception as e:
        print(f"[!] Warning generating RF-DETR confusion matrix & COCOeval: {e}")
        import traceback
        traceback.print_exc()

    # Trích xuất kết quả per_class_metrics và coco_overall từ cm_res
    per_class_metrics = cm_res.get("per_class", []) if isinstance(cm_res, dict) else []
    coco_overall = cm_res.get("overall", {}) if isinstance(cm_res, dict) else {}

    # Kiểm tra tính toàn vẹn của kết quả COCOeval & Confusion Matrix
    if not per_class_metrics:
        raise RuntimeError(
            "[LỖI ĐÁNH GIÁ] Không trích xuất được per_class_metrics từ kết quả COCOeval! "
            "Bắt buộc phải tính toán thành công trên tập Test, không dùng dữ liệu dự phòng."
        )

    # Xác định overall metrics: Ưu tiên lấy trực tiếp từ kết quả chính thống của model.evaluate()
    if results and "test/mAP_50_95" in results:
        overall_metrics = {
            "precision": float(results.get("test/precision", 0.0)),
            "recall": float(results.get("test/recall", 0.0)),
            "map50": float(results.get("test/mAP_50", 0.0)),
            "map50_95": float(results.get("test/mAP_50_95", 0.0)),
        }
        print(f"[+] Sử dụng kết quả chính thống từ model.evaluate(): {overall_metrics}")
    elif coco_overall and "map50_95" in coco_overall:
        overall_metrics = {
            "precision": float(coco_overall.get("precision", 0.0)),
            "recall": float(coco_overall.get("recall", 0.0)),
            "map50": float(coco_overall.get("map50", 0.0)),
            "map50_95": float(coco_overall.get("map50_95", 0.0)),
        }
    else:
        raise RuntimeError("[LỖI] Không tìm thấy kết quả đánh giá hợp lệ từ model.evaluate() hay COCOeval!")

    # 5. Đo đạc Benchmark Phần cứng (GFLOPs, Latency ms, FPS)
    hw_bench = measure_hardware_benchmark(
        model_obj=model.model if hasattr(model, "model") else model,
        model_name="RF-DETR",
        imgsz=imgsz,
        warmup_runs=20,
        num_runs=100,
    )

    reporter = MetricsReporter(output_dir=output_dir)
    md_report = reporter.save_test_report(
        model_name="RF-DETR",
        overall_metrics=overall_metrics,
        per_class_metrics=per_class_metrics,
        hardware_benchmark=hw_bench,
        confusion_matrix_paths=cm_paths,
    )
    return overall_metrics, md_report


if __name__ == "__main__":
    evaluate_rfdetr()