"""Roboflow Dataset Downloader Module
Tải dataset từ Roboflow theo format YOLO và COCO cho RT-DETR và RF-DETR.
Hỗ trợ tái sử dụng dữ liệu đã tải (dùng nhiều lần) và cho phép đổi version / force re-download khi cần.
"""
import os
import sys
import shutil
from pathlib import Path
from roboflow import Roboflow


def download_dataset(
    api_key: str = None,
    workspace: str = "nckhcict2025",
    project_name: str = "completed-project",
    version_num: int = 5,
    target_dir_yolo: str = None,
    target_dir_coco: str = None,
    download_coco: bool = True,
    force_redownload: bool = False,
):
    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY is not set. Please provide it via argument or environment variable."
        )

    # Tự động tạo đường dẫn thư mục theo version nếu không truyền cứng
    if target_dir_yolo is None:
        target_dir_yolo = f"/vol/datasets/{project_name}-{version_num}/yolo"
    if target_dir_coco is None:
        target_dir_coco = f"/vol/datasets/{project_name}-{version_num}/coco"

    # Nếu người dùng yêu cầu force re-download -> xóa bản cũ
    if force_redownload:
        print(f"[!] Force redownload: Đang xóa bản cũ của version {version_num}...")
        if os.path.exists(target_dir_yolo):
            shutil.rmtree(target_dir_yolo, ignore_errors=True)
        if os.path.exists(target_dir_coco):
            shutil.rmtree(target_dir_coco, ignore_errors=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)

    print(f"[*] Roboflow dataset: {workspace}/{project_name}/v{version_num}")
    
    # 1. Download YOLO format (for RT-DETR)
    yolo_yaml = Path(target_dir_yolo) / "data.yaml"
    if not yolo_yaml.exists():
        print(f"[*] [DOWNLOAD MỚI] Đang tải YOLO format vào {target_dir_yolo}...")
        yolo_dataset = version.download("yolov8", location=target_dir_yolo)
        print(f"[+] YOLO downloaded to: {target_dir_yolo}")
    else:
        print(f"[=] [TÁI SỬ DỤNG] YOLO dataset đã tồn tại sẵn tại: {target_dir_yolo} (Không cần tải lại)")

    # 2. Download COCO format (for RF-DETR) if requested
    if download_coco:
        coco_train_json = Path(target_dir_coco) / "train" / "_annotations.coco.json"
        if not coco_train_json.exists():
            print(f"[*] [DOWNLOAD MỚI] Đang tải COCO format vào {target_dir_coco}...")
            if os.path.exists(target_dir_coco):
                shutil.rmtree(target_dir_coco, ignore_errors=True)
            coco_dataset = version.download("coco", location=target_dir_coco)
            print(f"[+] COCO downloaded to: {target_dir_coco}")
        else:
            print(f"[=] [TÁI SỬ DỤNG] COCO dataset đã tồn tại sẵn tại: {target_dir_coco} (Không cần tải lại)")

        # Xóa file data.yaml trong thư mục COCO nếu có để tránh RF-DETR nhầm lẫn với YOLO format
        stray_yaml_in_coco = Path(target_dir_coco) / "data.yaml"
        if stray_yaml_in_coco.exists():
            try:
                stray_yaml_in_coco.unlink()
                print(f"[+] Đã xóa file data.yaml thừa trong {target_dir_coco}")
            except Exception as ce:
                print(f"[!] Warning deleting stray yaml in coco: {ce}")

    # Chuẩn hóa đường dẫn tuyệt đối trong data.yaml của YOLO
    yolo_yaml_file = Path(target_dir_yolo) / "data.yaml"
    if yolo_yaml_file.exists():
        try:
            import yaml
            with open(yolo_yaml_file, "r", encoding="utf-8") as yf:
                data_doc = yaml.safe_load(yf) or {}
            
            # Gán path tuyệt đối rõ ràng cho Ultralytics và sửa train/val/test
            abs_yolo_dir = str(Path(target_dir_yolo).resolve())
            data_doc["path"] = abs_yolo_dir
            if "train" in data_doc and "../" in str(data_doc["train"]):
                data_doc["train"] = str(data_doc["train"]).replace("../", "")
            if "val" in data_doc and "../" in str(data_doc["val"]):
                data_doc["val"] = str(data_doc["val"]).replace("../", "")
            if "valid" in data_doc and "../" in str(data_doc["valid"]):
                data_doc["valid"] = str(data_doc["valid"]).replace("../", "")
            if "test" in data_doc and "../" in str(data_doc["test"]):
                data_doc["test"] = str(data_doc["test"]).replace("../", "")

            with open(yolo_yaml_file, "w", encoding="utf-8") as yf:
                yaml.dump(data_doc, yf, default_flow_style=False)
            print(f"[+] Đã chuẩn hóa 'path' & 'train/val/test' trong {yolo_yaml_file}")
        except Exception as ye:
            print(f"[!] Warning fixing data.yaml: {ye}")

    return {
        "yolo_path": target_dir_yolo,
        "coco_path": target_dir_coco if download_coco else None,
        "version": version_num,
    }


if __name__ == "__main__":
    download_dataset()