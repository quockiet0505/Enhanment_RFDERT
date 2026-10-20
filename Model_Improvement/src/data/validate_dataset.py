"""Dataset Validator Module
Kiểm tra cấu trúc, số lượng ảnh, số lượng annotation, phân bố class và train/val/test splits.
"""
import os
import yaml
import json
from pathlib import Path
from collections import defaultdict


def validate_yolo_dataset(data_yaml_path: str) -> dict:
    """Kiểm tra tính hợp lệ của dataset định dạng YOLO."""
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_yaml_path).parent
    names = data_cfg.get("names", {})
    if isinstance(names, list):
        class_map = {i: name for i, name in enumerate(names)}
    else:
        class_map = {int(k): v for k, v in names.items()}

    splits = ["train", "val", "test"]
    stats = {
        "classes": class_map,
        "num_classes": len(class_map),
        "splits": {},
        "class_distribution": defaultdict(lambda: {"train": 0, "val": 0, "test": 0}),
    }

    for split in splits:
        split_key = "val" if split == "val" and "val" in data_cfg else ("valid" if "valid" in data_cfg else split)
        split_rel = data_cfg.get(split_key) or data_cfg.get(split)
        
        if not split_rel:
            print(f"[!] Split {split} not found in {data_yaml_path}")
            continue

        split_path = dataset_root / split_rel if not Path(split_rel).is_absolute() else Path(split_rel)
        # Check images directory
        img_dir = split_path if split_path.is_dir() else split_path.parent / "images"
        if not img_dir.exists():
            img_dir = split_path
            
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        
        # Check labels directory
        label_dir = img_dir.parent / "labels" if (img_dir.parent / "labels").exists() else img_dir
        
        total_annotations = 0
        for img_file in images:
            lbl_file = label_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                with open(lbl_file, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            cname = class_map.get(cid, f"class_{cid}")
                            stats["class_distribution"][cname][split] += 1
                            total_annotations += 1

        stats["splits"][split] = {
            "image_count": len(images),
            "annotation_count": total_annotations,
            "image_dir": str(img_dir),
        }

    print("=== DATASET VALIDATION REPORT ===")
    print(f"Total Classes: {stats['num_classes']}")
    for split, info in stats["splits"].items():
        print(f"Split [{split}]: {info['image_count']} images, {info['annotation_count']} annotations")
    
    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        validate_yolo_dataset(sys.argv[1])