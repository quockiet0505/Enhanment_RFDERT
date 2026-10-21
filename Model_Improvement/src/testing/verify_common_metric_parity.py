"""Re-evaluate saved predictions with the team's unmodified COCO evaluator class."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO


def verify(result_dir, reference_source, annotation):
    source = Path(reference_source).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    evaluator = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "COCOBenchmarkEvaluator")
    # Execute only the evaluator class; do not initialize the other script's Modal
    # app, images, config, downloads or model adapters.
    scope = {"np": np, "Any": Any, "Dict": Dict, "List": List, "Optional": Optional, "Tuple": Tuple}
    exec(compile(ast.Module(body=[evaluator], type_ignores=[]), str(reference_source), "exec"), scope)
    result_dir = Path(result_dir)
    stored = json.loads((result_dir / "result_RFDETR_DQK_Legacy.json").read_text(encoding="utf-8"))
    with gzip.open(result_dir / "predictions_RFDETR_DQK_Legacy.json.gz", "rt", encoding="utf-8") as file:
        predictions = json.load(file)
    raw_annotations = Path(annotation).read_bytes()
    if hashlib.sha256(raw_annotations).hexdigest() != stored["protocol"]["annotation_sha256"]:
        raise ValueError("Parity annotation differs from the annotation used in inference")
    gt = COCO(str(annotation))
    cat_ids = [category["id"] for category in stored["protocol"]["categories"]]
    calculated = scope["COCOBenchmarkEvaluator"].evaluate(
        gt, predictions, cat_ids, sorted(gt.getImgIds()), max_detections=100,
        operating_score_threshold=0.55, operating_iou_threshold=0.5)
    metrics = ["Precision", "Recall", "F1", "mAP50", "mAP50_95", "mAP75", "AR100"]
    differences = {key: float(calculated[key]) - float(stored[key]) for key in metrics}
    if any(abs(value) > 1e-8 for value in differences.values()):
        raise AssertionError(f"Metric parity failed: {differences}")
    if calculated["per_class"] != stored["per_class"]:
        raise AssertionError("Per-class metric parity failed")
    if calculated["operating_point"] != stored["operating_point"]:
        raise AssertionError("Operating-point count parity failed")
    report = {"passed": True, "reference_source": str(reference_source),
              "reference_source_sha256": hashlib.sha256(source.replace("\r\n", "\n").encode()).hexdigest(),
              "annotation_sha256": hashlib.sha256(raw_annotations).hexdigest(),
              "num_images": len(gt.getImgIds()), "num_predictions": len(predictions),
              "metric_differences": differences, "all_32_per_class_metrics_equal": True,
              "operating_point_equal": True}
    (result_dir / "common_evaluator_parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--reference-source", required=True)
    parser.add_argument("--annotation", required=True)
    args = parser.parse_args()
    verify(args.result_dir, args.reference_source, args.annotation)
