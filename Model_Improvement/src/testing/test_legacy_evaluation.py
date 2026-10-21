"""Small semantic tests for the COCO evaluator and checkpoint label conversion."""
import contextlib
import copy
import io
import unittest
from types import SimpleNamespace

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.testing.common_test import COCOBenchmarkEvaluator, validate_configuration, load_config
from src.testing.rfdetr_legacy_adapter import RFDETRAdapter


class EvaluationChecks(unittest.TestCase):
    def setUp(self):
        self.gt = COCO()
        # Sparse category IDs, plus an empty image and an absent category.
        self.gt.dataset = {"info": {}, "images": [{"id": 1, "width": 100, "height": 100},
                            {"id": 2, "width": 100, "height": 100}],
            "categories": [{"id": 7, "name": "food"}, {"id": 42, "name": "absent"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20],
                             "area": 400, "iscrowd": 0}]}
        self.gt.createIndex()
        self.perfect = {"image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20], "score": 0.9}

    def evaluate(self, predictions):
        return COCOBenchmarkEvaluator.evaluate(self.gt, predictions, [7, 42], [1, 2])

    def test_perfect_and_direct_coco_parity(self):
        result = self.evaluate([self.perfect])
        self.assertAlmostEqual(result["mAP50_95"], 1.0)
        self.assertEqual((result["Precision"], result["Recall"], result["F1"]), (1., 1., 1.))
        direct = COCOeval(self.gt, self.gt.loadRes([self.perfect]), "bbox")
        direct.params.imgIds = [1, 2]
        direct.params.catIds = [7, 42]
        direct.params.maxDets = [1, 10, 100]
        direct.evaluate(); direct.accumulate(); direct.summarize()
        for key, index in [("mAP50_95", 0), ("mAP50", 1), ("mAP75", 2), ("AR100", 8)]:
            self.assertAlmostEqual(result[key], float(direct.stats[index]), places=4)

    def test_empty_predictions(self):
        result = self.evaluate([])
        self.assertEqual(result["mAP50_95"], 0.)
        self.assertEqual(result["operating_point"]["FN"], 1)
        self.assertEqual(result["operating_point"]["FP"], 0)

    def test_duplicate_detection_is_false_positive(self):
        duplicate = dict(self.perfect, score=0.8)
        result = self.evaluate([self.perfect, duplicate])
        self.assertEqual(result["operating_point"]["TP"], 1)
        self.assertEqual(result["operating_point"]["FP"], 1)
        self.assertEqual(result["Precision"], 0.5)

    def test_wrong_class(self):
        result = self.evaluate([dict(self.perfect, category_id=42)])
        self.assertEqual(result["mAP50_95"], 0.)
        self.assertEqual(result["operating_point"]["FP"], 1)
        self.assertEqual(result["operating_point"]["FN"], 1)

    def test_empty_image_false_positive_lowers_ap(self):
        fp = dict(self.perfect, image_id=2, score=0.99)
        result = self.evaluate([fp, self.perfect])
        self.assertAlmostEqual(result["mAP50_95"], 0.5)
        self.assertEqual(result["Precision"], 0.5)

    def test_ap_is_not_thresholded_at_operating_confidence(self):
        result = self.evaluate([dict(self.perfect, score=0.4)])
        self.assertAlmostEqual(result["mAP50_95"], 1.)
        self.assertEqual(result["Recall"], 0.)
        self.assertEqual(result["operating_point"]["FN"], 1)

    def test_crowd_detections_are_ignored(self):
        self.gt.dataset["annotations"].append({"id": 2, "image_id": 2, "category_id": 7,
                      "bbox": [0, 0, 50, 50], "area": 2500, "iscrowd": 1})
        self.gt.createIndex()
        result = self.evaluate([self.perfect, dict(self.perfect, image_id=2)])
        self.assertEqual(result["operating_point"]["TP"], 1)
        self.assertEqual(result["operating_point"]["FP"], 0)
        self.assertEqual(result["operating_point"]["FN"], 0)

    def test_protocol_rejects_changed_confidence(self):
        config = copy.deepcopy(load_config())
        validate_configuration(config)
        config["benchmark_settings"]["conf_threshold"] = 0.5
        with self.assertRaises(ValueError):
            validate_configuration(config)

    def adapter(self, labels):
        adapter = RFDETRAdapter.__new__(RFDETRAdapter)
        adapter.conf_thresh = 0.
        adapter.background_class_id = 2
        adapter.max_detections = 1
        adapter.class_idx_to_cat_id = {0: 7, 1: 42}
        adapter.native_detections = lambda image: SimpleNamespace(
            xyxy=np.array([[-5, 10, 120, 30], [0, 0, 5, 5], [10, 10, 20, 20]], dtype=float),
            confidence=np.array([0.8, 0.99, 0.6]), class_id=np.array(labels))
        return adapter

    def test_mapping_background_global_cap_and_clipping(self):
        result = self.adapter([1, 2, 0]).predict_image(np.zeros((100, 100, 3), np.uint8), 100, 100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category_id"], 42)
        self.assertEqual(result[0]["bbox"], [0., 10., 100., 20.])

    def test_unknown_labels_fail(self):
        with self.assertRaises(ValueError):
            self.adapter([90, 2, 0]).predict_image(np.zeros((100, 100, 3), np.uint8), 100, 100)


def run_checks():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationChecks)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    print(f"[CHECKS] {result.testsRun} COCO/mapping/protocol tests passed", flush=True)
    return {"tests_run": result.testsRun, "passed": True, "log": log.getvalue()}


if __name__ == "__main__":
    run_checks()
