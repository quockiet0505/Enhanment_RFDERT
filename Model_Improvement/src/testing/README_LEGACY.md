# RF-DETR DQK legacy checkpoint evaluation

Run from the DQK directory:

```powershell
python -m modal run Model_Improvement/src/testing/common_test.py::evaluate_my_models
```

The configuration in `Model_Improvement/configs/models_config.json` enables only
`RFDETR_DQK_Legacy`. This evaluates the old `best.pth` as **RFDETRMedium without
DQK modules**, on the existing `food-detection-training` dataset volume. It does
not train a model. The baseline adapter rejects enhanced checkpoints. New enhanced
training/testing uses the separate adapter and versioned pipeline documented in
`References/HUONG_DAN_TRAIN_TEST_DQK_V1.md` at the DQK root.

Before evaluation, the runner checks the exact checkpoint SHA256, training class
names, all classification-head sizes, RF-DETR version and a strict state-dict
load. Every loaded checkpoint tensor is compared with the source tensor. It uses
the native RF-DETR 1.10.1 prediction API for RGB conversion, resize,
normalization and query-class postprocessing. A forward hook checks native
preprocessing against the explicit transform, and native boxes/scores/labels
against its postprocessor on a validation image.

Ten synthetic COCO/mapping/protocol cases run before test inference. The full
evaluation requires the audited test annotation hash, 32 classes and 1,481
images. Inference uses 640 pixels and batch size 1, threshold 0.0 for AP,
threshold 0.55 and IoU 0.5 for micro P/R/F1, and the common global prediction cap
of 100 plus COCO maxDets [1,10,100]. There is no threshold search on test.

Cloud outputs use a unique directory under `/runs/dqk_legacy_evaluation` on
`rfdetr-runs`. They are also downloaded to a new `Model_Improvement/Results/RESULT_*`
directory. Outputs include the raw compressed COCO predictions, per-class and
aggregate metrics, semantic checks, public config, and checkpoint/adapter
provenance. GPU latency includes native preprocessing, model execution and
postprocessing; disk decode is outside the timer. FLOPs are not measured in this
accuracy investigation.

The evaluator source and dependency versions differ from the team's original
script, so protocol IDs are not interchangeable merely because the protocol
names match. Compare metric outputs on the **same prediction JSON** to establish
COCO evaluator parity; do not falsify or replace source hashes to allow merging.

The older `src/rfdetr/evaluate.py` and confusion-matrix path were removed during
cleanup on 2026-10-21. Use the entrypoint above for legacy evaluation. Existing
checkpoints and historical result files are preserved.
