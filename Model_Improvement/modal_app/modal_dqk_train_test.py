"""Versioned DQK train -> best reload -> COCO test -> local report download."""
import json
import hashlib
import os
import sys
import traceback
import uuid
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "modal_app":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
app = modal.App("dqk-enhanced-train-test")
data_volume = modal.Volume.from_name("food-detection-training", create_if_missing=False)
runs_volume = modal.Volume.from_name("rfdetr-runs", create_if_missing=False)
claims = modal.Dict.from_name("dqk-versioned-run-claims", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "transformers==5.1.0",
        "rfdetr[train]==1.10.1", "tensorboard==2.18.0", "pycocotools==2.0.8",
        "pandas==2.2.3", "pyyaml==6.0.2", "pillow==11.0.0",
        "opencv-python-headless>=4.10.0.84", "matplotlib==3.9.2", "seaborn==0.13.2")
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(str(PROJECT_ROOT/"src"), remote_path="/root/src")
    .add_local_dir(str(PROJECT_ROOT/"configs"), remote_path="/root/configs")
    .add_local_dir(str(PROJECT_ROOT/"modal_app"), remote_path="/root/modal_app"))


def collect_reports(path, download_best=False):
    allowed = {".json", ".csv", ".md", ".yaml", ".yml", ".png", ".gz", ".log"}
    result = {str(file.relative_to(path)).replace("\\", "/"): file.read_bytes()
              for file in path.rglob("*") if file.is_file() and file.suffix in allowed}
    if download_best and (path/"checkpoints/best.pth").is_file():
        result["checkpoints/best.pth"] = (path/"checkpoints/best.pth").read_bytes()
    return result


@app.function(image=image, gpu="A100", cpu=8, memory=49152, timeout=60*60*24,
              volumes={"/data": data_volume, "/runs": runs_volume})
def run_pipeline(smoke: bool = False, epochs: int = 0, download_best: bool = False,
                 resume_run: str = ""):
    import cv2
    from src.common.run_versions import reserve_run, checked_run
    from src.rfdetr.train_dqk import train_rfdetr_dqk
    from src.testing.common_test import load_config, sha256_file, execute_common_coco_evaluation
    from src.testing.rfdetr_enhanced_adapter import RFDETREnhancedAdapter
    from src.testing.test_legacy_evaluation import run_checks
    runs_volume.reload()
    token = uuid.uuid4().hex
    code_hashes = {str(file.relative_to('/root')): hashlib.sha256(file.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                   for folder in ('src', 'configs') for file in (Path('/root')/folder).rglob('*')
                   if file.is_file() and file.suffix in {'.py', '.yaml', '.json'}}
    if resume_run:
        path = checked_run("/runs", resume_run)
        name = resume_run
        # Do not allow two clients to train/test the same version concurrently.
        lock = "active:"+name
        if not claims.put(lock, token, skip_if_exists=True):
            raise RuntimeError("This run is active or interrupted; inspect it before clearing its active claim")
        resume = path/"checkpoints/last.ckpt"
        if not resume.is_file():
            claims.pop(lock)
            raise FileNotFoundError("Full-state last.ckpt is required for explicit resume")
        old_manifest = json.loads((path/"run_manifest.json").read_text())
        if old_manifest["smoke"] != smoke:
            claims.pop(lock)
            raise ValueError("Cannot change smoke/full mode when resuming")
        if old_manifest.get("code_sha256") != code_hashes:
            claims.pop(lock)
            raise ValueError("Source/config changed; start a new version instead of resuming")
    else:
        stem = "rfdetr_dqk_smoke" if smoke else "rfdetr_dqk_enhanced"
        name, path = reserve_run("/runs", stem, lambda n: claims.put(n, token, skip_if_exists=True))
        lock = "active:"+name
        if not claims.put(lock, token, skip_if_exists=True):
            raise RuntimeError("Run activation collision")
        resume = None
    manifest = {"run_name": name, "smoke": smoke, "status": "running", "phase": "train",
                "dataset_dir": "/data/datasets/completed-project-5/coco",
                "resume": str(resume) if resume else None,
                "architecture": "dqk-four-modules-v1"}
    manifest["code_sha256"] = code_hashes
    manifest_path = path/"run_manifest.json"
    def persist():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        runs_volume.commit()
    print(f"[RUN] {name} -> {path}", flush=True)
    persist()
    success = False
    try:
        run_checks()
        best = train_rfdetr_dqk(manifest["dataset_dir"], "/root/configs/rfdetr_dqk.yaml",
            path/"checkpoints", runs_volume.commit, smoke=smoke,
            resume=str(resume) if resume else None, epochs=epochs or None)
        manifest.update(phase="test", best_checkpoint=best, checkpoint_sha256=sha256_file(best))
        persist()
        cfg = load_config()
        for model in cfg["models"].values():
            model["enabled"] = False
        model_cfg = dict(cfg["models"]["RFDETR_DQK_Enhanced"])
        model_cfg.update(enabled=True, weights=best, checkpoint_sha256=sha256_file(best),
                         display_name=f"RF-DETR DQK enhanced ({name})")
        model_cfg.pop("local_weights", None)
        cfg["models"]["RFDETR_DQK_Enhanced"] = model_cfg
        test_path = path/"test"
        test_path.mkdir(exist_ok=False if not resume_run else True)
        if smoke:
            # Plumbing check on validation only, never label it a test benchmark.
            adapter = RFDETREnhancedAdapter(model_cfg, best, cfg["classes"],
                {label: i+1 for i, label in enumerate(cfg["classes"])}, device="cuda:0")
            valid = Path(manifest["dataset_dir"])/"valid"
            annotation = json.loads((valid/"_annotations.coco.json").read_text())
            filename = annotation["images"][0]["file_name"]
            sample = next((p for p in [valid/filename, valid/"images"/filename] if p.is_file()), None)
            if sample is None:
                raise FileNotFoundError(filename)
            checks = adapter.verify_pipeline(cv2.imread(str(sample)))
            checks.update(smoke_only=True, full_test_not_run=True, load_validation=adapter.load_validation)
            (test_path/"smoke_pipeline.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        else:
            execute_common_coco_evaluation(dataset_dir=manifest["dataset_dir"], output_dir=str(test_path),
                target_models=["RFDETR_DQK_Enhanced"], config_data=cfg, device="cuda:0")
        manifest.update(status="complete", phase="complete")
        success = True
    except Exception:
        manifest.update(status="failed", error=traceback.format_exc())
        print(manifest["error"], flush=True)
    finally:
        persist()
        claims.pop(lock)
    return {"run_name": name, "success": success, "files": collect_reports(path, download_best)}


@app.function(image=image, volumes={"/runs": runs_volume}, timeout=1800)
def fetch_reports(run_name: str, download_best: bool = False):
    from src.common.run_versions import checked_run
    runs_volume.reload()
    path = checked_run("/runs", run_name)
    return {"run_name": run_name, "success": True, "files": collect_reports(path, download_best)}


def save_download(package):
    from src.common.run_versions import checked_run
    root = PROJECT_ROOT/"Results"
    root.mkdir(exist_ok=True)
    destination = root/package["run_name"]
    destination.mkdir(exist_ok=True)
    checked_run(root, package["run_name"])
    for relative, data in package["files"].items():
        file = (destination/relative).resolve()
        if not file.is_relative_to(destination.resolve()):
            raise ValueError("Invalid downloaded report path")
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
    print(f"[DOWNLOADED] {destination}")
    if not package["success"]:
        raise RuntimeError(f"Pipeline failed; inspect {destination/'run_manifest.json'}")


@app.local_entrypoint()
def main(smoke: bool = False, epochs: int = 0, download_best: bool = False,
         fetch_run: str = "", resume_run: str = ""):
    if fetch_run and resume_run:
        raise ValueError("Choose fetch or resume")
    if fetch_run:
        package = fetch_reports.remote(fetch_run, download_best)
    else:
        package = run_pipeline.remote(smoke, epochs, download_best, resume_run)
    save_download(package)
