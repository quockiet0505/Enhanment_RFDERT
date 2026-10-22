"""Profile an existing best checkpoint; no training or test inference rerun."""
import json
from pathlib import Path
import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
runs_volume = modal.Volume.from_name("rfdetr-runs", create_if_missing=False)
app = modal.App("dqk-complexity-audit")
profile_image = (modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "transformers==5.1.0",
        "rfdetr[train]==1.10.1", "tensorboard==2.18.0", "pycocotools==2.0.8",
        "pandas==2.2.3", "pyyaml==6.0.2", "pillow==11.0.0",
        "opencv-python-headless>=4.10.0.84", "matplotlib==3.9.2", "seaborn==0.13.2")
    .pip_install("fvcore==0.1.5.post20221221")
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(str(PROJECT_ROOT/"src"), remote_path="/root/src")
    .add_local_dir(str(PROJECT_ROOT/"configs"), remote_path="/root/configs"))


@app.function(image=profile_image, gpu="A100", timeout=1200,
              volumes={"/runs": runs_volume})
def profile(run_name):
    import importlib.metadata
    from src.common.run_versions import checked_run
    from src.rfdetr.lwdetr_dqk import build_enhanced_facade
    from src.testing.profile_complexity import measure
    from src.testing.common_test import sha256_file
    path = checked_run("/runs", run_name)
    checkpoint = path/"checkpoints/best.pth"
    facade, _ = build_enhanced_facade(str(checkpoint), "cuda:0", 640)
    report = measure(facade.model.model)
    report.update(run_name=run_name, checkpoint_sha256=sha256_file(str(checkpoint)),
                  fvcore_version=importlib.metadata.version("fvcore"),
                  profiler_source_sha256=sha256_file("/root/src/testing/profile_complexity.py"))
    # Keep the original accuracy artifacts untouched; attach a separate audit.
    (path/"test/complexity_profile.json").write_text(json.dumps(report, indent=2))
    runs_volume.commit()
    return report


@app.function(image=profile_image, timeout=120, volumes={"/runs": runs_volume})
def attach_saved_report(run_name):
    from src.common.run_versions import checked_run
    from src.testing.complexity_reports import attach
    files = attach(checked_run("/runs", run_name)/"test")
    runs_volume.commit()
    return files


@app.local_entrypoint()
def main(run_name: str = "rfdetr_dqk_enhanced_v1", attach_only: bool = False):
    if not attach_only:
        print(json.dumps(profile.remote(run_name), indent=2))
    files = attach_saved_report.remote(run_name)
    target = PROJECT_ROOT/"Results"/run_name/"test"
    target.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (target/name).write_bytes(data)
    print(f"Updated reports: {target}")
