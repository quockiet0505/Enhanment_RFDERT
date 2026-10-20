import modal
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = modal.App("rfdetr-m-dqk-enhanced-a100")
data_volume = modal.Volume.from_name("faster-rcnn-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("rfdetr-runs", create_if_missing=True)
vol_volume = modal.Volume.from_name("food-detection-training", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "rfdetr[train,loggers]==1.10.1",
        "pycocotools==2.0.8",
        "pandas==2.2.3",
        "pyyaml==6.0.2",
        "pillow==11.0.0",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "tqdm==4.67.1",
        "tensorboard==2.18.0",
        "tabulate>=0.9.0"
    )
    .add_local_dir("./src", remote_path="/root/src")
    .add_local_dir("./configs", remote_path="/root/configs")
)

@app.function(
    image=image, gpu="A100", cpu=16, memory=65536, timeout=60 * 60 * 12,
    volumes={"/data": data_volume, "/runs": runs_volume, "/vol": vol_volume},
)
def train_model_a100(data_dir="/vol/datasets/completed-project-5/coco"):
    from src.rfdetr.train_dqk import train_rfdetr_dqk as train_rfdetr
    
    print("[*] Đang chạy huấn luyện RF-DETR phiên bản cải tiến DQK trên GPU A100...")
    best_ckpt = train_rfdetr(
        dataset_dir=data_dir,
        config_path="/root/configs/rfdetr_dqk.yaml",
        output_dir="/runs/rfdetr_dqk_enhanced/checkpoints",
        log_dir="/runs/rfdetr_dqk_enhanced/logs",
        commit_fn=runs_volume.commit
    )
    runs_volume.commit()
    vol_volume.commit()
    print(f"[+] Hoàn tất huấn luyện. Checkpoint tốt nhất lưu tại: {best_ckpt}")
    return best_ckpt

@app.local_entrypoint()
def main(spawn: bool = False):
    print("=" * 80)
    print("  KHỞI CHẠY HUẤN LUYỆN RF-DETR-DQK-ENHANCED TRÊN MODAL A100 GPU")
    print("=" * 80)
    if spawn:
        train_model_a100.spawn()
    else:
        result = train_model_a100.remote()
        print(f"Kết quả trả về: {result}")