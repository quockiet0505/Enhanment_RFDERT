"""Epoch Logger Module
Ghi nhận và lưu log chi tiết từng epoch theo đúng yêu cầu:
- Epoch
- Train loss
- Class loss
- Box loss
- GIoU
- Learning rate
- Thời gian thực tế chạy (seconds)

Đặc biệt: Hỗ trợ resume an toàn, không bị ghi đè/mất log cũ khi ngắt ngang và tiếp tục từ checkpoint.
"""
import os
import time
import pandas as pd
from typing import Optional, Dict, Any, Callable


class EpochLogger:
    def __init__(
        self,
        log_file_path: str,
        model_name: str,
        commit_fn: Optional[Callable[[], None]] = None,
    ):
        self.log_file_path = log_file_path
        self.model_name = model_name
        self.commit_fn = commit_fn
        self.epoch_start_time = None
        self.logs_dict = {}  # Key by epoch number to prevent duplication
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.load_existing_logs()

    def load_existing_logs(self):
        """Đọc log cũ đã lưu trong storage nếu job bị ngắt ngang và chạy lại."""
        if os.path.exists(self.log_file_path):
            try:
                df = pd.read_csv(self.log_file_path)
                for _, row in df.iterrows():
                    epoch_num = int(row["Epoch"])
                    self.logs_dict[epoch_num] = row.to_dict()
                print(f"[+] Loaded {len(self.logs_dict)} existing epoch logs from: {self.log_file_path}")
            except Exception as e:
                print(f"[!] Warning reading existing log file: {e}")

    def on_epoch_start(self, epoch: int):
        self.epoch_start_time = time.time()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        class_loss: float,
        box_loss: float,
        giou_loss: float,
        learning_rate: float,
        val_precision: Optional[float] = None,
        val_recall: Optional[float] = None,
        val_map50: Optional[float] = None,
        val_map50_95: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        if elapsed_seconds is None and self.epoch_start_time is not None:
            elapsed_seconds = round(time.time() - self.epoch_start_time, 2)

        entry = {
            "Epoch": int(epoch),
            "Train loss": round(float(train_loss), 5),
            "Class loss": round(float(class_loss), 5),
            "Box loss": round(float(box_loss), 5),
            "GIoU": round(float(giou_loss), 5),
            "Learning rate": f"{learning_rate:.6e}",
            "Epoch Time (s)": elapsed_seconds or 0.0,
        }

        if val_precision is not None:
            entry["Val Precision"] = round(float(val_precision), 4)
        if val_recall is not None:
            entry["Val Recall"] = round(float(val_recall), 4)
        if val_map50 is not None:
            entry["val_mAP50"] = round(float(val_map50), 4)
        if val_map50_95 is not None:
            entry["val_mAP50_95"] = round(float(val_map50_95), 4)

        # Lưu hoặc cập nhật theo Epoch (không bị mất log khi resume)
        self.logs_dict[int(epoch)] = entry
        self.save_csv()
        self.print_epoch(entry)

        # Gọi hàm commit storage ngay lập tức sau từng epoch nếu có
        if self.commit_fn is not None:
            try:
                self.commit_fn()
            except Exception as e:
                print(f"[!] Volume commit warning: {e}")

        return entry

    def print_epoch(self, entry: Dict[str, Any]):
        print(
            f"[{self.model_name}] Epoch {entry['Epoch']:02d} | "
            f"Train Loss: {entry['Train loss']:.4f} | "
            f"Class Loss: {entry['Class loss']:.4f} | "
            f"Box Loss: {entry['Box loss']:.4f} | "
            f"GIoU: {entry['GIoU']:.4f} | "
            f"LR: {entry['Learning rate']} | "
            f"Time: {entry['Epoch Time (s)']}s"
        )

    def save_csv(self):
        sorted_entries = [self.logs_dict[k] for k in sorted(self.logs_dict.keys())]
        df = pd.DataFrame(sorted_entries)
        df.to_csv(self.log_file_path, index=False)

    def get_summary_table_md(self) -> str:
        sorted_entries = [self.logs_dict[k] for k in sorted(self.logs_dict.keys())]
        df = pd.DataFrame(sorted_entries)
        return df.to_markdown(index=False)