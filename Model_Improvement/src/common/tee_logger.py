"""TeeLogger Module
Chịu trách nhiệm ghi lại toàn bộ stdout và stderr trong quá trình đánh giá/huấn luyện
ra file log riêng biệt cho từng mô hình, đảm bảo tính tuần tự, minh bạch, không bị lẫn lộn hay ghi đè.
"""
import sys
import os
import shutil
import time
from datetime import datetime
from typing import Optional, List


class TeeStream:
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, data):
        try:
            self.original_stream.write(data)
            self.original_stream.flush()
        except Exception:
            pass

        try:
            self.log_file.write(data)
            self.log_file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass
        try:
            self.log_file.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.original_stream, "isatty", lambda: False)()

    def fileno(self):
        return getattr(self.original_stream, "fileno", lambda: 1)()


class TeeLogger:
    """Context manager ghi lại toàn bộ log (stdout & stderr) vào file và sao chép đến các đích yêu cầu."""
    def __init__(
        self,
        primary_log_path: str,
        secondary_log_path: Optional[str] = None,
        model_name: str = "MODEL",
        task_name: str = "EVALUATION",
    ):
        self.primary_log_path = primary_log_path
        self.secondary_log_path = secondary_log_path
        self.model_name = model_name
        self.task_name = task_name

        self.file_obj = None
        self.orig_stdout = None
        self.orig_stderr = None
        self.tee_stdout = None
        self.tee_stderr = None
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        os.makedirs(os.path.dirname(self.primary_log_path), exist_ok=True)
        self.file_obj = open(self.primary_log_path, "w", encoding="utf-8", errors="replace")

        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr

        self.tee_stdout = TeeStream(self.orig_stdout, self.file_obj)
        self.tee_stderr = TeeStream(self.orig_stderr, self.file_obj)

        sys.stdout = self.tee_stdout
        sys.stderr = self.tee_stderr

        start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"================================================================================\n"
            f"  LOGGING SESSION: {self.model_name} - {self.task_name}\n"
            f"  Start Time     : {start_dt}\n"
            f"  Primary Log    : {self.primary_log_path}\n"
            f"  Secondary Log  : {self.secondary_log_path or 'None'}\n"
            f"================================================================================\n"
        )
        print(header)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            elapsed = time.time() - (self.start_time or time.time())
            end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = "COMPLETED SUCCESSFULLY" if exc_type is None else f"FAILED WITH ERROR: {exc_val}"
            footer = (
                f"\n================================================================================\n"
                f"  END LOGGING SESSION: {self.model_name} - {self.task_name}\n"
                f"  End Time           : {end_dt}\n"
                f"  Elapsed Time       : {elapsed:.2f}s ({elapsed/60:.2f} mins)\n"
                f"  Status             : {status}\n"
                f"================================================================================\n"
            )
            print(footer)
        finally:
            # Khôi phục stream gốc
            if self.orig_stdout:
                sys.stdout = self.orig_stdout
            if self.orig_stderr:
                sys.stderr = self.orig_stderr

            if self.file_obj:
                try:
                    self.file_obj.flush()
                    self.file_obj.close()
                except Exception:
                    pass

            # Đồng bộ sang file phụ (secondary_log_path) nếu được cấu hình
            if self.secondary_log_path and os.path.exists(self.primary_log_path):
                try:
                    os.makedirs(os.path.dirname(self.secondary_log_path), exist_ok=True)
                    shutil.copy2(self.primary_log_path, self.secondary_log_path)
                except Exception as e:
                    print(f"[!] Warning copying secondary log: {e}", file=self.orig_stderr)