"""Hardware Benchmark Module
Đo đạc và đối sánh hiệu năng phần cứng chuẩn khoa học cho mô hình Object Detection:
- Số lượng tham số (Parameters in Millions)
- Độ phức tạp tính toán GFLOPs (tại input 640x640)
- Độ trễ Inference Latency (ms) trên GPU (Warmup + CUDA Event timing)
- Tốc độ khung hình Throughput FPS (Frames Per Second @ batch=1)
- Kích thước mô hình (MB) từ checkpoint file
"""
import os
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def measure_hardware_benchmark(
    model_obj: Any,
    model_name: str = "Model",
    imgsz: int = 640,
    warmup_runs: int = 20,
    num_runs: int = 100,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Thực hiện đo đạc hiệu năng GPU với quy trình Warmup và CUDA Event timing chuẩn mực."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.startswith("cuda") else "CPU"
    print(f"\n[*] Bắt đầu benchmark phần cứng cho {model_name} trên thiết bị: {device_name} (imgsz={imgsz})...")

    target_device = "cuda" if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

    # 1. Trích xuất PyTorch nn.Module phù hợp cho từng kiến trúc
    torch_model = None
    if isinstance(model_obj, torch.nn.Module) and type(model_obj).__name__ == "DetectionModel":
        torch_model = model_obj
    elif hasattr(model_obj, "model") and type(getattr(model_obj, "model")).__name__ == "DetectionModel":
        torch_model = model_obj.model
    elif hasattr(model_obj, "model") and hasattr(model_obj.model, "model") and isinstance(model_obj.model.model, torch.nn.Module):
        torch_model = model_obj.model.model
    elif hasattr(model_obj, "model") and isinstance(model_obj.model, torch.nn.Module):
        torch_model = model_obj.model
    elif isinstance(model_obj, torch.nn.Module):
        torch_model = model_obj
    elif hasattr(model_obj, "eval"):
        torch_model = model_obj

    # 2. Tính số lượng tham số (Parameters M) và kích thước mô hình
    params_m = 0.0
    model_size_mb = float("nan")

    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            ckpt_bytes = Path(checkpoint_path).stat().st_size
            model_size_mb = round(ckpt_bytes / (1024 * 1024), 2)
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_dict = state.get("model", state) if isinstance(state, dict) else state
            if isinstance(model_dict, dict):
                total_params = sum(p.numel() for p in model_dict.values() if hasattr(p, "numel"))
                if total_params > 0:
                    params_m = round(total_params / 1e6, 2)
        except Exception as exc:
            print(f"[!] Warning loading checkpoint for parameter count: {exc}")

    if params_m == 0.0:
        if torch_model is not None and hasattr(torch_model, "parameters"):
            total_params = sum(p.numel() for p in torch_model.parameters())
            params_m = round(total_params / 1e6, 2)
        elif hasattr(model_obj, "parameters"):
            total_params = sum(p.numel() for p in model_obj.parameters())
            params_m = round(total_params / 1e6, 2)

    # 3. Tính GFLOPs bằng PyTorch FlopCounterMode
    gflops = float("nan")
    candidates_to_profile = []
    curr = model_obj
    while curr is not None and len(candidates_to_profile) < 5:
        if isinstance(curr, torch.nn.Module) and curr not in candidates_to_profile:
            candidates_to_profile.append(curr)
        curr = getattr(curr, "model", None)

    try:
        from torch.utils.flop_counter import FlopCounterMode
        for cand in reversed(candidates_to_profile):
            try:
                cand_dev = next(cand.parameters()).device if list(cand.parameters()) else target_device
                d_inp = torch.randn(1, 3, imgsz, imgsz, device=cand_dev)
                cand.eval()
                raw_flops = 0
                method_used = ""
                try:
                    with torch.no_grad():
                        with FlopCounterMode(display=False) as flop_counter:
                            _ = cand(d_inp)
                        raw_flops = flop_counter.get_total_flops()
                    method_used = "no_grad"
                except Exception:
                    d_inp = torch.randn(1, 3, imgsz, imgsz, device=cand_dev, requires_grad=True)
                    with FlopCounterMode(display=False) as flop_counter:
                        _ = cand(d_inp)
                    raw_flops = flop_counter.get_total_flops()
                    method_used = "requires_grad"

                if raw_flops > 0:
                    gflops = round(float(raw_flops) / 1e9, 2)
                    torch_model = cand
                    print(f"[+] FlopCounterMode đo đạc thành công trên {type(cand).__name__}: {gflops} GFLOPs (method={method_used})")
                    break
            except Exception as e:
                print(f"[!] Warning đo FLOPs trên {type(cand).__name__}: {e}")
    except ImportError:
        print("[!] FlopCounterMode không khả dụng trong phiên bản PyTorch này")

    if gflops != gflops and torch_model is not None:
        try:
            from fvcore.nn import FlopCountAnalysis
            cand_dev = next(torch_model.parameters()).device if hasattr(torch_model, "parameters") else target_device
            fca = FlopCountAnalysis(torch_model, torch.randn(1, 3, imgsz, imgsz, device=cand_dev))
            raw = float(fca.total())
            if raw > 0:
                gflops = round(raw / 1e9, 2)
                print(f"[+] fvcore đo đạc thành công: {gflops} GFLOPs")
        except Exception:
            pass

    # 4. Đo Latency (ms) và FPS bằng CUDA Event timing
    latency_ms = float("nan")
    fps = float("nan")

    try:
        if torch_model is not None:
            torch_model = torch_model.to(target_device)
            torch_model.eval()
            d_inp = torch.randn(1, 3, imgsz, imgsz, device=target_device)

            if target_device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                with torch.no_grad():
                    for _ in range(warmup_runs):
                        _ = torch_model(d_inp)
                    torch.cuda.synchronize()

                    timings = []
                    for _ in range(num_runs):
                        start_event.record()
                        _ = torch_model(d_inp)
                        end_event.record()
                        torch.cuda.synchronize()
                        timings.append(start_event.elapsed_time(end_event))

                    if timings:
                        import numpy as np
                        latency_ms = round(float(np.mean(timings)), 2)
                        fps = round(1000.0 / max(1e-6, latency_ms), 2)
            else:
                with torch.no_grad():
                    for _ in range(5):
                        _ = torch_model(d_inp)
                    t0 = time.perf_counter()
                    for _ in range(20):
                        _ = torch_model(d_inp)
                    t1 = time.perf_counter()
                    latency_ms = round(((t1 - t0) / 20.0) * 1000.0, 2)
                    fps = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0.0
    except Exception as e:
        print(f"[!] Warning benchmarking latency: {e}")

    result = {
        "model_name": model_name,
        "device": device_name,
        "imgsz": imgsz,
        "batch_size": 1,
        "params_m": params_m,
        "gflops": gflops,
        "latency_ms": latency_ms,
        "fps": fps,
        "model_size_mb": model_size_mb,
    }

    print(f"[Benchmark] Params={params_m}M | GFLOPs={gflops} | Latency={latency_ms} ms | FPS={fps} | Size={model_size_mb} MB | Device={device_name}")
    return result