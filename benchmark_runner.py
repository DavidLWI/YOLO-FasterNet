"""
benchmark_runner.py
===================
Fair side-by-side benchmarking for Ultralytics-based YOLO / Flow-Count models.

Usage
-----
  python benchmark_runner.py --config configs/benchmark.yaml
  python benchmark_runner.py --baseline yolov8n.pt --candidate fasternet_yolo.pt --source video.mp4
  python benchmark_runner.py --help

Outputs
-------
  results/<run_id>/
      metrics.json        — full per-frame records
      summary.json        — aggregated statistics
      report.html         — self-contained visual dashboard
      frames/             — optional saved annotated frames
"""

import argparse
import json
import time
import os
import sys
import csv
import logging
import platform
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml
import numpy as np
import cv2
import torch
import psutil

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("ultralytics not installed – YOLO support disabled.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    frame_idx: int
    model_label: str
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    detections: int          # objects detected or density count (Flow-Count)
    confidence_mean: float   # mean confidence (YOLO) or predicted count (Flow-Count)
    gpu_mem_mb: float
    cpu_percent: float


@dataclass
class BenchmarkSummary:
    model_label: str
    model_path: str
    device: str
    total_frames: int
    resolution: str
    warmup_frames: int
    # Timing
    preprocess_ms_mean: float = 0.0
    preprocess_ms_std: float = 0.0
    inference_ms_mean: float = 0.0
    inference_ms_std: float = 0.0
    postprocess_ms_mean: float = 0.0
    postprocess_ms_std: float = 0.0
    total_ms_mean: float = 0.0
    total_ms_std: float = 0.0
    fps_mean: float = 0.0
    fps_std: float = 0.0
    # Memory
    gpu_mem_mb_mean: float = 0.0
    gpu_mem_mb_peak: float = 0.0
    cpu_percent_mean: float = 0.0
    # Accuracy proxy
    detections_mean: float = 0.0
    confidence_mean: float = 0.0
    # Meta
    timestamp: str = ""
    extra: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Model wrappers
# ──────────────────────────────────────────────────────────────────────────────

class ModelWrapper:
    """Abstract wrapper – subclass for each model type."""

    label: str = "model"

    def warmup(self, shape=(640, 640)):
        dummy = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        for _ in range(5):
            self.infer(dummy)

    def infer(self, frame: np.ndarray) -> tuple:
        """Returns (preprocess_ms, inference_ms, postprocess_ms, n_dets, conf_mean)."""
        raise NotImplementedError

    def release(self):
        pass


class UltralyticsYOLOWrapper(ModelWrapper):
    def __init__(self, model_path: str, device: str, imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.45, half: bool = False,
                 label: str = "yolo"):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics package not found.")
        self.label = label
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        self.device = device
        log.info(f"[{label}] Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        if half and device != "cpu":
            self.model.half()

    def infer(self, frame: np.ndarray) -> tuple:
        t0 = time.perf_counter()
        # Ultralytics handles pre/post internally; we time full predict call
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            stream=False,
        )
        t1 = time.perf_counter()

        result = results[0]
        speed = result.speed  # dict: preprocess/inference/postprocess in ms
        pre_ms  = speed.get("preprocess", 0.0)
        inf_ms  = speed.get("inference", 0.0)
        post_ms = speed.get("postprocess", 0.0)

        boxes = result.boxes
        n_dets = len(boxes) if boxes is not None else 0
        conf_mean = float(boxes.conf.mean()) if n_dets > 0 else 0.0

        total_ms = (t1 - t0) * 1000
        return pre_ms, inf_ms, post_ms, total_ms, n_dets, conf_mean

    def warmup(self, shape=(640, 640)):
        log.info(f"[{self.label}] Warming up ({5} frames)…")
        dummy = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        for _ in range(5):
            self.infer(dummy)


class FlowCountWrapper(ModelWrapper):
    """
    Adapter for a Flow-Count / crowd-counting model.
    Expects the model to expose a `.predict(frame) -> density_map` interface.
    Adjust `_load_model` and `infer` to match your actual Flow-Count API.
    """

    def __init__(self, model_path: str, device: str, label: str = "flow-count"):
        self.label = label
        self.device = device
        self.model_path = model_path
        self.model = self._load_model(model_path, device)

    def _load_model(self, path, device):
        """
        Replace this with your actual Flow-Count loader.
        Example stub loads a TorchScript or checkpoint.
        """
        try:
            model = torch.jit.load(path, map_location=device)
            model.eval()
            log.info(f"[{self.label}] TorchScript model loaded from {path}")
            return model
        except Exception:
            pass
        try:
            import importlib.util, sys as _sys
            # Attempt generic PyTorch checkpoint
            ckpt = torch.load(path, map_location=device)
            model = ckpt.get("model") or ckpt
            model.eval()
            log.info(f"[{self.label}] Checkpoint loaded from {path}")
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Flow-Count model from {path}.\n"
                "Override FlowCountWrapper._load_model() for your format."
            ) from exc

    def infer(self, frame: np.ndarray) -> tuple:
        import torchvision.transforms.functional as TF
        t0 = time.perf_counter()

        # ---- preprocess ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(rgb).unsqueeze(0).to(self.device)
        t_pre = time.perf_counter()

        # ---- inference ----
        with torch.no_grad():
            density = self.model(tensor)
        t_inf = time.perf_counter()

        # ---- postprocess ----
        count = float(density.sum().item())
        t_post = time.perf_counter()

        pre_ms  = (t_pre - t0)     * 1000
        inf_ms  = (t_inf - t_pre)  * 1000
        post_ms = (t_post - t_inf) * 1000
        total_ms = (t_post - t0)   * 1000

        return pre_ms, inf_ms, post_ms, total_ms, int(round(count)), count


# ──────────────────────────────────────────────────────────────────────────────
# GPU memory helper
# ──────────────────────────────────────────────────────────────────────────────

def gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def gpu_peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Core benchmark loop
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    model: ModelWrapper,
    source,
    cfg: dict,
) -> tuple[list[FrameRecord], BenchmarkSummary]:
    """
    Run one model over the video source and return per-frame records + summary.
    `source` can be: int (webcam), str path to video, or list of image paths.
    """
    max_frames   = cfg.get("max_frames", None)
    warmup_n     = cfg.get("warmup_frames", 10)
    save_frames  = cfg.get("save_frames", False)
    out_dir      = Path(cfg.get("output_dir", "results")) / cfg.get("run_id", "run")
    resolution   = cfg.get("resolution", None)  # (W, H) or None

    cap = cv2.VideoCapture(source) if not isinstance(source, list) else None
    frame_list = source if isinstance(source, list) else []

    if cap is not None and not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    # warm-up
    log.info(f"[{model.label}] Warming up ({warmup_n} frames)…")
    wh = resolution or (640, 640)
    model.warmup(shape=(wh[1], wh[0]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    records: list[FrameRecord] = []
    frame_idx = 0

    if save_frames:
        frame_out = out_dir / "frames" / model.label
        frame_out.mkdir(parents=True, exist_ok=True)

    while True:
        if max_frames and frame_idx >= max_frames:
            break

        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        elif frame_idx < len(frame_list):
            frame = cv2.imread(frame_list[frame_idx])
            if frame is None:
                frame_idx += 1
                continue
        else:
            break

        if resolution:
            frame = cv2.resize(frame, resolution)

        pre_ms, inf_ms, post_ms, total_ms, n_dets, conf = model.infer(frame)
        mem = gpu_mem_mb()
        cpu = psutil.cpu_percent(interval=None)

        rec = FrameRecord(
            frame_idx=frame_idx,
            model_label=model.label,
            preprocess_ms=round(pre_ms, 3),
            inference_ms=round(inf_ms, 3),
            postprocess_ms=round(post_ms, 3),
            total_ms=round(total_ms, 3),
            detections=n_dets,
            confidence_mean=round(conf, 4),
            gpu_mem_mb=round(mem, 1),
            cpu_percent=round(cpu, 1),
        )
        records.append(rec)

        if frame_idx % 50 == 0:
            log.info(
                f"[{model.label}] frame {frame_idx:5d} | "
                f"inf {inf_ms:6.1f} ms | total {total_ms:6.1f} ms | "
                f"dets {n_dets:3d} | GPU {mem:6.0f} MB"
            )

        frame_idx += 1

    if cap is not None:
        cap.release()

    summary = _summarise(records, model, cfg)
    return records, summary


def _summarise(records: list[FrameRecord], model: ModelWrapper, cfg: dict) -> BenchmarkSummary:
    if not records:
        raise ValueError(f"No frames recorded for model {model.label}.")

    def arr(key):
        return np.array([getattr(r, key) for r in records], dtype=float)

    total_ms_arr = arr("total_ms")
    fps_arr = 1000.0 / np.maximum(total_ms_arr, 1e-6)

    resolution = cfg.get("resolution")
    res_str = f"{resolution[0]}x{resolution[1]}" if resolution else "native"

    device_info = str(cfg.get("device", "auto"))
    if torch.cuda.is_available():
        device_info = torch.cuda.get_device_name(0)

    s = BenchmarkSummary(
        model_label=model.label,
        model_path=str(cfg.get("model_path", "")),
        device=device_info,
        total_frames=len(records),
        resolution=res_str,
        warmup_frames=cfg.get("warmup_frames", 10),
        preprocess_ms_mean=round(float(arr("preprocess_ms").mean()), 2),
        preprocess_ms_std=round(float(arr("preprocess_ms").std()), 2),
        inference_ms_mean=round(float(arr("inference_ms").mean()), 2),
        inference_ms_std=round(float(arr("inference_ms").std()), 2),
        postprocess_ms_mean=round(float(arr("postprocess_ms").mean()), 2),
        postprocess_ms_std=round(float(arr("postprocess_ms").std()), 2),
        total_ms_mean=round(float(total_ms_arr.mean()), 2),
        total_ms_std=round(float(total_ms_arr.std()), 2),
        fps_mean=round(float(fps_arr.mean()), 2),
        fps_std=round(float(fps_arr.std()), 2),
        gpu_mem_mb_mean=round(float(arr("gpu_mem_mb").mean()), 1),
        gpu_mem_mb_peak=round(float(gpu_peak_mb()), 1),
        cpu_percent_mean=round(float(arr("cpu_percent").mean()), 1),
        detections_mean=round(float(arr("detections").mean()), 2),
        confidence_mean=round(float(arr("confidence_mean").mean()), 4),
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(model_cfg: dict) -> ModelWrapper:
    """
    model_cfg keys:
      path   (str)  — path to weights / .pt / .onnx
      type   (str)  — "yolo" | "flow-count"
      label  (str)  — display name
      device (str)  — "cuda:0" / "cpu" / "auto"
      half   (bool) — FP16 (YOLO only)
      imgsz  (int)  — inference image size (YOLO)
      conf   (float)
      iou    (float)
    """
    mtype  = model_cfg.get("type", "yolo").lower()
    path   = model_cfg["path"]
    label  = model_cfg.get("label", Path(path).stem)
    device = model_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if mtype == "yolo":
        return UltralyticsYOLOWrapper(
            model_path=path,
            device=device,
            imgsz=model_cfg.get("imgsz", 640),
            conf=model_cfg.get("conf", 0.25),
            iou=model_cfg.get("iou", 0.45),
            half=model_cfg.get("half", False),
            label=label,
        )
    elif mtype in ("flow-count", "flowcount", "crowd"):
        return FlowCountWrapper(path, device=device, label=label)
    else:
        raise ValueError(f"Unknown model type: {mtype!r}. Choose 'yolo' or 'flow-count'.")


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_results(records: list[FrameRecord], summary: BenchmarkSummary, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-frame CSV
    csv_path = out_dir / f"{summary.model_label}_frames.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        w.writeheader()
        w.writerows(asdict(r) for r in records)
    log.info(f"  Saved frame CSV  → {csv_path}")

    # summary JSON
    summary_path = out_dir / f"{summary.model_label}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    log.info(f"  Saved summary    → {summary_path}")

    return csv_path, summary_path


def merge_summaries(summaries: list[BenchmarkSummary], out_dir: Path) -> Path:
    merged = [asdict(s) for s in summaries]
    p = out_dir / "all_summaries.json"
    with open(p, "w") as f:
        json.dump(merged, f, indent=2)
    log.info(f"  Merged summaries → {p}")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    "models": [],
    "source": "video.mp4",
    "max_frames": None,
    "warmup_frames": 10,
    "resolution": None,        # [W, H] or null for native
    "save_frames": False,
    "output_dir": "results",
    "run_id": None,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    merged = {**DEFAULT_CFG, **cfg}
    if merged["run_id"] is None:
        merged["run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    return merged


def cli_to_config(args) -> dict:
    """Convert argparse namespace to the same dict shape as a YAML config."""
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = {
        **DEFAULT_CFG,
        "source": args.source,
        "max_frames": args.max_frames,
        "warmup_frames": args.warmup,
        "resolution": [int(x) for x in args.resolution.split("x")] if args.resolution else None,
        "output_dir": args.output_dir,
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "models": [
            {
                "path": args.baseline,
                "type": args.model_type,
                "label": args.baseline_label or Path(args.baseline).stem,
                "device": device,
                "half": args.half,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
            },
            {
                "path": args.candidate,
                "type": args.model_type,
                "label": args.candidate_label or Path(args.candidate).stem,
                "device": device,
                "half": args.half,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
            },
        ],
    }
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fair side-by-side model benchmarking tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a YAML config (recommended):
  python benchmark_runner.py --config configs/benchmark.yaml

  # Quick CLI comparison of two YOLO models:
  python benchmark_runner.py \\
      --baseline yolov8n.pt \\
      --candidate fasternet_yolo.pt \\
      --source traffic.mp4 \\
      --max-frames 500

  # Crowd-counting comparison:
  python benchmark_runner.py \\
      --baseline baseline_crowd.pt \\
      --candidate fasternet_crowd.pt \\
      --source crowd.mp4 \\
      --model-type flow-count \\
      --resolution 1280x720
        """,
    )

    parser.add_argument("--config", help="Path to YAML config file")

    # Quick CLI args (used when --config is not given)
    parser.add_argument("--baseline",        help="Path to baseline model weights")
    parser.add_argument("--candidate",       help="Path to candidate model weights")
    parser.add_argument("--baseline-label",  default=None)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--source",          default="0", help="Video path or webcam index")
    parser.add_argument("--model-type",      default="yolo", choices=["yolo", "flow-count"])
    parser.add_argument("--device",          default=None, help="cuda:0 / cpu (auto-detected if omitted)")
    parser.add_argument("--max-frames",      type=int, default=None)
    parser.add_argument("--warmup",          type=int, default=10)
    parser.add_argument("--resolution",      default=None, help="WxH e.g. 1280x720")
    parser.add_argument("--half",            action="store_true", help="FP16 inference")
    parser.add_argument("--imgsz",           type=int, default=640)
    parser.add_argument("--conf",            type=float, default=0.25)
    parser.add_argument("--iou",             type=float, default=0.45)
    parser.add_argument("--output-dir",      default="results")
    parser.add_argument("--no-report",       action="store_true", help="Skip HTML report generation")

    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = load_config(args.config)
        log.info(f"Config loaded from {args.config}")
    elif args.baseline and args.candidate:
        cfg = cli_to_config(args)
    else:
        parser.print_help()
        sys.exit(1)

    run_id  = cfg["run_id"]
    out_dir = Path(cfg["output_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {out_dir}")

    # Save resolved config for reproducibility
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f)

    resolution = cfg.get("resolution")
    if resolution:
        resolution = tuple(resolution)

    all_records: dict[str, list[FrameRecord]] = {}
    all_summaries: list[BenchmarkSummary] = []

    source = cfg["source"]
    # Try int (webcam)
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    for model_cfg in cfg["models"]:
        model_cfg.setdefault("device", "auto")
        label = model_cfg.get("label", Path(model_cfg["path"]).stem)
        log.info(f"\n{'='*60}")
        log.info(f"  Benchmarking: {label}")
        log.info(f"{'='*60}")

        per_frame_cfg = {
            **cfg,
            "model_path": model_cfg["path"],
        }

        try:
            model = build_model(model_cfg)
            records, summary = run_benchmark(model, source, per_frame_cfg)
            model.release()

            all_records[label] = records
            all_summaries.append(summary)
            save_results(records, summary, out_dir)

        except Exception as exc:
            log.error(f"Model {label} failed: {exc}", exc_info=True)
            continue

    if not all_summaries:
        log.error("No models completed successfully. Exiting.")
        sys.exit(1)

    merge_summaries(all_summaries, out_dir)

    # Generate HTML report
    if not (hasattr(args, "no_report") and args.no_report):
        try:
            from report_generator import generate_report
            report_path = generate_report(all_summaries, all_records, out_dir)
            log.info(f"\n  HTML report      → {report_path}")
        except Exception as exc:
            log.warning(f"Could not generate HTML report: {exc}")

    log.info(f"\n{'='*60}")
    log.info("  BENCHMARK COMPLETE")
    log.info(f"  Results: {out_dir}")
    log.info(f"{'='*60}\n")

    # Print quick comparison table to console
    _print_console_table(all_summaries)


def _print_console_table(summaries: list[BenchmarkSummary]):
    cols = ["model_label", "fps_mean", "inference_ms_mean", "total_ms_mean",
            "gpu_mem_mb_mean", "gpu_mem_mb_peak", "detections_mean"]
    headers = ["Model", "FPS", "Inf(ms)", "Total(ms)", "GPU(MB)", "PeakGPU(MB)", "Dets/Count"]
    widths = [20, 8, 10, 10, 10, 13, 12]

    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep  = "-+-".join("-" * w for w in widths)
    print(f"\n{line}")
    print(sep)
    for s in summaries:
        vals = [
            getattr(s, c) for c in cols
        ]
        row = " | ".join(str(v).ljust(w) for v, w in zip(vals, widths))
        print(row)
    print()


if __name__ == "__main__":
    main()
