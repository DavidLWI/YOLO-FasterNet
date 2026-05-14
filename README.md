# Sample Prompt
```bash
python benchmark_runner.py --baseline  weights/yolo26n.pt --candidate weights/yolo26n_fasternet.pt --source videos/cars.mp4 --max-frames 500
python predict.py --model weights/yolo26n_fasternet.pt --source videos/cars.mp4
```

# Model Benchmark Suite
Fair, reproducible, side-by-side performance comparisons for Ultralytics-based YOLO and Flow-Count models.

---

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yaml
conda activate benchmark

# 2a. One-command comparison (two YOLO models)
python benchmark_runner.py \
    --baseline  weights/yolo26n.pt \
    --candidate weights/yolo26n_fasternet.pt \
    --source    videos/cars.mp4 \
    --max-frames 500

# 2b. Using a config file (recommended for reproducibility)
python benchmark_runner.py --config benchmark.yaml
```

Results land in `results/<timestamp>/`.  Open **report.html** in any browser for an interactive dashboard.

---

## File Overview

```
benchmark/
├── benchmark_runner.py   — main benchmark engine
├── report_generator.py   — self-contained HTML report builder
├── accuracy_eval.py      — YOLO mAP & Flow-Count MAE/MSE evaluation
├── environment.yaml      — reproducible conda environment
├── benchmark.yaml        — example config (edit this)
└── results/              — auto-created; one sub-folder per run
    └── <run_id>/
        ├── config_used.yaml         — exact config snapshot
        ├── <model>_frames.csv       — per-frame records
        ├── <model>_summary.json     — aggregated stats
        ├── all_summaries.json       — merged comparison
        └── report.html              — interactive dashboard
```

---

## Fairness Guarantees

The benchmark enforces identical conditions across all models:

| Condition | How enforced |
|---|---|
| Same input frames | Single VideoCapture; frames shared in same order |
| Same resolution | `--resolution WxH` resizes before inference |
| Same warmup | Configurable `warmup_frames` runs before recording |
| Same confidence / IoU | Per-model config in YAML |
| GPU memory reset | `torch.cuda.reset_peak_memory_stats()` before each model |
| Timing separation | `result.speed` from Ultralytics gives pre/inf/post independently |

---

## CLI Reference

### benchmark_runner.py

```
--config           Path to YAML config (recommended)
--baseline         Path to baseline model weights
--candidate        Path to candidate model weights
--baseline-label   Display name for baseline (default: filename stem)
--candidate-label  Display name for candidate
--source           Video path or webcam index (default: 0)
--model-type       yolo | flow-count
--device           cuda:0 / cpu (auto-detected if omitted)
--max-frames       Stop after N frames (default: full video)
--warmup           Warm-up frames before recording (default: 10)
--resolution       WxH resize, e.g. 1280x720 (default: native)
--half             Enable FP16 inference (GPU only)
--imgsz            YOLO inference image size (default: 640)
--conf             YOLO confidence threshold (default: 0.25)
--iou              YOLO IoU threshold (default: 0.45)
--output-dir       Where to write results (default: results/)
--no-report        Skip HTML report generation
```

### accuracy_eval.py

```bash
# YOLO mAP (runs Ultralytics val() on your dataset)
python accuracy_eval.py yolo \
    --weights weights/yolov8n.pt \
    --data    data/coco.yaml \
    --out     results/yolo_accuracy.json

# Flow-Count MAE/MSE from benchmark CSV + a ground-truth CSV
python accuracy_eval.py flowcount \
    --predictions results/<run_id>/FlowCount-baseline_frames.csv \
    --gt-counts   data/gt_counts.csv \
    --out         results/flowcount_accuracy.json

# Ground-truth CSV format:
# frame_idx,count
# 0,42
# 1,38
# ...
```

---

## Config File

Edit `benchmark.yaml`:

```yaml
source: "videos/traffic.mp4"
max_frames: 500
resolution: [1280, 720]
warmup_frames: 20

models:
  - path: "weights/yolo26n.pt"
    type: yolo
    label: "YOLOv26n-baseline"
    device: auto
    imgsz: 640
    conf: 0.25
    iou: 0.45
    half: false

  - path: "weights/yolo26n_fasternet.pt"
    type: yolo
    label: "YOLOv26n-FasterNet"
    device: "auto"
    imgsz: 640
    conf: 0.25
    iou: 0.45
    half: false
```

You can add **as many models as you like** — they run sequentially and all appear in the same report.

---

## Adding a Custom Model Type

Subclass `ModelWrapper` in `benchmark_runner.py`:

```python
class MyModelWrapper(ModelWrapper):
    def __init__(self, model_path, device, label="my-model"):
        self.label = label
        self.model = load_my_model(model_path, device)

    def infer(self, frame):
        # Must return: (pre_ms, inf_ms, post_ms, total_ms, n_dets, conf_mean)
        t0 = time.perf_counter()
        result = self.model(frame)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        return 0.0, total_ms, 0.0, total_ms, len(result), 0.0
```

Then register it in `build_model()`:
```python
elif mtype == "my-model":
    return MyModelWrapper(path, device=device, label=label)
```

---

## Output Metrics

| Metric | Description |
|---|---|
| `fps_mean / fps_std` | End-to-end frames per second |
| `inference_ms_mean` | Model-only forward pass (from Ultralytics speed dict) |
| `preprocess_ms_mean` | Frame pre-processing time |
| `postprocess_ms_mean` | NMS / decode time |
| `total_ms_mean` | Wall-clock time for full predict() call |
| `gpu_mem_mb_mean` | Average allocated GPU memory (MB) |
| `gpu_mem_mb_peak` | Peak GPU memory (MB) per run |
| `cpu_percent_mean` | Average CPU utilisation |
| `detections_mean` | Average objects detected (YOLO) or predicted count (Flow-Count) |
| `confidence_mean` | Average detection confidence (YOLO) or predicted count (Flow-Count) |

---

## Reproducibility

Every run saves `config_used.yaml` — the exact configuration — alongside results.
Share the `results/<run_id>/` folder for full reproducibility.
