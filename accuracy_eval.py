"""
accuracy_eval.py
================
Optional accuracy metrics on top of the benchmark records.

YOLO:   Computes mAP@0.5 and mAP@0.5:0.95 against a COCO-format ground-truth.
        Uses Ultralytics built-in val().

Flow-Count: Computes MAE / MSE / RMSE from predicted vs GT counts.

Usage
-----
  # YOLO mAP:
  python accuracy_eval.py yolo \\
      --weights weights/yolov8n.pt \\
      --data coco.yaml \\
      --device auto

  # Flow-Count MAE/MSE from a CSV of (predicted, gt) pairs:
  python accuracy_eval.py flowcount \\
      --predictions results/20240101_120000/FlowCount-baseline_frames.csv \\
      --gt-counts gt_counts.csv

  # Flow-Count: run model on video and compare to GT counts CSV:
  python accuracy_eval.py flowcount-video \\
      --weights weights/flowcount.pt \\
      --source crowd.mp4 \\
      --gt-counts gt_counts.csv
"""

import argparse
import sys
import json
import math
import csv
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# YOLO mAP via Ultralytics val()
# ──────────────────────────────────────────────────────────────────────────────

def eval_yolo_map(weights: str, data_yaml: str, device: str, imgsz: int = 640,
                  half: bool = False) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed.")

    if device == "auto":
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    log.info(f"Running Ultralytics val() on {weights} with data={data_yaml}")
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz, device=device,
                        half=half, verbose=False)

    results = {
        "weights":    weights,
        "data":       data_yaml,
        "device":     device,
        "mAP50":      round(float(metrics.box.map50),  4),
        "mAP50_95":   round(float(metrics.box.map),    4),
        "precision":  round(float(metrics.box.p.mean()), 4),
        "recall":     round(float(metrics.box.r.mean()), 4),
        "f1":         round(float(metrics.box.f1.mean()), 4),
    }

    log.info(json.dumps(results, indent=2))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Flow-Count accuracy: MAE / MSE from predictions vs GT
# ──────────────────────────────────────────────────────────────────────────────

def compute_count_metrics(predicted: list[float], gt: list[float]) -> dict:
    p = np.array(predicted, dtype=float)
    g = np.array(gt, dtype=float)
    if len(p) != len(g):
        raise ValueError(f"Length mismatch: predicted={len(p)}, gt={len(g)}")
    mae  = float(np.abs(p - g).mean())
    mse  = float(((p - g) ** 2).mean())
    rmse = math.sqrt(mse)
    rel  = float((np.abs(p - g) / np.maximum(g, 1)).mean())
    return {
        "n_samples": len(p),
        "MAE":  round(mae, 3),
        "MSE":  round(mse, 3),
        "RMSE": round(rmse, 3),
        "rel_err": round(rel, 4),
        "gt_mean": round(float(g.mean()), 2),
        "pred_mean": round(float(p.mean()), 2),
    }


def eval_flowcount_from_csv(pred_csv: str, gt_csv: str) -> dict:
    """
    pred_csv: the *_frames.csv from benchmark_runner (has 'confidence_mean' or 'detections').
    gt_csv:   two-column CSV: frame_idx, count
    """
    pred_map = {}
    with open(pred_csv) as f:
        for row in csv.DictReader(f):
            fi = int(row["frame_idx"])
            # Use 'confidence_mean' as the predicted count for flow-count models
            pred_map[fi] = float(row.get("confidence_mean", row.get("detections", 0)))

    gt_map = {}
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            fi = int(row["frame_idx"])
            gt_map[fi] = float(row["count"])

    # Match on common frames
    common = sorted(set(pred_map) & set(gt_map))
    if not common:
        raise ValueError("No common frame indices between prediction and GT CSVs.")

    predicted = [pred_map[i] for i in common]
    gt        = [gt_map[i]   for i in common]

    metrics = compute_count_metrics(predicted, gt)
    log.info(json.dumps(metrics, indent=2))
    return metrics


def eval_flowcount_video(weights: str, source: str, gt_csv: str, device: str) -> dict:
    """Run Flow-Count model on video, compare to GT CSV."""
    import torch
    import cv2
    import torchvision.transforms.functional as TF

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load GT
    gt_map = {}
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            gt_map[int(row["frame_idx"])] = float(row["count"])

    # Load model
    try:
        model = torch.jit.load(weights, map_location=device)
    except Exception:
        ckpt = torch.load(weights, map_location=device)
        model = ckpt.get("model") or ckpt
    model.eval()

    cap = cv2.VideoCapture(source)
    predicted, gt_counts = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in gt_map:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = TF.to_tensor(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                density = model(t)
            predicted.append(float(density.sum().item()))
            gt_counts.append(gt_map[idx])
        idx += 1
    cap.release()

    if not predicted:
        raise ValueError("No frames matched GT indices.")

    metrics = compute_count_metrics(predicted, gt_counts)
    log.info(json.dumps(metrics, indent=2))
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Accuracy evaluation for YOLO and Flow-Count")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # YOLO
    yp = sub.add_parser("yolo", help="Compute mAP via Ultralytics val()")
    yp.add_argument("--weights",  required=True)
    yp.add_argument("--data",     required=True, help="COCO-format data YAML")
    yp.add_argument("--device",   default="auto")
    yp.add_argument("--imgsz",    type=int, default=640)
    yp.add_argument("--half",     action="store_true")
    yp.add_argument("--out",      default=None, help="Save JSON results to file")

    # Flow-Count from CSVs
    fp = sub.add_parser("flowcount", help="MAE/MSE from benchmark CSV + GT CSV")
    fp.add_argument("--predictions", required=True, help="*_frames.csv from benchmark_runner")
    fp.add_argument("--gt-counts",   required=True, help="CSV with frame_idx,count columns")
    fp.add_argument("--out",         default=None)

    # Flow-Count from video
    fv = sub.add_parser("flowcount-video", help="MAE/MSE running model on video")
    fv.add_argument("--weights",   required=True)
    fv.add_argument("--source",    required=True)
    fv.add_argument("--gt-counts", required=True)
    fv.add_argument("--device",    default="auto")
    fv.add_argument("--out",       default=None)

    args = parser.parse_args()

    if args.cmd == "yolo":
        results = eval_yolo_map(args.weights, args.data, args.device, args.imgsz, args.half)
    elif args.cmd == "flowcount":
        results = eval_flowcount_from_csv(args.predictions, args.gt_counts)
    elif args.cmd == "flowcount-video":
        results = eval_flowcount_video(args.weights, args.source, args.gt_counts, args.device)

    print(json.dumps(results, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
