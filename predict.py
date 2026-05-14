import argparse
from ultralytics import YOLO

# Set up argument parser
parser = argparse.ArgumentParser(description="YOLO Inference Script")
parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
parser.add_argument("--source", type=str, required=True, help="Path to video or image")
parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (default: 0.15)")
parser.add_argument("--save", action="store_true", default=True, help="Save output video")
parser.add_argument("--nosave", action="store_false", dest="save", help="Don't save output video")
parser.add_argument("--show", action="store_true", default=True, help="Show output in window")
parser.add_argument("--noshow", action="store_false", dest="show", help="Don't show output window")
parser.add_argument("--track", action="store_true", default=False, help="Use object tracking instead of detection")
parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cpu', '0', 'cuda')")

args = parser.parse_args()

# Load model
model = YOLO(args.model)

# Run inference
if args.track:
    results = model.track(
        args.source,
        save=args.save,
        show=args.show,
        conf=args.conf,
        device=args.device
    )
else:
    results = model.predict(
        args.source,
        save=args.save,
        show=args.show,
        conf=args.conf,
        device=args.device
    )

print("Inference complete!")
