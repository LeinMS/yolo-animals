import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data_split.yaml")
    ap.add_argument("--model", default="yolo12n.pt", help="e.g. yolo12n.pt or yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze-layers", type=int, default=10)
    ap.add_argument("--freeze-epochs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60, help="Total epochs for full finetune stage")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--device", default=None, help="cuda:0 / 0 / cpu. If None -> auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Stage 1: freeze backbone
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.freeze_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze_layers,
        project=args.project,
        name=f"{args.name}_freeze",
        device=device,
        seed=args.seed,
    )

    best1 = Path(args.project) / f"{args.name}_freeze" / "weights" / "best.pt"
    if not best1.exists():
        # fallback: last.pt
        best1 = Path(args.project) / f"{args.name}_freeze" / "weights" / "last.pt"

    # Stage 2: full finetune
    model = YOLO(str(best1))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=0,
        project=args.project,
        name=f"{args.name}_full",
        device=device,
        seed=args.seed,
    )

    best2 = Path(args.project) / f"{args.name}_full" / "weights" / "best.pt"
    print(f"Done. Best weights: {best2}")

if __name__ == "__main__":
    main()
