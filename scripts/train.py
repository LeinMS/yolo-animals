import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze-layers", type=int, default=10)
    ap.add_argument("--freeze-epochs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    data_path = str(Path(args.data).resolve())

    # Stage 1
    model = YOLO(args.model)
    r1 = model.train(
        data=data_path,
        epochs=args.freeze_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze_layers,
        project=args.project,
        name=f"{args.name}_freeze",
        device=device,
        seed=args.seed,
    )

    save1 = Path(r1.save_dir) if hasattr(r1, "save_dir") else Path(args.project) / f"{args.name}_freeze"
    best1 = save1 / "weights" / "best.pt"
    if not best1.exists():
        best1 = save1 / "weights" / "last.pt"
    if not best1.exists():
        raise FileNotFoundError(f"Cannot find weights after stage1 in {save1}")

    # Stage 2
    model = YOLO(str(best1))
    r2 = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=0,
        project=args.project,
        name=f"{args.name}_full",
        device=device,
        seed=args.seed,
    )

    save2 = Path(r2.save_dir) if hasattr(r2, "save_dir") else Path(args.project) / f"{args.name}_full"
    best2 = save2 / "weights" / "best.pt"
    print(f"Done. Best weights: {best2}")

if __name__ == "__main__":
    main()
