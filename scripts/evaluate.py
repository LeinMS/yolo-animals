import argparse
import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import roc_auc_score

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def resolve_split_items(split_value: str, yaml_dir: Path) -> list[Path]:
    """
    split_value может быть:
      - директорией (images/val)
      - txt файлом со списком путей
    Пути интерпретируем относительно директории yaml.
    """
    p = (yaml_dir / split_value).resolve() if not Path(split_value).is_absolute() else Path(split_value)
    if p.is_file() and p.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out = []
        for ln in lines:
            pp = (yaml_dir / ln).resolve() if not Path(ln).is_absolute() else Path(ln)
            if pp.exists() and pp.suffix.lower() in IMG_EXTS:
                out.append(pp)
        return out
    if p.is_dir():
        files = []
        for ext in IMG_EXTS:
            files.extend(list(p.rglob(f"*{ext}")))
        return sorted(files)
    raise FileNotFoundError(f"Split path not found: {p}")

def infer_label_path(img_path: Path) -> Path:
    # основной путь: заменить /images/ -> /labels/ и .ext -> .txt
    s = img_path.as_posix()
    if "/images/" in s:
        s2 = s.replace("/images/", "/labels/")
        return Path(s2).with_suffix(".txt")
    # fallback: sibling labels dir by convention
    return img_path.with_suffix(".txt")

def read_present_classes(label_path: Path) -> set[int]:
    if not label_path.exists():
        return set()
    present = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cls = int(line.split()[0])
            present.add(cls)
        except Exception:
            continue
    return present

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml of EVAL dataset (unknown dataset)")
    ap.add_argument("--weights", required=True, help="Path to trained weights .pt")
    ap.add_argument("--split", default="auto", help="train/val/test/auto")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001, help="Low conf ok; we take max per class anyway")
    ap.add_argument("--thr", type=float, default=0.25, help="Threshold for extra precision/recall table")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    yaml_dir = data_path.parent

    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    num_classes = len(names)

    # choose split
    split_key = args.split
    if split_key == "auto":
        for candidate in ["test", "val", "train"]:
            if candidate in data:
                split_key = candidate
                break
        if split_key == "auto":
            raise ValueError("No train/val/test in data.yaml")

    split_value = data.get(split_key)
    if split_value is None:
        raise ValueError(f"Split '{split_key}' not found in data.yaml")

    img_paths = resolve_split_items(str(split_value), yaml_dir=yaml_dir)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found for split {split_key}")

    model = YOLO(args.weights)

    y_true = np.zeros((len(img_paths), num_classes), dtype=np.int32)
    y_score = np.zeros((len(img_paths), num_classes), dtype=np.float32)

    for i, img in enumerate(tqdm(img_paths, desc=f"Predict {split_key}")):
        # GT
        label_path = infer_label_path(img)
        present = read_present_classes(label_path)
        for c in present:
            if 0 <= c < num_classes:
                y_true[i, c] = 1

        # Pred
        res = model.predict(
            source=str(img),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False
        )[0]

        # max conf per class
        if res.boxes is not None and len(res.boxes) > 0:
            cls = res.boxes.cls.detach().cpu().numpy().astype(int)
            confs = res.boxes.conf.detach().cpu().numpy().astype(float)
            for c, cf in zip(cls, confs):
                if 0 <= c < num_classes and cf > y_score[i, c]:
                    y_score[i, c] = cf

    # ROC AUC per class
    rows = []
    aucs = []
    for c in range(num_classes):
        yt = y_true[:, c]
        ys = y_score[:, c]
        n_pos = int(yt.sum())
        n_total = int(len(yt))

        if n_pos == 0 or n_pos == n_total:
            auc = None
        else:
            auc = float(roc_auc_score(yt, ys))
            aucs.append(auc)

        rows.append({
            "class_id": c,
            "class_name": names[c] if c < len(names) else str(c),
            "n_images_pos": n_pos,
            "n_images_total": n_total,
            "roc_auc": auc,
            "avg_score_pos": float(ys[yt == 1].mean()) if n_pos > 0 else None,
            "avg_score_neg": float(ys[yt == 0].mean()) if n_pos < n_total else None,
        })

    per_class = pd.DataFrame(rows)

    # threshold table (image-level)
    thr_rows = []
    for c in range(num_classes):
        yt = y_true[:, c].astype(int)
        yp = (y_score[:, c] >= args.thr).astype(int)

        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        thr_rows.append({
            "class_id": c,
            "class_name": names[c] if c < len(names) else str(c),
            "thr": args.thr,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

    thr_df = pd.DataFrame(thr_rows)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    per_class_path = outdir / "per_class_auc.csv"
    thr_path = outdir / "per_class_threshold.csv"
    summary_path = outdir / "summary.json"

    per_class.to_csv(per_class_path, index=False)
    thr_df.to_csv(thr_path, index=False)

    summary = {
        "split": split_key,
        "n_images": len(img_paths),
        "macro_roc_auc": float(np.mean(aucs)) if len(aucs) > 0 else None,
        "n_classes": num_classes,
        "n_classes_with_auc": int(len(aucs)),
        "weights": str(Path(args.weights).resolve()),
        "data": str(data_path),
        "threshold": args.thr,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Console output (кратко и по делу)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop-10 classes by AUC:")
    print(per_class.dropna(subset=["roc_auc"]).sort_values("roc_auc", ascending=False).head(10).to_string(index=False))
    print(f"\nSaved:\n  {per_class_path}\n  {thr_path}\n  {summary_path}")

if __name__ == "__main__":
    main()
