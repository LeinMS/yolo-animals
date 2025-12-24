import argparse
import random
import re
from pathlib import Path
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def guess_group_key(p: Path) -> str:
    """
    Эвристика группировки, чтобы не смешивать близкие кадры между train/val.
    Подходит под:
      video_1764883932063_frame_19.80.jpg -> group: video_1764883932063
      frame_000000_1.png                  -> group: frame_000000
      img108_3.jpg                        -> group: img108
    """
    name = p.name

    m = re.match(r"(video_\d+)_frame_", name)
    if m:
        return m.group(1)

    m = re.match(r"(frame_\d+)", name)
    if m:
        return m.group(1)

    m = re.match(r"(img\d+)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # общий fallback: отрезаем суффикс после последнего "_"
    stem = p.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem

def read_list(list_path: Path, base_dir: Path) -> list[Path]:
    lines = [ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for ln in lines:
        pp = (base_dir / ln).resolve() if not Path(ln).is_absolute() else Path(ln)
        out.append(pp)
    return out

def write_list(paths: list[Path], out_path: Path, base_dir: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rels = []
    for p in paths:
        try:
            rels.append(p.resolve().relative_to(base_dir.resolve()).as_posix())
        except Exception:
            rels.append(p.resolve().as_posix())
    out_path.write_text("\n".join(rels) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Dataset root directory (where data.yaml lives)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--in-yaml", default="data.yaml")
    ap.add_argument("--in-train-list", default="train.txt")
    ap.add_argument("--out-yaml", default="data_split.yaml")
    ap.add_argument("--out-splits-dir", default="splits")
    args = ap.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    random.seed(args.seed)
    root = Path(args.data_root).resolve()

    in_yaml = root / args.in_yaml
    in_train_list = root / args.in_train_list

    if not in_yaml.exists():
        raise FileNotFoundError(f"Missing {in_yaml}")
    if not in_train_list.exists():
        raise FileNotFoundError(f"Missing {in_train_list}")

    # Читаем исходный список
    img_paths = read_list(in_train_list, base_dir=root)

    # Фильтруем только существующие картинки
    img_paths = [p for p in img_paths if p.exists() and p.suffix.lower() in IMG_EXTS]
    if len(img_paths) < 10:
        raise RuntimeError(f"Too few images found via {in_train_list}: {len(img_paths)}")

    # Группируем
    groups: dict[str, list[Path]] = {}
    for p in img_paths:
        g = guess_group_key(p)
        groups.setdefault(g, []).append(p)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train_g = max(1, int(n_groups * args.train_ratio))
    n_val_g = max(1, int(n_groups * args.val_ratio))
    n_test_g = max(1, n_groups - n_train_g - n_val_g)

    # подстрахуемся, если округления сломали
    while n_train_g + n_val_g + n_test_g > n_groups:
        n_test_g = max(1, n_test_g - 1)
    while n_train_g + n_val_g + n_test_g < n_groups:
        n_train_g += 1

    train_g = set(group_keys[:n_train_g])
    val_g = set(group_keys[n_train_g:n_train_g + n_val_g])
    test_g = set(group_keys[n_train_g + n_val_g:])

    train_imgs = [p for g in train_g for p in groups[g]]
    val_imgs = [p for g in val_g for p in groups[g]]
    test_imgs = [p for g in test_g for p in groups[g]]

    splits_dir = root / args.out_splits_dir
    train_txt = splits_dir / "train.txt"
    val_txt = splits_dir / "val.txt"
    test_txt = splits_dir / "test.txt"

    write_list(sorted(train_imgs), train_txt, base_dir=root)
    write_list(sorted(val_imgs), val_txt, base_dir=root)
    write_list(sorted(test_imgs), test_txt, base_dir=root)

    # Обновляем YAML
    data = yaml.safe_load(in_yaml.read_text(encoding="utf-8"))
    data["path"] = "."  # root
    data["train"] = str(train_txt.relative_to(root)).replace("\\", "/")
    data["val"] = str(val_txt.relative_to(root)).replace("\\", "/")
    data["test"] = str(test_txt.relative_to(root)).replace("\\", "/")

    out_yaml = root / args.out_yaml
    out_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print("Prepared splits:")
    print(f"  train images: {len(train_imgs)}")
    print(f"  val images:   {len(val_imgs)}")
    print(f"  test images:  {len(test_imgs)}")
    print(f"Wrote: {out_yaml}")

if __name__ == "__main__":
    main()
