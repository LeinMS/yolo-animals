import argparse
import random
import re
from pathlib import Path
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def guess_group_key(name: str) -> str:
    # video_1764883932063_frame_19.80.jpg -> video_1764883932063
    m = re.match(r"(video_\d+)_frame_", name)
    if m:
        return m.group(1)
    # frame_000000_1.png -> frame_000000
    m = re.match(r"(frame_\d+)", name)
    if m:
        return m.group(1)
    # img108_3.jpg -> img108
    m = re.match(r"(img\d+)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    stem = Path(name).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem

def label_path_for_image(root: Path, img: Path) -> Path:
    # expects root/images/<split>/xxx.jpg and root/labels/<split>/xxx.txt
    parts = img.relative_to(root).parts
    # parts: ("images","train","file.jpg") etc
    if len(parts) < 3 or parts[0] != "images":
        # fallback: replace /images/ with /labels/
        s = img.as_posix().replace("/images/", "/labels/")
        return Path(s).with_suffix(".txt")
    split = parts[1]
    return root / "labels" / split / (img.stem + ".txt")

def is_nonempty_label(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        return bool(p.read_text(encoding="utf-8").strip())
    except Exception:
        return False

def write_list(root: Path, out_path: Path, images: list[Path]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rels = [img.relative_to(root).as_posix() for img in images]
    out_path.write_text("\n".join(rels) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root with images/ labels/ data.yaml")
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="splits")
    ap.add_argument("--outyaml", default="data_split.yaml")
    ap.add_argument("--base-split", default="train", help="which images folder to split from (default: train)")
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    root = Path(args.root).resolve()
    random.seed(args.seed)

    img_dir = root / "images" / args.base_split
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)

    images = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if len(images) < 20:
        raise RuntimeError(f"Too few images found in {img_dir}: {len(images)}")

    # group -> list of images
    groups = {}
    for img in images:
        g = guess_group_key(img.name)
        groups.setdefault(g, []).append(img)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    # Prefer to distribute "positive" groups (where at least one non-empty label exists)
    pos_groups = []
    neg_groups = []
    for g in group_keys:
        imgs = groups[g]
        any_pos = any(is_nonempty_label(label_path_for_image(root, im)) for im in imgs)
        (pos_groups if any_pos else neg_groups).append(g)

    # Helper: allocate groups into splits with target proportions
    def allocate(keys):
        n = len(keys)
        n_train = max(1, int(n * args.train))
        n_val = max(1, int(n * args.val))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train > 1:
                n_train -= 1
        return keys[:n_train], keys[n_train:n_train+n_val], keys[n_train+n_val:]

    pos_train, pos_val, pos_test = allocate(pos_groups)
    neg_train, neg_val, neg_test = allocate(neg_groups)

    train_groups = pos_train + neg_train
    val_groups   = pos_val   + neg_val
    test_groups  = pos_test  + neg_test

    # build final lists
    train_imgs = sorted([im for g in train_groups for im in groups[g]])
    val_imgs   = sorted([im for g in val_groups for im in groups[g]])
    test_imgs  = sorted([im for g in test_groups for im in groups[g]])

    outdir = root / args.outdir
    train_txt = outdir / "train.txt"
    val_txt = outdir / "val.txt"
    test_txt = outdir / "test.txt"

    write_list(root, train_txt, train_imgs)
    write_list(root, val_txt, val_imgs)
    write_list(root, test_txt, test_imgs)

    # write new yaml
    data_yaml = root / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data["path"] = "."
    data["train"] = str(train_txt.relative_to(root)).replace("\\", "/")
    data["val"]   = str(val_txt.relative_to(root)).replace("\\", "/")
    data["test"]  = str(test_txt.relative_to(root)).replace("\\", "/")

    out_yaml = root / args.outyaml
    out_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print("Wrote:")
    print(" ", out_yaml)
    print("Splits:")
    print("  train images:", len(train_imgs))
    print("  val images:  ", len(val_imgs))
    print("  test images: ", len(test_imgs))

if __name__ == "__main__":
    main()
