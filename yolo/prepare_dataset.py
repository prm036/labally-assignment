#!/usr/bin/env python3
"""
Convert the normalized (x0,y0,x1,y1) annotations into a YOLO-format dataset.

Input : annotations_normalized.py  -> ANN = {filename: [(label, x0, y0, x1, y1), ...]}
        a folder of the source images
Output: <out>/
            images/{train,val}/*.png
            labels/{train,val}/*.txt      # one line per box: "cls cx cy w h" (normalized)
            data.yaml                      # ready for `yolo train data=data.yaml`

Class ids (YOLO is 0-indexed):
    0 = chemical_structure
    1 = reaction_scheme
"""
import argparse, importlib.util, os, random, re, shutil, unicodedata
from PIL import Image

CLASSES = ["chemical_structure", "reaction_scheme"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}


def load_ann(py_path):
    spec = importlib.util.spec_from_file_location("ann_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ANN


def norm_name(s):
    # collapse unicode whitespace (e.g. U+202F narrow no-break space in screenshot names)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s))


def resolve_images(src_dir):
    """Map normalized filename -> actual path on disk."""
    out = {}
    for f in os.listdir(src_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
            out[norm_name(f)] = os.path.join(src_dir, f)
    return out


def to_yolo_line(cls_id, x0, y0, x1, y1):
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    # clamp to [0,1]
    x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in (x0, y0, x1, y1))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with the source images")
    ap.add_argument("--ann", default="annotations_normalized.py", help="path to annotations_normalized.py")
    ap.add_argument("--out", default="dataset", help="output dataset folder")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of images held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ANN = load_ann(args.ann)
    disk = resolve_images(args.images)

    pairs = []  # (annotation_filename, real_image_path)
    missing = []
    for fname in ANN:
        p = disk.get(norm_name(fname))
        if p is None:
            missing.append(fname)
        else:
            pairs.append((fname, p))
    if missing:
        print(f"[warn] {len(missing)} annotated file(s) had no matching image and were skipped:")
        for m in missing:
            print("   ", m)
    if not pairs:
        raise SystemExit("No images matched the annotations — check --images path.")

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, round(len(pairs) * args.val_frac))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    counts = {c: 0 for c in CLASSES}
    for split, items in splits.items():
        for fname, src_path in items:
            stem = os.path.splitext(os.path.basename(src_path))[0]
            # normalize output stem so labels/images line up and have no awkward whitespace
            safe = re.sub(r"\s+", "_", stem)
            ext = os.path.splitext(src_path)[1].lower()
            dst_img = os.path.join(args.out, "images", split, safe + ext)
            dst_lbl = os.path.join(args.out, "labels", split, safe + ".txt")
            shutil.copy(src_path, dst_img)

            lines = []
            for label, x0, y0, x1, y1 in ANN[fname]:
                if label not in CLASS_ID:
                    continue
                line = to_yolo_line(CLASS_ID[label], x0, y0, x1, y1)
                if line:
                    lines.append(line)
                    counts[label] += 1
            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    data_yaml = os.path.join(args.out, "data.yaml")
    abs_out = os.path.abspath(args.out)
    with open(data_yaml, "w") as f:
        f.write(
            f"path: {abs_out}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(CLASSES)}\n"
            f"names: {CLASSES}\n"
        )

    print(f"\nDataset written to {abs_out}")
    print(f"  train images: {len(splits['train'])}")
    print(f"  val images:   {len(splits['val'])}")
    print(f"  boxes -> {counts}")
    print(f"  data.yaml -> {data_yaml}")


if __name__ == "__main__":
    main()
