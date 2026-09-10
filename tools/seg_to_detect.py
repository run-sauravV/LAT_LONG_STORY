"""Convert soil_yolo SEGMENTATION labels to DETECTION (bbox) labels.

app.py's vegetation/bare-ground branch consumes a YOLO *detect* model
(result.boxes), but soil_yolo holds seg polygons. This script reads each
seg polygon, takes its bounding box, and writes a detect dataset:

    soil_yolo_det/
        images/train|val/   (copied from soil_yolo/images/ when present)
        labels/train|val/   (bbox labels, class id unchanged)

Train with:  yolo detect train data=configs/soil_det.yaml model=yolo11n.pt ...
Run from project root:  python tools/seg_to_detect.py
"""
from pathlib import Path
import shutil

SRC = Path("soil_yolo")
DST = Path("soil_yolo_det")


def seg_line_to_box(parts):
    """parts: [class_id, x1, y1, x2, y2, ...] (normalized) -> (cls, xc, yc, w, h)."""
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]
    ys = coords[1::2]
    if not xs:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return cls, (x1 + x2) / 2, (y1 + y2) / 2, w, h


def main():
    converted, skipped, copied = 0, 0, 0
    for split in ("train", "val"):
        src_labels = SRC / "labels" / split
        src_images = SRC / "images" / split
        dst_labels = DST / "labels" / split
        dst_images = DST / "images" / split
        dst_labels.mkdir(parents=True, exist_ok=True)
        dst_images.mkdir(parents=True, exist_ok=True)

        if not src_labels.exists():
            print(f"no labels dir: {src_labels}, skipping split")
            continue

        for label_path in sorted(src_labels.glob("*.txt")):
            boxes = []
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 7:  # need class + at least 3 points
                    continue
                box = seg_line_to_box(parts)
                if box:
                    cls, xc, yc, w, h = box
                    boxes.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            if not boxes:
                skipped += 1
                continue
            (dst_labels / label_path.name).write_text("\n".join(boxes))
            converted += 1

            img = src_images / (label_path.stem + ".jpg")
            if img.exists():
                shutil.copy2(img, dst_images / img.name)
                copied += 1

    print(f"converted: {converted}, skipped (no valid polygon): {skipped}, images copied: {copied}")
    if copied == 0:
        print("WARNING: no images found under soil_yolo/images/ — add them before training.")


if __name__ == "__main__":
    main()
