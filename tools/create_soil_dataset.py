import os
import glob
import cv2
import rasterio
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

SOIL_DATA = "soil_data"

OUTPUT = "soil_yolo"

PATCH_SIZE = 512
STRIDE = 512

# Official dataset mask value:
# 4 = Soil
SOIL_VALUE = 4

# Keep a patch if at least this fraction is soil.
MIN_SOIL_FRACTION = 0.002


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, "labels", split), exist_ok=True)


# ============================================================
# CONVERT BINARY MASK TO YOLO SEGMENTATION
# ============================================================

def mask_to_yolo(mask):
    """
    Convert binary soil mask to YOLO segmentation polygons.
    Returns a list of strings.
    """

    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []

    height, width = mask.shape

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny regions
        if area < 20:
            continue

        polygon = contour.reshape(-1, 2)

        # Need at least 3 points
        if len(polygon) < 3:
            continue

        coords = []

        for x, y in polygon:
            coords.append(x / width)
            coords.append(y / height)

        line = "0 " + " ".join(f"{v:.6f}" for v in coords)

        lines.append(line)

    return lines


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split):

    rgb_files = glob.glob(
        os.path.join(SOIL_DATA, split, "*.tif")
    )

    # Only RGB files, not labelled masks
    rgb_files = [
        f for f in rgb_files
        if "_Labelled" not in os.path.basename(f)
    ]

    print()
    print("=" * 60)
    print("Processing:", split)
    print("Images found:", len(rgb_files))
    print("=" * 60)

    total_patches = 0
    kept_patches = 0

    for rgb_path in sorted(rgb_files):

        filename = os.path.basename(rgb_path)

        mask_path = os.path.join(
            os.path.dirname(rgb_path),
            os.path.splitext(filename)[0] + "_Labelled.tif"
        )

        if not os.path.exists(mask_path):
            print("WARNING: Mask missing:", filename)
            continue

        print()
        print("Image:", filename)

        with rasterio.open(rgb_path) as rgb_src, \
             rasterio.open(mask_path) as mask_src:

            width = rgb_src.width
            height = rgb_src.height

            print("Size:", width, "x", height)

            image_id = os.path.splitext(filename)[0]

            patch_number = 0

            for y in range(0, height - PATCH_SIZE + 1, STRIDE):

                for x in range(0, width - PATCH_SIZE + 1, STRIDE):

                    # Read RGB patch
                    rgb = rgb_src.read(
                        [1, 2, 3],
                        window=rasterio.windows.Window(
                            x, y, PATCH_SIZE, PATCH_SIZE
                        )
                    )

                    # Read corresponding mask patch
                    mask = mask_src.read(
                        1,
                        window=rasterio.windows.Window(
                            x, y, PATCH_SIZE, PATCH_SIZE
                        )
                    )

                    # Convert RGB from (3,H,W) to (H,W,3)
                    rgb = np.transpose(rgb, (1, 2, 0))

                    # Handle unusual values safely
                    rgb = np.nan_to_num(rgb)

                    # Convert to uint8
                    if rgb.dtype != np.uint8:
                        rgb_min = rgb.min()
                        rgb_max = rgb.max()

                        if rgb_max > rgb_min:
                            rgb = (
                                (rgb - rgb_min)
                                / (rgb_max - rgb_min)
                                * 255
                            )

                        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

                    # Soil mask
                    soil_mask = (mask == SOIL_VALUE)

                    soil_fraction = soil_mask.mean()

                    total_patches += 1

                    # Skip patches with almost no soil
                    if soil_fraction < MIN_SOIL_FRACTION:
                        patch_number += 1
                        continue

                    # Convert mask to YOLO polygons
                    yolo_lines = mask_to_yolo(soil_mask)

                    if len(yolo_lines) == 0:
                        patch_number += 1
                        continue

                    output_name = (
                        f"{image_id}_patch_{patch_number:05d}"
                    )

                    image_out = os.path.join(
                        OUTPUT,
                        "images",
                        split,
                        output_name + ".jpg"
                    )

                    label_out = os.path.join(
                        OUTPUT,
                        "labels",
                        split,
                        output_name + ".txt"
                    )

                    # Save image
                    cv2.imwrite(
                        image_out,
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )

                    # Save YOLO label
                    with open(label_out, "w") as f:
                        f.write("\n".join(yolo_lines))

                    kept_patches += 1

                    patch_number += 1

            print("Patches kept from image:", patch_number)


    print()
    print("=" * 60)
    print(split, "COMPLETE")
    print("Total patches checked:", total_patches)
    print("Patches kept:", kept_patches)
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

process_split("train")
process_split("val")

print()
print("=" * 60)
print("SOIL YOLO DATASET CREATION COMPLETE")
print("=" * 60)