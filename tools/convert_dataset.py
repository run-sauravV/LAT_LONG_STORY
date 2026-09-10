import os
import glob
import cv2
import numpy as np
import rasterio

# ============================================================
# FOLDERS
# ============================================================

RAW_IMAGES = "dataset/raw_images"
RAW_LABELS = "dataset/raw_labels"

TRAIN_IMAGES = "dataset/images/train"
VAL_IMAGES = "dataset/images/val"

TRAIN_LABELS = "dataset/labels/train"
VAL_LABELS = "dataset/labels/val"

for folder in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS
]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# GET ALL IMAGES
# ============================================================

import random

image_files = glob.glob(
    os.path.join(RAW_IMAGES, "*_S2Hand.tif")
)

random.seed(42)
random.shuffle(image_files)

# First 80 = train
# Last 20 = validation
split_index = 80

converted = 0
failed = 0

# ============================================================
# PROCESS IMAGES
# ============================================================

for index, image_path in enumerate(image_files):

    filename = os.path.basename(image_path)
    base_id = filename.replace("_S2Hand.tif", "")

    label_path = os.path.join(
        RAW_LABELS,
        base_id + "_LabelHand.tif"
    )

    if not os.path.exists(label_path):
        print("WARNING: Missing label:", base_id)
        failed += 1
        continue

    # --------------------------------------------------------
    # TRAIN / VAL
    # --------------------------------------------------------

    if index < split_index:
        image_dir = TRAIN_IMAGES
        label_dir = TRAIN_LABELS
    else:
        image_dir = VAL_IMAGES
        label_dir = VAL_LABELS

    output_image = os.path.join(
        image_dir,
        base_id + ".jpg"
    )

    output_label = os.path.join(
        label_dir,
        base_id + ".txt"
    )

    try:

        # ----------------------------------------------------
        # READ SENTINEL-2 IMAGE
        # ----------------------------------------------------

        with rasterio.open(image_path) as src:
            image = src.read()

        # B2 = Blue
        # B3 = Green
        # B4 = Red

        blue = image[1].astype(np.float32)
        green = image[2].astype(np.float32)
        red = image[3].astype(np.float32)

        rgb = np.stack(
            [red, green, blue],
            axis=-1
        )

        # Valid satellite pixels
        valid_pixels = np.any(
            image != 0,
            axis=0
        )

        rgb[~valid_pixels] = 0

        # ----------------------------------------------------
        # CONTRAST STRETCH
        # ----------------------------------------------------

        rgb_uint8 = np.zeros(
            rgb.shape,
            dtype=np.uint8
        )

        for channel in range(3):

            band = rgb[:, :, channel]

            values = band[valid_pixels]

            if len(values) > 0:

                low = np.percentile(values, 2)
                high = np.percentile(values, 98)

                if high > low:

                    stretched = (
                        (band - low)
                        / (high - low)
                        * 255
                    )

                    stretched = np.clip(
                        stretched,
                        0,
                        255
                    )

                    rgb_uint8[:, :, channel] = (
                        stretched.astype(np.uint8)
                    )

        rgb_uint8[~valid_pixels] = 0

        # ----------------------------------------------------
        # READ LABEL
        # ----------------------------------------------------

        with rasterio.open(label_path) as src:
            mask = src.read(1)

        # Water = 1
        water_mask = (
            mask == 1
        ).astype(np.uint8)

        # Remove water from no-data area
        water_mask[~valid_pixels] = 0

        # ----------------------------------------------------
        # SAVE RGB IMAGE
        # ----------------------------------------------------

        cv2.imwrite(
            output_image,
            cv2.cvtColor(
                rgb_uint8,
                cv2.COLOR_RGB2BGR
            )
        )

        # ----------------------------------------------------
        # FIND WATER POLYGONS
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            water_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        height, width = water_mask.shape

        saved_polygons = 0

        # ----------------------------------------------------
        # SAVE YOLO SEGMENTATION LABEL
        # ----------------------------------------------------

        with open(output_label, "w") as f:

            for contour in contours:

                # Remove tiny regions
                if cv2.contourArea(contour) < 100:
                    continue

                epsilon = (
                    0.002
                    * cv2.arcLength(
                        contour,
                        True
                    )
                )

                polygon = cv2.approxPolyDP(
                    contour,
                    epsilon,
                    True
                )

                if len(polygon) < 3:
                    continue

                points = polygon.reshape(
                    -1,
                    2
                )

                normalized = []

                for x, y in points:

                    normalized.append(
                        x / width
                    )

                    normalized.append(
                        y / height
                    )

                f.write(
                    "0 "
                    + " ".join(
                        f"{p:.6f}"
                        for p in normalized
                    )
                    + "\n"
                )

                saved_polygons += 1

        converted += 1

        print(
            f"[{converted}/100] {base_id} "
            f"-> {saved_polygons} water polygons"
        )

    except Exception as e:

        failed += 1
        print(
            "ERROR:",
            base_id,
            "|",
            str(e)
        )

# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("BATCH CONVERSION COMPLETED")
print("========================================")
print("Converted:", converted)
print("Failed:", failed)
print("Training images:", min(80, converted))
print("Validation images:", max(0, converted - 80))
print()
print("Images:")
print("  dataset/images/train")
print("  dataset/images/val")
print()
print("Labels:")
print("  dataset/labels/train")
print("  dataset/labels/val")