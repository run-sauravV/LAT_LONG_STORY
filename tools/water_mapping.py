from ultralytics import YOLO
import cv2
import numpy as np
import os

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_PATH = "runs/segment/runs/water_yolov8/weights/best.pt"
IMAGE_PATH = "dataset/images/test_water.jpg"

OUTPUT_DIR = "runs/water_mapping"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep detections above this confidence
CONFIDENCE = 0.10


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)

print("Running water segmentation...")

results = model.predict(
    source=IMAGE_PATH,
    conf=CONFIDENCE,
    imgsz=512,
    device="cpu",
    verbose=False
)

result = results[0]


# --------------------------------------------------
# LOAD ORIGINAL IMAGE
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

height, width = image.shape[:2]


# --------------------------------------------------
# COMBINE ALL PREDICTED MASKS
# --------------------------------------------------

combined_mask = np.zeros((height, width), dtype=np.uint8)

if result.masks is not None:

    masks = result.masks.data.cpu().numpy()

    print(f"Detected {len(masks)} water regions.")

    for mask in masks:

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        combined_mask[mask > 0.5] = 255

else:
    print("No water masks detected.")


# --------------------------------------------------
# CLEAN THE MASK
# --------------------------------------------------

kernel = np.ones((5, 5), np.uint8)

combined_mask = cv2.morphologyEx(
    combined_mask,
    cv2.MORPH_OPEN,
    kernel
)

combined_mask = cv2.morphologyEx(
    combined_mask,
    cv2.MORPH_CLOSE,
    kernel
)


# --------------------------------------------------
# CALCULATE WATER COVERAGE
# --------------------------------------------------

water_pixels = np.count_nonzero(combined_mask)
total_pixels = combined_mask.size

water_percentage = (water_pixels / total_pixels) * 100

print(f"Water coverage: {water_percentage:.2f}%")


# --------------------------------------------------
# SAVE BINARY WATER MAP
# --------------------------------------------------

binary_path = os.path.join(
    OUTPUT_DIR,
    "water_binary_map.png"
)

cv2.imwrite(binary_path, combined_mask)


# --------------------------------------------------
# CREATE OVERLAY MAP
# --------------------------------------------------

overlay = image.copy()

# Create a blue water layer
blue_layer = np.zeros_like(image)
blue_layer[:, :, 0] = 255

# Blend blue onto detected water
water_area = combined_mask > 0

overlay[water_area] = cv2.addWeighted(
    image[water_area],
    0.45,
    blue_layer[water_area],
    0.55,
    0
)


# --------------------------------------------------
# SAVE OVERLAY
# --------------------------------------------------

overlay_path = os.path.join(
    OUTPUT_DIR,
    "water_2d_mapping.png"
)

cv2.imwrite(overlay_path, overlay)


# --------------------------------------------------
# SAVE CONTOUR MAP
# --------------------------------------------------

contour_image = image.copy()

contours, _ = cv2.findContours(
    combined_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(
    contour_image,
    contours,
    -1,
    (255, 0, 0),
    2
)

contour_path = os.path.join(
    OUTPUT_DIR,
    "water_feature_boundaries.png"
)

cv2.imwrite(contour_path, contour_image)


# --------------------------------------------------
# FINISHED
# --------------------------------------------------

print()
print("Mapping completed successfully!")
print()
print("Output files:")
print(binary_path)
print(overlay_path)
print(contour_path)