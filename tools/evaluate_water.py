from ultralytics import YOLO
import cv2
import numpy as np
import os

MODEL_PATH = "runs/segment/runs/water_yolov8/weights/best.pt"
IMAGE_PATH = "dataset/images/test_water.jpg"
LABEL_PATH = "dataset/labels/test_water.txt"

print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)

print("Running prediction...")

results = model.predict(
    source=IMAGE_PATH,
    conf=0.10,
    imgsz=512,
    device="cpu",
    verbose=False
)

result = results[0]

# --------------------------------------------------
# Create predicted mask
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

height, width = image.shape[:2]

pred_mask = np.zeros((height, width), dtype=np.uint8)

if result.masks is not None:

    masks = result.masks.data.cpu().numpy()

    print(f"Predicted water regions: {len(masks)}")

    for mask in masks:

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        pred_mask[mask > 0.5] = 1

else:
    print("No predicted masks found.")


# --------------------------------------------------
# Create ground-truth mask from YOLO polygons
# --------------------------------------------------

gt_mask = np.zeros((height, width), dtype=np.uint8)

with open(LABEL_PATH, "r") as file:

    for line in file:

        values = line.strip().split()

        if len(values) < 7:
            continue

        # Skip class ID
        coordinates = list(map(float, values[1:]))

        points = []

        for i in range(0, len(coordinates), 2):

            x = int(coordinates[i] * width)
            y = int(coordinates[i + 1] * height)

            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        cv2.fillPoly(gt_mask, [points], 1)


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

gt = gt_mask.astype(bool)
pred = pred_mask.astype(bool)

intersection = np.logical_and(gt, pred).sum()
union = np.logical_or(gt, pred).sum()

gt_pixels = gt.sum()
pred_pixels = pred.sum()

true_positive = intersection
false_positive = np.logical_and(~gt, pred).sum()
false_negative = np.logical_and(gt, ~pred).sum()

iou = intersection / union if union > 0 else 0

dice = (
    2 * intersection /
    (gt_pixels + pred_pixels)
    if (gt_pixels + pred_pixels) > 0
    else 0
)

precision = (
    true_positive / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

gt_percentage = (gt_pixels / (height * width)) * 100
pred_percentage = (pred_pixels / (height * width)) * 100


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("========================================")
print("WATER SEGMENTATION EVALUATION")
print("========================================")

print(f"Ground-truth water coverage : {gt_percentage:.2f}%")
print(f"Predicted water coverage    : {pred_percentage:.2f}%")
print()

print(f"IoU                         : {iou:.4f}")
print(f"Dice Score                  : {dice:.4f}")
print(f"Precision                   : {precision:.4f}")
print(f"Recall                      : {recall:.4f}")

print("========================================")