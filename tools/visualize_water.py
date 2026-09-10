import cv2
import numpy as np

# ============================================================
# FILES
# ============================================================

image_path = "dataset/images/test_water.jpg"
label_path = "dataset/labels/test_water.txt"

output_path = "results/water_overlay.jpg"

# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not open image!")
    exit()

height, width = image.shape[:2]

# Create transparent overlay
overlay = image.copy()

# ============================================================
# READ YOLO WATER POLYGONS
# ============================================================

with open(label_path, "r") as f:
    lines = f.readlines()

polygon_count = 0

for line in lines:

    values = line.strip().split()

    if len(values) < 7:
        continue

    # First value = class ID
    class_id = int(values[0])

    # We use class 0 = water
    if class_id != 0:
        continue

    coordinates = list(map(float, values[1:]))

    points = []

    for i in range(0, len(coordinates), 2):

        x = int(coordinates[i] * width)
        y = int(coordinates[i + 1] * height)

        points.append([x, y])

    points = np.array(points, dtype=np.int32)

    if len(points) >= 3:

        # Fill water region
        cv2.fillPoly(
            overlay,
            [points],
            (255, 0, 0)
        )

        # Draw water boundary
        cv2.polylines(
            image,
            [points],
            True,
            (0, 255, 0),
            2
        )

        polygon_count += 1

# ============================================================
# COMBINE IMAGE + WATER MASK
# ============================================================

result = cv2.addWeighted(
    overlay,
    0.35,
    image,
    0.65,
    0
)

# ============================================================
# SAVE
# ============================================================

cv2.imwrite(output_path, result)

print("----------------------------------------")
print("WATER VISUALIZATION COMPLETED")
print("----------------------------------------")
print("Image size:", width, "x", height)
print("Water polygons:", polygon_count)
print("Overlay saved:", output_path)