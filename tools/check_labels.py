from pathlib import Path
import cv2
import random

IMAGE_DIRS = [
    Path("dataset/images/train"),
    Path("dataset/images/val"),
]

OUTPUT_DIR = Path("label_check")
OUTPUT_DIR.mkdir(exist_ok=True)


def draw_labels(image_path, label_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return

    h, w = image.shape[:2]

    if not label_path.exists():
        print(f"No label: {label_path}")
        return

    lines = label_path.read_text().splitlines()

    polygon_count = 0

    for line in lines:
        parts = line.split()

        if len(parts) < 7:
            continue

        class_id = parts[0]
        coords = list(map(float, parts[1:]))

        points = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            points.append([x, y])

        points = cv2.UMat(
            cv2.convexHull(
                cv2.UMat(cv2.UMat.get(cv2.UMat(points))).get()
            )
        )

        polygon_count += 1

    print(f"{image_path.name}: {polygon_count} polygons")


for image_dir in IMAGE_DIRS:
    images = list(image_dir.glob("*.jpg"))

    if not images:
        continue

    samples = random.sample(images, min(3, len(images)))

    for image_path in samples:
        label_path = Path(
            str(image_path)
            .replace("images", "labels")
            .replace(".jpg", ".txt")
        )

        draw_labels(image_path, label_path)

print("Label check finished.")