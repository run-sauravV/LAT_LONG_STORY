from pathlib import Path
import cv2
import statistics


image_dirs = [
    Path("dataset/images/train"),
    Path("dataset/images/val"),
]

areas = []


for image_dir in image_dirs:
    for image_path in image_dir.glob("*.jpg"):

        label_path = Path(
            str(image_path)
            .replace("images", "labels")
            .replace(".jpg", ".txt")
        )

        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        for line in label_path.read_text().splitlines():

            if not line.strip():
                continue

            parts = line.split()

            if len(parts) < 7:
                continue

            coordinates = [
                float(x)
                for x in parts[1:]
            ]

            points = []

            for i in range(0, len(coordinates), 2):
                x = coordinates[i] * width
                y = coordinates[i + 1] * height
                points.append([x, y])

            contour = cv2.UMat(
                cv2.UMat(
                    __import__("numpy").array(points, dtype="float32")
                )
            )

            area = cv2.contourArea(contour)

            areas.append(area)


print()
print("POLYGONS:", len(areas))
print("MIN PIXELS:", round(min(areas), 1))
print("MEDIAN PIXELS:", round(statistics.median(areas), 1))
print("MAX PIXELS:", round(max(areas), 1))
print("<10 px:", sum(x < 10 for x in areas))
print("<50 px:", sum(x < 50 for x in areas))
print("<100 px:", sum(x < 100 for x in areas))
print("<500 px:", sum(x < 500 for x in areas))
print("<1000 px:", sum(x < 1000 for x in areas))
print("<2500 px:", sum(x < 2500 for x in areas))