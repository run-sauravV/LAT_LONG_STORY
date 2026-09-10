import os
import subprocess

# ============================================================
# SETTINGS
# ============================================================

IMAGE_BUCKET = "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/"
LABEL_BUCKET = "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/"

IMAGE_DIR = "dataset/raw_images"
LABEL_DIR = "dataset/raw_labels"

NUM_PAIRS = 100

# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# ============================================================
# GET IMAGE LIST
# ============================================================

print("Getting Sentinel-2 image list...")

result = subprocess.run(
    ["gsutil", "ls", IMAGE_BUCKET],
    capture_output=True,
    text=True,
    check=True
)

image_files = [
    line.strip()
    for line in result.stdout.splitlines()
    if line.strip().endswith(".tif")
]

print("Available images:", len(image_files))

# ============================================================
# DOWNLOAD MATCHING IMAGE + LABEL PAIRS
# ============================================================

downloaded = 0

for image_url in image_files:

    if downloaded >= NUM_PAIRS:
        break

    filename = os.path.basename(image_url)

    # Example:
    # Bolivia_103757_S2Hand.tif
    #
    # Matching label:
    # Bolivia_103757_LabelHand.tif

    base_id = filename.replace("_S2Hand.tif", "")

    label_filename = base_id + "_LabelHand.tif"
    label_url = LABEL_BUCKET + label_filename

    image_output = os.path.join(IMAGE_DIR, filename)
    label_output = os.path.join(LABEL_DIR, label_filename)

    # Skip if both already exist
    if os.path.exists(image_output) and os.path.exists(label_output):
        downloaded += 1
        print(
            f"[{downloaded}/{NUM_PAIRS}] Already exists: {base_id}"
        )
        continue

    print(
        f"\n[{downloaded + 1}/{NUM_PAIRS}] Downloading: {base_id}"
    )

    try:

        subprocess.run(
            ["gsutil", "cp", image_url, IMAGE_DIR],
            check=True
        )

        subprocess.run(
            ["gsutil", "cp", label_url, LABEL_DIR],
            check=True
        )

        downloaded += 1

        print("Pair downloaded successfully.")

    except subprocess.CalledProcessError:

        print("WARNING: Could not download this pair.")
        print("Skipping:", base_id)

print("\n========================================")
print("DOWNLOAD COMPLETED")
print("========================================")
print("Image/label pairs:", downloaded)
print("Images folder:", IMAGE_DIR)
print("Labels folder:", LABEL_DIR)