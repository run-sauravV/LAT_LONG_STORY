from ultralytics import YOLO

# Fine-tune nano seg model on the water dataset (CPU).
# Run from project root:  python tools/train_water.py
model = YOLO("yolo11n-seg.pt")
model.train(
    data="configs/data.yaml",
    epochs=30,
    imgsz=512,
    batch=8,
    workers=0,
    device="cpu",
    project="runs",
    name="water_yolov11n",
    exist_ok=True,
)
print("TRAINING DONE")
