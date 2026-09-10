from ultralytics import YOLO

# Fine-tune the trained water model for 50 MORE epochs (fresh optimizer,
# same data/settings as the first run). best.pt has no optimizer state,
# so resume=True is invalid -- this is the correct way to continue.
# Run from project root:  python tools/train_water_more.py
model = YOLO("runs/segment/runs/water_yolov11n/weights/best.pt")
model.train(
    data="configs/data.yaml",
    epochs=50,
    imgsz=512,
    batch=8,
    workers=0,
    device="cpu",
    project="runs",
    name="water_yolov11n_ft",
    exist_ok=True,
)
print("MORE TRAINING DONE")
