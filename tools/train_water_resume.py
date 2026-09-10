from ultralytics import YOLO

# Continue water training from the previous run (keeps optimizer/scheduler state).
# Run from project root:  python tools/train_water_resume.py
# Target: epoch 80 (50 more). ~1 min/epoch on CPU.
model = YOLO("runs/segment/runs/water_yolov11n/weights/last.pt")
model.train(resume=True, epochs=80)
print("RESUME TRAINING DONE")
