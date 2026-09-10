# tools/

One-off / dev scripts. Not imported by `app.py`. Run from project root:

```bash
python tools/convert_dataset.py
python tools/evaluate_water.py
```

- `convert_dataset.py` — batch Sentinel-2 -> YOLO-seg (replaces deleted `convert_one.py`)
- `download_dataset.py`, `create_soil_dataset.py` — dataset builders
- `water_mapping.py`, `evaluate_water.py`, `visualize_water.py` — inference / metrics / overlays
- `check_labels.py`, `analyze_polygon_sizes.py` — label sanity checks
