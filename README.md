# SIH26015 — Watershed Monitoring System

Field photo analysis: CV (water / vegetation segmentation) + GIS (DEM, watershed, NDVI, weather).

## Run it

```powershell
# terminal 1 — backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
# terminal 2 — frontend
cd frontend
npm.cmd run dev
```

Open **http://localhost:3000**. Upload a geo-tagged photo from `assets/demo/`.

## Layout → branches (4 branches + main)

| Branch       | Owns                                    | Members work here, merge into `main` |
|--------------|-----------------------------------------|--------------------------------------|
| `main`       | everything (integration, releases)      | —                                    |
| `backend`    | `app.py`, `requirements.txt`            | API, CV/GIS pipeline wiring          |
| `frontend`   | `frontend/`, `legacy/`                  | Next.js UI + legacy page             |
| `ai-ml`      | `tools/`, `configs/`, `water_best.pt`   | datasets, training, model weights    |
| `gis-engine` | `gis_engine/`                           | DEM / hydrology / watershed code     |

Shared: `assets/demo/` (test photos), `results/` (samples), `README.md`, `LICENSE`, `.env.example`.

```powershell
# first push (run once, from repo root)
git init -b main
git add .
git add -f water_best.pt
git commit -m "SIH watershed app"
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
git checkout -b backend; git push -u origin backend
git checkout -b frontend; git push -u origin frontend
git checkout -b ai-ml; git push -u origin ai-ml
git checkout -b gis-engine; git push -u origin gis-engine
git checkout main
```

Never commit `.env` (keys stay local; deploy via host secrets). Retrain notes live in `tools/README.md`.
