import requests, time

BASE = "http://127.0.0.1:8000"
IMG = "assets/demo/demo_field.jpg"

with open(IMG, "rb") as f:
    r = requests.post(f"{BASE}/api/v1/upload", files={"photo": ("demo_field.jpg", f, "image/jpeg")}, timeout=30)
print("upload:", r.status_code)
r.raise_for_status()
uid = r.json()["upload_id"]

for i in range(12):
    time.sleep(6)
    rr = requests.get(f"{BASE}/api/v1/results/{uid}", timeout=30).json()
    print(f"poll {i}: {rr['status']}")
    if rr["status"] == "complete":
        cv = rr["cv_output"]
        print("water_bodies:", len(cv["water_bodies"]))
        for w in cv["water_bodies"][:8]:
            print("  conf:", w["confidence"], "area_sqm:", w["area_sqm"])
        print("cv_status:", cv["cv_model_status"])
        break
    if rr["status"] == "failed":
        print("error:", rr.get("error"))
        break
