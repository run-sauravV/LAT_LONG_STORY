"""Resumable Zenodo downloader with retries. Run from project root:
    python tools/download_soil.py
Progress -> C:/Users/Xiaomi/AppData/Local/Temp/opencode/dl-soil.log
"""
import os
import time
import urllib.request

TMP = r"C:\Users\Xiaomi\AppData\Local\Temp\opencode"
LOG = os.path.join(TMP, "dl-soil.log")

FILES = {
    "train.zip": ("https://zenodo.org/api/records/18757882/files/train.zip/content", 1147687035),
    "val.zip": ("https://zenodo.org/api/records/18757882/files/val.zip/content", 492571852),
}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def download(name, url, total):
    dest = os.path.join(TMP, name)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    while have < total:
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab") as f:
                while True:
                    chunk = r.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
            log(f"{name}: {have/1e6:.0f}/{total/1e6:.0f} MB")
        except Exception as e:
            log(f"{name}: error at {have/1e6:.0f}MB ({e}), retrying in 10s")
            time.sleep(10)
    log(f"{name}: DONE")


if __name__ == "__main__":
    open(LOG, "w").write("")
    for name, (url, total) in FILES.items():
        download(name, url, total)
    log("ALL DONE")
