"""Extract virl39k images from the single images.zip shipped with TIGER-Lab/ViRL39K.

Unzips into ${DATA_ROOT}/virl39k_rubrics/images/, then writes a "_processed"
copy of the rubrics jsonl filtered to rows whose image file exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))
VIRL_DIR = DATA_ROOT / "virl39k_rubrics"
IMAGES_DIR = VIRL_DIR / "images"
RAW_JSONL = VIRL_DIR / "virl39k_rubrics.jsonl"
OUT_JSONL = VIRL_DIR / "virl39k_rubrics_processed.jsonl"
ZIP_PATH = VIRL_DIR / "images.zip"
SENTINEL = IMAGES_DIR / ".extract_done"


def unzip(zip_path: Path, dest: Path) -> None:
    if not zip_path.is_file():
        sys.exit(f"Missing zip: {zip_path}")
    if SENTINEL.exists():
        print(f"Images already extracted to {dest}, skipping unzip.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path} -> {dest} ...")
    rc = subprocess.call(["unzip", "-q", "-o", str(zip_path), "-d", str(dest)])
    if rc != 0:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
        except Exception as e:
            sys.exit(f"unzip failed (rc={rc}) and fallback errored: {e}")
    SENTINEL.write_text("ok\n")
    print("Unzip complete.")


def write_processed_jsonl() -> None:
    if not RAW_JSONL.exists():
        sys.exit(f"Missing rubrics jsonl: {RAW_JSONL}")
    total = kept = missing = 0
    with RAW_JSONL.open() as fin, OUT_JSONL.open("w") as fout:
        for line in fin:
            total += 1
            d = json.loads(line)
            ipath = d.get("image_path", "")
            if not ipath or not (IMAGES_DIR / ipath).is_file():
                missing += 1
                continue
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Processed jsonl: kept={kept} missing={missing} total={total} -> {OUT_JSONL}")


def main() -> None:
    VIRL_DIR.mkdir(parents=True, exist_ok=True)
    unzip(ZIP_PATH, IMAGES_DIR)
    write_processed_jsonl()


if __name__ == "__main__":
    main()
