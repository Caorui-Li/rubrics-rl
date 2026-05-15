"""Extract wethink images from LLaVA-CoT-100k split zip shards.

Concatenates image.zip.part-aa ... part-ap into a single zip and unzips
into ${DATA_ROOT}/wethink_rubrics/images/.  Then writes a "_processed"
copy of the rubrics jsonl, filtering to rows whose image file exists.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))
WETHINK_DIR = DATA_ROOT / "wethink_rubrics"
IMAGES_DIR = WETHINK_DIR / "images"
RAW_JSONL = WETHINK_DIR / "wethink_rubrics_20k.jsonl"
OUT_JSONL = WETHINK_DIR / "wethink_rubrics_20k_processed.jsonl"
MERGED_ZIP = WETHINK_DIR / "image.zip"
SENTINEL = IMAGES_DIR / ".extract_done"


def merge_zip_parts() -> Path:
    parts = sorted(WETHINK_DIR.glob("image.zip.part-*"))
    if not parts:
        sys.exit(f"No image.zip.part-* shards found under {WETHINK_DIR}")
    if MERGED_ZIP.exists():
        print(f"Merged zip already exists: {MERGED_ZIP} ({MERGED_ZIP.stat().st_size} bytes)")
        return MERGED_ZIP
    print(f"Merging {len(parts)} shards into {MERGED_ZIP} ...")
    with open(MERGED_ZIP, "wb") as out:
        for p in parts:
            print(f"  + {p.name}")
            with open(p, "rb") as src:
                shutil.copyfileobj(src, out, length=64 * 1024 * 1024)
    print(f"Merged: {MERGED_ZIP.stat().st_size} bytes")
    return MERGED_ZIP


def unzip(zip_path: Path, dest: Path) -> None:
    if SENTINEL.exists():
        print(f"Images already extracted to {dest}, skipping unzip.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path} -> {dest} ...")
    cmd = ["unzip", "-q", "-o", str(zip_path), "-d", str(dest)]
    rc = subprocess.call(cmd)
    if rc != 0:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
        except Exception as e:
            sys.exit(f"unzip failed (rc={rc}) and zipfile fallback errored: {e}")
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
    WETHINK_DIR.mkdir(parents=True, exist_ok=True)
    zp = merge_zip_parts()
    unzip(zp, IMAGES_DIR)
    write_processed_jsonl()


if __name__ == "__main__":
    main()
