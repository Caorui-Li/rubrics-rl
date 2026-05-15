"""Extract geothought images from the upstream xinlingdedeng/Geo-Thought parquet.

The rubrics jsonl (GlowLED/geothought-rubrics-17k-test) does NOT ship images;
it references rows of the original Geo-Thought dataset via _sample_id like
'train-4'.  This script downloads the source parquet, extracts each row's
image, and writes a "_processed" jsonl that adds image_path = '<sample_id>.png'.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))
GEO_DIR = DATA_ROOT / "geothought_rubrics"
IMAGES_DIR = GEO_DIR / "images"
SOURCE_DIR = GEO_DIR / "geothought_source"
RAW_JSONL = GEO_DIR / "geothought_rubrics.jsonl"
OUT_JSONL = GEO_DIR / "geothought_rubrics_processed.jsonl"
SOURCE_REPO = "xinlingdedeng/Geo-Thought"
SENTINEL = IMAGES_DIR / ".extract_done"


def download_source_parquet() -> None:
    if SOURCE_DIR.exists() and any(SOURCE_DIR.rglob("*.parquet")):
        print(f"Source parquet already present under {SOURCE_DIR}")
        return
    print(f"Downloading {SOURCE_REPO} -> {SOURCE_DIR} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
        local_dir=str(SOURCE_DIR),
    )


def _image_bytes(value) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict):
        b = value.get("bytes")
        if b:
            return bytes(b)
        path = value.get("path")
        if path and os.path.isfile(path):
            with open(path, "rb") as f:
                return f.read()
        return None
    if hasattr(value, "save"):
        buf = io.BytesIO()
        try:
            value.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None
    return None


def extract_images() -> dict[str, str]:
    """Return mapping sample_id -> image_path (relative to IMAGES_DIR)."""
    import pandas as pd

    if SENTINEL.exists():
        print(f"Images already extracted to {IMAGES_DIR}, skipping image dump.")
        existing = {p.stem: p.name for p in IMAGES_DIR.glob("train-*.png")}
        print(f"Found {len(existing)} existing images.")
        return existing

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    parquets = sorted(SOURCE_DIR.rglob("*.parquet"))
    if not parquets:
        sys.exit(f"No parquet files found under {SOURCE_DIR}")
    print(f"Loading {len(parquets)} parquet(s) ...")
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    print(f"Loaded {len(df)} rows from source parquet.")

    mapping: dict[str, str] = {}
    bad = 0
    for idx, value in enumerate(df["images"]):
        b = _image_bytes(value)
        if b is None:
            bad += 1
            continue
        sid = f"train-{idx}"
        fname = f"{sid}.png"
        (IMAGES_DIR / fname).write_bytes(b)
        mapping[sid] = fname
        if (idx + 1) % 1000 == 0:
            print(f"  wrote {idx + 1} images")
    SENTINEL.write_text("ok\n")
    print(f"Extracted {len(mapping)} images, {bad} unreadable.")
    return mapping


def write_processed_jsonl(mapping: dict[str, str]) -> None:
    if not RAW_JSONL.exists():
        sys.exit(f"Missing rubrics jsonl: {RAW_JSONL}")
    total = kept = missing = 0
    with RAW_JSONL.open() as fin, OUT_JSONL.open("w") as fout:
        for line in fin:
            total += 1
            d = json.loads(line)
            sid = d.get("_sample_id", "")
            fname = mapping.get(sid)
            if not fname or not (IMAGES_DIR / fname).is_file():
                missing += 1
                continue
            d["image_path"] = fname
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Processed jsonl: kept={kept} missing={missing} total={total} -> {OUT_JSONL}")


def main() -> None:
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    download_source_parquet()
    mapping = extract_images()
    write_processed_jsonl(mapping)


if __name__ == "__main__":
    main()
