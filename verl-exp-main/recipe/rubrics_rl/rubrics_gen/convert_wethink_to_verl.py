# -*- coding: utf-8 -*-
"""Convert one or more rubrics JSONL files to a single merged verl Parquet.

Supported schema (wethink_rubrics / geothought_rubrics and compatible datasets):
  - problem     string, may contain one <image> tag
  - image_path  string, relative to --image_base_path
  - rubrics     {"total_weight": N, "criteria": [{id, criterion, weight}, ...]}
                OR a bare list [{id, criterion, weight}, ...]

Multiple --input_jsonl files are merged before the train/val split so that
training sees a single shuffled dataset.  data_source is auto-derived from
each file's stem (e.g. wethink_rubrics_20k.jsonl → "wethink_rubrics_20k")
unless overridden with --data_source.

Image sources are stored as  split_zip:<hf_repo>:<rel_path>.
Supply --image_base_path pointing to the directory where those zips have been
extracted, so that  <image_base_path>/<image_path>  resolves to a real file.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image_bytes(image_path: str, image_base_path: str | None) -> bytes | None:
    candidates: list[str] = []

    if image_base_path:
        candidates.append(os.path.join(image_base_path, image_path))

    if os.path.isabs(image_path):
        candidates.append(image_path)

    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError as e:
                print(f"Warning: could not read {path}: {e}")
                return None

    print(f"Warning: image not found for path '{image_path}' "
          f"(searched: {candidates})")
    return None


# ---------------------------------------------------------------------------
# Single-sample conversion
# ---------------------------------------------------------------------------

def convert_sample(
    sample: dict[str, Any],
    data_source: str,
    image_base_path: str | None,
) -> dict[str, Any] | None:
    question: str = sample.get("problem", "")
    answer: str = sample.get("answer", "")
    image_path: str = sample.get("image_path", "")
    raw_rubrics = sample.get("rubrics")

    # --- question must have exactly one <image> tag -----------------------
    tag_count = question.count("<image>")
    if tag_count == 0:
        question = "<image>\n" + question
    elif tag_count > 1:
        return None

    # --- extract rubrics criteria list ------------------------------------
    if raw_rubrics is None:
        print(f"Warning: missing rubrics for sample {sample.get('_sample_id')}")
        return None

    if isinstance(raw_rubrics, str):
        try:
            raw_rubrics = json.loads(raw_rubrics)
        except json.JSONDecodeError as e:
            print(f"Warning: rubrics JSON parse error for "
                  f"{sample.get('_sample_id')}: {e}")
            return None

    if isinstance(raw_rubrics, dict) and "criteria" in raw_rubrics:
        criteria: list = raw_rubrics["criteria"]
    elif isinstance(raw_rubrics, list):
        criteria = raw_rubrics
    else:
        print(f"Warning: unrecognised rubrics format for "
              f"{sample.get('_sample_id')}: {type(raw_rubrics)}")
        return None

    if not criteria:
        return None

    # --- load image -------------------------------------------------------
    if not image_path:
        return None

    img_bytes = _load_image_bytes(image_path, image_base_path)
    if img_bytes is None:
        return None

    # --- assemble verl record --------------------------------------------
    extra_info = {
        "prompt": question,
        "rubrics": criteria,
    }

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": extra_info,
        "images": [{"bytes": img_bytes}],
        "rubrics": json.dumps(criteria, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Per-file loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    samples: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: {path}:{lineno} parse error: {e}")
    return samples


def _stem(path: str) -> str:
    """Return filename without extension, used as default data_source."""
    base = os.path.basename(path)
    # strip up to two suffixes (.jsonl, .json, ?download=true, etc.)
    for _ in range(2):
        root, ext = os.path.splitext(base)
        if ext:
            base = root
        else:
            break
    return base


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_to_parquet(
    input_jsonls: list[str],
    output_parquet: str,
    data_source: str | None,
    image_base_paths: list[str | None],
    train_val_split: float,
    seed: int,
) -> None:
    # Pad image_base_paths to match number of inputs (last value is reused)
    n = len(input_jsonls)
    if len(image_base_paths) == 0:
        image_base_paths = [None] * n
    elif len(image_base_paths) == 1:
        image_base_paths = image_base_paths * n
    elif len(image_base_paths) != n:
        raise ValueError(
            f"--image_base_paths has {len(image_base_paths)} values but "
            f"--input_jsonl has {n}. Pass 1 (shared) or one per file."
        )

    all_converted: list[dict] = []

    for path, img_base in zip(input_jsonls, image_base_paths):
        src = data_source if data_source else _stem(path)
        print(f"\n── {path}  (data_source={src}, image_base={img_base})")
        raw = _load_jsonl(path)
        print(f"   Loaded {len(raw)} raw samples")

        converted, skipped = [], 0
        for sample in raw:
            result = convert_sample(sample, src, img_base)
            if result is None:
                skipped += 1
            else:
                converted.append(result)

        print(f"   Converted {len(converted)}, skipped {skipped}")
        all_converted.extend(converted)

    total = len(all_converted)
    print(f"\nTotal converted: {total} samples from {len(input_jsonls)} file(s)")

    if not all_converted:
        print("No samples converted — nothing to write.")
        return

    # preview
    preview = all_converted[0].copy()
    preview["images"] = f"<{len(preview['images'])} image(s)>"
    print("\n" + "=" * 72)
    print("Sample (first record, images elided):")
    print(json.dumps(preview, indent=2, ensure_ascii=False))
    print("=" * 72 + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)

    # shuffle then split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)
    split = int(total * train_val_split)
    train_idx, val_idx = indices[:split], indices[split:]

    base = output_parquet.removesuffix(".parquet")
    train_path = f"{base}_train.parquet"
    val_path = f"{base}_val.parquet"

    full_df = pd.DataFrame(all_converted)
    full_df.to_parquet(output_parquet, index=False)
    print(f"Full  → {output_parquet}  ({total} rows)")

    pd.DataFrame([all_converted[i] for i in train_idx]).to_parquet(train_path, index=False)
    pd.DataFrame([all_converted[i] for i in val_idx]).to_parquet(val_path, index=False)
    print(f"Train → {train_path}  ({len(train_idx)} rows)")
    print(f"Val   → {val_path}  ({len(val_idx)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert one or more rubrics JSONL files to a merged verl Parquet"
    )
    p.add_argument(
        "--input_jsonl", required=True, nargs="+",
        help="One or more input JSONL files (merged before train/val split)",
    )
    p.add_argument(
        "--output_parquet", required=True,
        help="Output base path, e.g. dataset/rubrics_mixed/mixed.parquet\n"
             "Produces: <base>.parquet, <base>_train.parquet, <base>_val.parquet",
    )
    p.add_argument(
        "--data_source", default=None,
        help="Override data_source tag for all files. "
             "If omitted, each file's stem is used as its data_source.",
    )
    p.add_argument(
        "--image_base_paths", nargs="+", default=[],
        metavar="PATH",
        help="Image root directory per input file.\n"
             "  1 value  → shared across all files\n"
             "  N values → one per --input_jsonl file (same order)",
    )
    p.add_argument("--train_val_split", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    convert_to_parquet(
        input_jsonls=args.input_jsonl,
        output_parquet=args.output_parquet,
        data_source=args.data_source,
        image_base_paths=args.image_base_paths,
        train_val_split=args.train_val_split,
        seed=args.seed,
    )
