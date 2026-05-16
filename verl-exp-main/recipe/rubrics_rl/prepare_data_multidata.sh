#!/bin/bash
# Run once on a single node before training.
# Extracts images and converts all three datasets to verl parquet.

set -euo pipefail
set -x

BASEDIR=$(cd "$(dirname "$0")/../.." && pwd)

# DATA_ROOT: set to a large-storage mount if local disk is limited.
# Example: export DATA_ROOT=/mnt/storage/data
export DATA_ROOT=${DATA_ROOT:-"${BASEDIR}/data"}

WETHINK_JSONL="${DATA_ROOT}/wethink_rubrics/wethink_rubrics_20k_processed.jsonl"
WETHINK_IMAGES="${DATA_ROOT}/wethink_rubrics/images"

GEOTHOUGHT_JSONL="${DATA_ROOT}/geothought_rubrics/geothought_rubrics_processed.jsonl"
GEOTHOUGHT_IMAGES="${DATA_ROOT}/geothought_rubrics/images"

VIRL39K_RUBRICS_JSONL="${DATA_ROOT}/virl39k_rubrics/virl39k_rubrics_processed.jsonl"
VIRL39K_RUBRICS_IMAGES="${DATA_ROOT}/virl39k_rubrics/images"

DATASET_DIR="${BASEDIR}/dataset/rubrics_mixed"

# ── Step 1: Extract wethink images ─────────────────────────────────────────
echo "════════════════════════════════════════"
echo "Step 1: Extract wethink images"
echo "════════════════════════════════════════"
python3 "${BASEDIR}/data/extract_wethink_images.py"

# ── Step 2: Extract geothought images ──────────────────────────────────────
echo "════════════════════════════════════════"
echo "Step 2: Extract geothought images"
echo "════════════════════════════════════════"
python3 "${BASEDIR}/data/extract_geothought_images.py"

# ── Step 3: Extract virl39k rubrics images ─────────────────────────────────
echo "════════════════════════════════════════"
echo "Step 3: Extract virl39k rubrics images"
echo "════════════════════════════════════════"
python3 "${BASEDIR}/data/extract_virl39k_rubrics_images.py"

# ── Step 4: Convert to verl parquet ────────────────────────────────────────
echo "════════════════════════════════════════"
echo "Step 4: Convert to verl parquet"
echo "════════════════════════════════════════"
python3 "${BASEDIR}/recipe/rubrics_rl/rubrics_gen/convert_wethink_to_verl.py" \
    --input_jsonl \
        "${WETHINK_JSONL}" \
        "${GEOTHOUGHT_JSONL}" \
        "${VIRL39K_RUBRICS_JSONL}" \
    --output_parquet "${DATASET_DIR}/mixed.parquet" \
    --image_base_paths \
        "${WETHINK_IMAGES}" \
        "${GEOTHOUGHT_IMAGES}" \
        "${VIRL39K_RUBRICS_IMAGES}" \
    --train_val_split 0.9 \
    --seed 42

echo "Data preparation complete. Parquet files written to ${DATASET_DIR}/"
