#!/bin/bash
# Prepare wethink_rubrics + geothought_rubrics datasets for RL training.
# Converts JSONL files to a merged verl Parquet with train/val split.
#
# Usage:
#   bash recipe/rubrics_rl/prepare_wethink_data.sh
#
# Key env vars (override as needed):
#   WETHINK_IMAGE_BASE_PATH     image root for wethink_rubrics
#   GEOTHOUGHT_IMAGE_BASE_PATH  image root for geothought_rubrics
#   OUTPUT_DIR                  where to write the parquet files
#   TRAIN_VAL_SPLIT             fraction used for training (default 0.9)
#   SEED                        random seed for shuffle (default 42)

set -euo pipefail

BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
WETHINK_IMAGE_BASE_PATH=${WETHINK_IMAGE_BASE_PATH:-"${BASEDIR}/datasets/llava-cot-100k"}
GEOTHOUGHT_IMAGE_BASE_PATH=${GEOTHOUGHT_IMAGE_BASE_PATH:-"${BASEDIR}/datasets/geothought-images"}

OUTPUT_DIR=${OUTPUT_DIR:-"${BASEDIR}/dataset/wethink_rubrics"}
TRAIN_VAL_SPLIT=${TRAIN_VAL_SPLIT:-0.9}
SEED=${SEED:-42}

WETHINK_JSONL="${BASEDIR}/data/wethink_rubrics/wethink_rubrics_20k.jsonl"
GEOTHOUGHT_JSONL="${BASEDIR}/data/geothought_rubrics/geothought_rubrics.jsonl"

CONVERT_SCRIPT="${BASEDIR}/recipe/rubrics_rl/rubrics_gen/convert_wethink_to_verl.py"
OUTPUT_PARQUET="${OUTPUT_DIR}/mixed.parquet"

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
echo "=== prepare_wethink_data.sh ==="
echo "BASEDIR:                    ${BASEDIR}"
echo "WETHINK_IMAGE_BASE_PATH:    ${WETHINK_IMAGE_BASE_PATH}"
echo "GEOTHOUGHT_IMAGE_BASE_PATH: ${GEOTHOUGHT_IMAGE_BASE_PATH}"
echo "OUTPUT_DIR:                 ${OUTPUT_DIR}"
echo ""

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    echo "ERROR: conversion script not found: ${CONVERT_SCRIPT}"
    exit 1
fi

# Collect available JSONL files and their corresponding image base paths
INPUT_JSONLS=()
IMAGE_BASE_PATHS=()

# --- wethink ---
if [[ -f "${WETHINK_JSONL}" ]]; then
    WETHINK_FILE="${WETHINK_JSONL}"
else
    WETHINK_FILE=$(ls "${BASEDIR}/data/wethink_rubrics/"*.jsonl* 2>/dev/null | head -1 || true)
fi

if [[ -n "${WETHINK_FILE:-}" ]]; then
    INPUT_JSONLS+=("${WETHINK_FILE}")
    IMAGE_BASE_PATHS+=("${WETHINK_IMAGE_BASE_PATH}")
    echo "Found wethink:     ${WETHINK_FILE}"
    echo "  images:          ${WETHINK_IMAGE_BASE_PATH}"
else
    echo "WARNING: wethink JSONL not found, skipping."
fi

# --- geothought ---
if [[ -f "${GEOTHOUGHT_JSONL}" ]]; then
    GEOTHOUGHT_FILE="${GEOTHOUGHT_JSONL}"
else
    GEOTHOUGHT_FILE=$(ls "${BASEDIR}/data/geothought_rubrics/"*.jsonl* 2>/dev/null | head -1 || true)
fi

if [[ -n "${GEOTHOUGHT_FILE:-}" ]]; then
    INPUT_JSONLS+=("${GEOTHOUGHT_FILE}")
    IMAGE_BASE_PATHS+=("${GEOTHOUGHT_IMAGE_BASE_PATH}")
    echo "Found geothought:  ${GEOTHOUGHT_FILE}"
    echo "  images:          ${GEOTHOUGHT_IMAGE_BASE_PATH}"
else
    echo "WARNING: geothought JSONL not found, skipping."
fi

if [[ ${#INPUT_JSONLS[@]} -eq 0 ]]; then
    echo "ERROR: no input JSONL files found."
    exit 1
fi

# Warn if any image directory is missing
for IMG_DIR in "${IMAGE_BASE_PATHS[@]}"; do
    if [[ ! -d "${IMG_DIR}" ]]; then
        echo "WARNING: image directory not found: ${IMG_DIR}"
        echo "         Images will fail to load. Set the corresponding env var before running."
        read -r -p "Continue anyway? [y/N] " CONFIRM
        if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
            exit 1
        fi
        break
    fi
done

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Run conversion
# ---------------------------------------------------------------------------
echo ""
echo "=== Converting ${#INPUT_JSONLS[@]} file(s) → ${OUTPUT_PARQUET} ==="

python3 "${CONVERT_SCRIPT}" \
    --input_jsonl "${INPUT_JSONLS[@]}" \
    --image_base_paths "${IMAGE_BASE_PATHS[@]}" \
    --output_parquet "${OUTPUT_PARQUET}" \
    --train_val_split "${TRAIN_VAL_SPLIT}" \
    --seed "${SEED}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Done ==="
echo "Output files:"
for f in "${OUTPUT_DIR}/mixed.parquet" \
          "${OUTPUT_DIR}/mixed_train.parquet" \
          "${OUTPUT_DIR}/mixed_val.parquet"; do
    if [[ -f "${f}" ]]; then
        SIZE=$(du -sh "${f}" | cut -f1)
        echo "  ${SIZE}  ${f}"
    fi
done

echo ""
echo "To use in training, set:"
echo "  DATASET_TRAIN=${OUTPUT_DIR}/mixed_train.parquet"
echo "  DATASET_VAL=${OUTPUT_DIR}/mixed_val.parquet"
