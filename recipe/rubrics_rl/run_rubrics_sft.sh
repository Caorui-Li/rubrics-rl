#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASEDIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
DATA_DIR=${DATA_DIR:-${BASEDIR}/dataset/ViRL39K}
FULL_DATA=${FULL_DATA:-${DATA_DIR}/rubrics_sft.jsonl}
TRAIN_DATA=${TRAIN_DATA:-${DATA_DIR}/rubrics_sft_train.jsonl}
VAL_DATA=${VAL_DATA:-${DATA_DIR}/rubrics_sft_val.jsonl}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-${BASEDIR}/recipe/rubrics_rl/sft_system_prompt.txt}

MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/models/QwenVL/Qwen3-VL-32B-Thinking}
OUTPUT_DIR=${OUTPUT_DIR:-${BASEDIR}/ms-swift/output/rubrics_sft_qwen3_vl}

TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# Align with Qwen3-VL examples that cap image tokens rather than using MAX_PIXELS.
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}

SPLIT_SEED=${SPLIT_SEED:-42}
TRAIN_RATIO=${TRAIN_RATIO:-0.9}

NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
MAX_LENGTH=${MAX_LENGTH:-6144}
SAVE_STEPS=${SAVE_STEPS:-100}
EVAL_STEPS=${EVAL_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-5}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}

LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-32}
SWIFT_PYTHON=${SWIFT_PYTHON:-python}

python - <<'PY' "${FULL_DATA}" "${TRAIN_DATA}" "${VAL_DATA}" "${TRAIN_RATIO}" "${SPLIT_SEED}"
import json
import os
import random
import sys

full_data, train_data, val_data, train_ratio, split_seed = sys.argv[1:]
train_ratio = float(train_ratio)
split_seed = int(split_seed)

with open(full_data, 'r', encoding='utf-8') as f:
    rows = [line for line in f if line.strip()]

random.Random(split_seed).shuffle(rows)
split_idx = int(len(rows) * train_ratio)
train_rows = rows[:split_idx]
val_rows = rows[split_idx:]

for path, subset in [(train_data, train_rows), (val_data, val_rows)]:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(subset)

print(f'total_rows={len(rows)} train_rows={len(train_rows)} val_rows={len(val_rows)}')
PY

SYSTEM_PROMPT=$(cat "${SYSTEM_PROMPT_FILE}")

cd "${BASEDIR}/ms-swift"

export PYTHONPATH="${BASEDIR}/ms-swift${PYTHONPATH:+:${PYTHONPATH}}"

NPROC_PER_NODE=${NPROC_PER_NODE} \
${SWIFT_PYTHON} -m swift.cli.main sft \
    --model "${MODEL_PATH}" \
    --dataset "${TRAIN_DATA}" \
    --val_dataset "${VAL_DATA}" \
    --split_dataset_ratio 0 \
    --system "${SYSTEM_PROMPT}" \
    --load_from_cache_file true \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate "${LEARNING_RATE}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules all-linear \
    --router_aux_loss_coef 1e-5 \
    --experts_impl grouped_mm \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --gradient_checkpointing true \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps "${LOGGING_STEPS}" \
    --max_length "${MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --deepspeed zero3 \
    --use_liger_kernel true
