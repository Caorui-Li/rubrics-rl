#!/bin/bash
# Eval Script 3: bs=64, Trained baseline Model, all four tasks

set -e

cd "$(dirname "$0")"

# Configuration
MODEL_PATH="/llm-align/liuchonghan/Qwen25_0128/actor/hf_model"
BATCH_SIZE=64
SYSTEM_PROMPT="eval_prompt.txt"

echo "=========================================="
echo "Trained Model Full Evaluation"
echo "Model: ${MODEL_PATH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "=========================================="

# MathVista
echo ""
echo ">>> Evaluating on MathVista..."
python -m run_qwen25vl_vllm \
    --model_path ${MODEL_PATH} \
    --tasks mathvista \
    --system_prompt ${SYSTEM_PROMPT} \
    --batch_size ${BATCH_SIZE} \
    --log_samples

# MathVision Reason Test
echo ""
echo ">>> Evaluating on MathVision Reason..."
python -m run_qwen25vl_vllm \
    --model_path ${MODEL_PATH} \
    --tasks mathvision_reason_test \
    --system_prompt ${SYSTEM_PROMPT} \
    --batch_size ${BATCH_SIZE} \
    --log_samples

# MathVerse
echo ""
echo ">>> Evaluating on MathVerse..."
python -m run_qwen25vl_vllm \
    --model_path ${MODEL_PATH} \
    --tasks mathverse \
    --system_prompt ${SYSTEM_PROMPT} \
    --batch_size ${BATCH_SIZE} \
    --log_samples

# MMMU (with interleaved visuals)
echo ""
echo ">>> Evaluating on MMMU..."
python -m run_qwen25vl_vllm \
    --model_path ${MODEL_PATH} \
    --tasks mmmu \
    --system_prompt ${SYSTEM_PROMPT} \
    --batch_size ${BATCH_SIZE} \
    --interleave_visuals \
    --log_samples

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
