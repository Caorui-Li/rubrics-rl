#!/bin/bash
# Eval Script 2: bs=64, Qwen/Qwen2.5-VL-7B-Instruct, all four tasks

set -e

cd "$(dirname "$0")"

# Configuration
MODEL_PATH="/llm-align/liuchonghan/Qwen2.5-VL-7B-Instruct"
BATCH_SIZE=64

# Force IPv4 for distributed communication (comprehensive settings)
export CUDA_VISIBLE_DEVICES=1,2,3,4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_ASYNC_ERROR_HANDLING=1

echo "=========================================="
echo "Qwen2.5-VL-7B-Instruct Full Evaluation"
echo "Batch Size: ${BATCH_SIZE}, 4 GPUs (Tensor Parallel)"
echo "=========================================="

# MathVista
echo ""
echo ">>> Evaluating on MathVista..."
python -m run_qwen25vl_vllm \
    --model_path ${MODEL_PATH} \
    --tasks mathvista \
    --batch_size ${BATCH_SIZE} \
    --tensor_parallel_size 4 \
    --log_samples

# MathVision Reason Test
echo ""
echo ">>> Evaluating on MathVision Reason..."
python -m run_qwen25vl_vllm \
    --model_path ${MODEL_PATH} \
    --tasks mathvision_reason_test \
    --batch_size ${BATCH_SIZE} \
    --tensor_parallel_size 4 \
    --log_samples

# MathVerse
echo ""
echo ">>> Evaluating on MathVerse..."
python -m run_qwen25vl_vllm \
    --model_path ${MODEL_PATH} \
    --tasks mathverse \
    --batch_size ${BATCH_SIZE} \
    --tensor_parallel_size 4 \
    --log_samples

# MMMU (with interleaved visuals)
echo ""
echo ">>> Evaluating on MMMU..."
python -m run_qwen25vl_vllm \
    --model_path ${MODEL_PATH} \
    --tasks mmmu \
    --batch_size ${BATCH_SIZE} \
    --tensor_parallel_size 4 \
    --interleave_visuals \
    --log_samples

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
