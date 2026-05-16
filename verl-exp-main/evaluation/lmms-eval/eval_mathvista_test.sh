#!/bin/bash
# Eval Script 1: bs=1, Qwen/Qwen2.5-VL-7B-Instruct, mathvista limit 5

set -e

cd "$(dirname "$0")"

# Configuration
MODEL_PATH="/llm-align/liuchonghan/Qwen2.5-VL-7B-Instruct"

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
echo "MathVista Quick Test (bs=1, 4 GPUs: 1-4)"
echo "=========================================="

python -m run_qwen25vl_vllm \
    --model_path ${MODEL_PATH} \
    --tasks mathvista \
    --limit 5 \
    --batch_size 1 \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.8 \
    --log_samples

echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
