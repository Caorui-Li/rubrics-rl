#!/bin/bash
# Run Qwen2.5-VL 7B evaluation on Math benchmarks
# Tasks: MathVista, MathVision, MathVerse, MMMU

set -e

# Configuration
export HF_HOME="./.cache/huggingface"

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

MODEL_NAME="/llm-align/liuchonghan/Qwen2.5-VL-7B-Instruct"
MAX_PIXELS=12845056
BATCH_SIZE=1
# For interleaved tasks like MMMU, use interleave_visuals=True
# For other tasks, use interleave_visuals=False

echo "=========================================="
echo "Qwen2.5-VL 7B Math Evaluation"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# Change to lmms-eval directory
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"

# Check if lmms-eval is installed, install if not
if ! python -c "import lmms_eval" 2>/dev/null; then
    echo "Installing lmms-eval in editable mode..."
    pip install -e .
fi

# Create output directory with timestamp
OUTPUT_DIR="eval_results/qwen25vl_7b_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Function to run evaluation
run_eval() {
    local TASK=$1
    local INTERLEAVE=$2
    local EXTRA_ARGS=$3
    
    echo "----------------------------------------"
    echo "Running evaluation on: $TASK"
    echo "Interleave visuals: $INTERLEAVE"
    echo "----------------------------------------"
    
    OUTPUT_FILE="${OUTPUT_DIR}/${TASK}_results.json"
    
    accelerate launch --num_processes=1 --main_process_port=12346 -m lmms_eval \
        --model qwen2_5_vl \
        --model_args=pretrained=${MODEL_NAME},max_pixels=${MAX_PIXELS},attn_implementation=flash_attention_2,interleave_visuals=${INTERLEAVE},tensor_parallel_size=4 \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --output_path ${OUTPUT_FILE} \
        ${EXTRA_ARGS}
    
    echo "Results saved to: ${OUTPUT_FILE}"
    echo ""
}

# ==========================================
# 1. MathVista Evaluation
# ==========================================
echo ""
echo ">>> Evaluating on MathVista..."
run_eval "mathvista" "False" ""

# ==========================================
# 2. MathVision Evaluation
# ==========================================
echo ""
echo ">>> Evaluating on MathVision..."
run_eval "mathvision_test" "False" ""
run_eval "mathvision_reason_test" "False" ""

# ==========================================
# 3. MathVerse Evaluation
# ==========================================
echo ""
echo ">>> Evaluating on MathVerse..."
run_eval "mathverse" "False" ""

# ==========================================
# 4. MMMU Evaluation
# ==========================================
echo ""
echo ">>> Evaluating on MMMU..."
run_eval "mmmu" "True" ""



echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
