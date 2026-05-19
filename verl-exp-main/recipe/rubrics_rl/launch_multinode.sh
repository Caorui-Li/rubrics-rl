#!/bin/bash
# Multi-node training launcher (Ray-based verl).
# Run prepare_data_multidata.sh first (once, on any single node).
#
# This script:
#   1. Starts a Ray head on NODES[0]
#   2. Starts Ray workers on the remaining nodes
#   3. Runs the training script once on the head node

set -euo pipefail

# ── Node configuration ─────────────────────────────────────────────────────
# List all node IPs/hostnames in order; index 0 is the head node.
NODES=(
    "10.0.0.1"   # node 0 — Ray head
    "10.0.0.2"   # node 1
    # "10.0.0.3" # add more nodes here
)

HEAD="${NODES[0]}"
RAY_PORT=${RAY_PORT:-6379}
NNODES="${#NODES[@]}"
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ── Shared settings ────────────────────────────────────────────────────────
# Required — must be set before running this script.
: "${REF_MODEL_PATH:?'REF_MODEL_PATH is required'}"
: "${JUDGE_MODEL:?'JUDGE_MODEL is required'}"
: "${LLM_AS_A_JUDGE_BASE:?'LLM_AS_A_JUDGE_BASE is required'}"
: "${JUDGE_MODEL_API_KEY:?'JUDGE_MODEL_API_KEY is required'}"

# Optional
DATA_ROOT=${DATA_ROOT:-""}
SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-""}
WANDB_API_KEY=${WANDB_API_KEY:-""}

# Path to the training script on the head node.
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$(cd "$(dirname "$0")" && pwd)/run_rubrics_rl_multidata.sh"}

# ── Step 1: Start Ray head ─────────────────────────────────────────────────
echo "Starting Ray head on ${HEAD}:${RAY_PORT} ..."
ssh -o StrictHostKeyChecking=no "${HEAD}" \
    "ray stop --force 2>/dev/null || true; ray start --head --port=${RAY_PORT} --num-gpus=${NGPUS_PER_NODE}"

# ── Step 2: Start Ray workers ──────────────────────────────────────────────
for i in "${!NODES[@]}"; do
    [ "$i" -eq 0 ] && continue
    NODE="${NODES[$i]}"
    echo "Starting Ray worker on node ${i}: ${NODE} ..."
    ssh -o StrictHostKeyChecking=no "${NODE}" \
        "ray stop --force 2>/dev/null || true; ray start --address='${HEAD}:${RAY_PORT}' --num-gpus=${NGPUS_PER_NODE}"
done

echo "Ray cluster started. Waiting 5s for workers to register ..."
sleep 5

# ── Step 3: Run training on head node ─────────────────────────────────────
echo "Launching training on head node ${HEAD} ..."
ssh -o StrictHostKeyChecking=no "${HEAD}" bash -s << EOF
export NNODES=${NNODES}
export RAY_ADDRESS=auto
export REF_MODEL_PATH="${REF_MODEL_PATH}"
export JUDGE_MODEL="${JUDGE_MODEL}"
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE}"
export JUDGE_MODEL_API_KEY="${JUDGE_MODEL_API_KEY}"
$([ -n "${DATA_ROOT}" ] && echo "export DATA_ROOT='${DATA_ROOT}'" || true)
$([ -n "${SAVE_CHECKPOINT_DIR}" ] && echo "export SAVE_CHECKPOINT_DIR='${SAVE_CHECKPOINT_DIR}'" || true)
$([ -n "${WANDB_API_KEY}" ] && echo "export WANDB_API_KEY='${WANDB_API_KEY}'" || true)
bash "${TRAIN_SCRIPT}"
EOF

echo "Training finished."
