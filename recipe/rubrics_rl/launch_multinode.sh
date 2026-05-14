#!/bin/bash
# Multi-node training launcher.
# Run prepare_data_multidata.sh first (once, on any single node).
# Then configure NODES below and run this script once — it SSHes into every
# node and starts run_rubrics_rl_multidata.sh with the correct NODE_RANK.

set -euo pipefail

# ── Node configuration ─────────────────────────────────────────────────────
# List all node IPs/hostnames in order; index 0 is the master node.
NODES=(
    "10.0.0.1"   # node 0 — master
    "10.0.0.2"   # node 1
    # "10.0.0.3" # add more nodes here
)

MASTER_ADDR="${NODES[0]}"
MASTER_PORT=${MASTER_PORT:-29500}
NNODES="${#NODES[@]}"

# ── Shared settings ────────────────────────────────────────────────────────
# These are forwarded to every node via SSH.
# Required — must be set before running this script.
: "${REF_MODEL_PATH:?'REF_MODEL_PATH is required'}"
: "${JUDGE_MODEL:?'JUDGE_MODEL is required'}"
: "${LLM_AS_A_JUDGE_BASE:?'LLM_AS_A_JUDGE_BASE is required'}"
: "${JUDGE_MODEL_API_KEY:?'JUDGE_MODEL_API_KEY is required'}"

# Optional
DATA_ROOT=${DATA_ROOT:-""}
SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-""}
WANDB_API_KEY=${WANDB_API_KEY:-""}

# Path to the training script on each node (must be the same on all nodes).
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$(cd "$(dirname "$0")" && pwd)/run_rubrics_rl_multidata.sh"}

# ── Launch ─────────────────────────────────────────────────────────────────
echo "Launching on ${NNODES} nodes: ${NODES[*]}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

PIDS=()
for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    echo "  → node ${i}: ${NODE}"

    ssh -o StrictHostKeyChecking=no "${NODE}" bash -s -- "${i}" << EOF &
export NODE_RANK=${i}
export NNODES=${NNODES}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export REF_MODEL_PATH="${REF_MODEL_PATH}"
export JUDGE_MODEL="${JUDGE_MODEL}"
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE}"
export JUDGE_MODEL_API_KEY="${JUDGE_MODEL_API_KEY}"
$([ -n "${DATA_ROOT}" ] && echo "export DATA_ROOT='${DATA_ROOT}'" || true)
$([ -n "${SAVE_CHECKPOINT_DIR}" ] && echo "export SAVE_CHECKPOINT_DIR='${SAVE_CHECKPOINT_DIR}'" || true)
$([ -n "${WANDB_API_KEY}" ] && echo "export WANDB_API_KEY='${WANDB_API_KEY}'" || true)
bash "${TRAIN_SCRIPT}"
EOF
    PIDS+=($!)
done

# Wait for all nodes; exit with error if any node fails.
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "ERROR: node ${i} (${NODES[$i]}) failed." >&2
        FAILED=1
    fi
done

if [ "${FAILED}" -eq 0 ]; then
    echo "All nodes finished successfully."
else
    echo "One or more nodes failed." >&2
    exit 1
fi
