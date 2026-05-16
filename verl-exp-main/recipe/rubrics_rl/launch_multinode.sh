#!/bin/bash
# Multi-node training launcher for verl (Ray-based, no SSH required).
#
# Pipeline:
#   1. prepare_data_multidata.sh   (run once on any single node)
#   2. configure NODES below + export required env vars on the host you run
#      THIS script on
#   3. bash launch_multinode.sh    (prints per-node commands)
#   4. on each node, paste the block this script prints for that node
#
# verl PPO uses Ray to schedule actors across nodes. Steps inside each block:
#   - head node:   ray start --head ...        →   bash run_rubrics_rl_multidata.sh
#   - worker node: ray start --address=...     →   (just stays joined; no python here)
#
# Only the head node runs the training script — Ray dispatches workers itself.

set -euo pipefail

# ── Node configuration ─────────────────────────────────────────────────────
# List all node IPs/hostnames in order; index 0 is the head node.
NODES=(
    "10.0.0.1"   # node 0 — head
    "10.0.0.2"   # node 1 — worker
    # "10.0.0.3" # add more workers here
)

HEAD_ADDR="${NODES[0]}"
RAY_PORT=${RAY_PORT:-6379}
NNODES="${#NODES[@]}"
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

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

# Path to the training script on the head node (must exist at the same path
# on every node, since Ray workers may need to import recipe code).
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$(cd "$(dirname "$0")" && pwd)/run_rubrics_rl_multidata.sh"}

# ── Helper: print the shared `export` lines for a node ─────────────────────
print_exports() {
    echo "export REF_MODEL_PATH='${REF_MODEL_PATH}'"
    echo "export JUDGE_MODEL='${JUDGE_MODEL}'"
    echo "export LLM_AS_A_JUDGE_BASE='${LLM_AS_A_JUDGE_BASE}'"
    echo "export JUDGE_MODEL_API_KEY='${JUDGE_MODEL_API_KEY}'"
    if [ -n "${DATA_ROOT}" ]; then
        echo "export DATA_ROOT='${DATA_ROOT}'"
    fi
    if [ -n "${SAVE_CHECKPOINT_DIR}" ]; then
        echo "export SAVE_CHECKPOINT_DIR='${SAVE_CHECKPOINT_DIR}'"
    fi
    if [ -n "${WANDB_API_KEY}" ]; then
        echo "export WANDB_API_KEY='${WANDB_API_KEY}'"
    fi
    echo "export NNODES=${NNODES}"
    echo "export N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
}

# ── Print per-node commands ────────────────────────────────────────────────
echo "Multi-node Ray cluster: ${NNODES} nodes, head ${HEAD_ADDR}:${RAY_PORT}"
echo
echo "Order of operations:"
echo "  1) Run the head block on node 0 first."
echo "  2) Wait until 'ray start --head' returns, then run each worker block."
echo "  3) After all workers have joined, run the final 'bash ...' line on the"
echo "     head node — Ray will dispatch actors to every node automatically."
echo

# ── Node 0: head ───────────────────────────────────────────────────────────
echo "────────────────────────────────────────────────────────────────"
echo "Node 0 (HEAD) — paste on host ${NODES[0]}"
echo "────────────────────────────────────────────────────────────────"
print_exports
echo "ray stop || true"
echo "ray start --head --port=${RAY_PORT} --node-ip-address=${HEAD_ADDR} --num-gpus=${N_GPUS_PER_NODE} --dashboard-host=0.0.0.0"
echo "# After every worker has joined (check with: ray status), run:"
echo "bash ${TRAIN_SCRIPT}"
echo

# ── Worker nodes ───────────────────────────────────────────────────────────
for i in "${!NODES[@]}"; do
    if [ "$i" -eq 0 ]; then
        continue
    fi
    NODE="${NODES[$i]}"
    echo "────────────────────────────────────────────────────────────────"
    echo "Node ${i} (WORKER) — paste on host ${NODE}"
    echo "────────────────────────────────────────────────────────────────"
    print_exports
    echo "ray stop || true"
    echo "ray start --address=${HEAD_ADDR}:${RAY_PORT} --node-ip-address=${NODE} --num-gpus=${N_GPUS_PER_NODE}"
    echo "# Worker is now joined. Do NOT run the training script here — only"
    echo "# the head node runs python -m verl.trainer.main_ppo (via bash ${TRAIN_SCRIPT})."
    echo
done

echo "When training finishes (or to abort), run 'ray stop' on every node."
