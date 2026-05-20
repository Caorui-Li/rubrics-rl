#!/bin/bash
# Training script for verl PPO with Ray.
# Single-node: this script auto-starts a local Ray cluster.
# Multi-node:  first bring up the Ray cluster on every node (head on rank 0,
#              workers via `ray start --address=...`), then run THIS script
#              ONCE on the head node only. See launch_multinode.sh for the
#              exact commands to paste on each node.

set -euo pipefail
set -x

BASEDIR=$(cd "$(dirname "$0")/../.." && pwd)

# ── Cluster configuration ──────────────────────────────────────────────────
# Set NNODES on the head node before running. Default: single-node.
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_DIR="${BASEDIR}/dataset/rubrics_mixed"
DATASET_TRAIN="${DATASET_DIR}/mixed_train.parquet"
DATASET_VAL="${DATASET_DIR}/mixed_val.parquet"

SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-"${BASEDIR}/verl_checkpoints"}
REF_MODEL_PATH=${REF_MODEL_PATH:?'REF_MODEL_PATH is required (path to Qwen2.5-VL-7B-Instruct)'}

# ── Env ────────────────────────────────────────────────────────────────────
# Judge model — set these to use an external API provider:
#
#   GPT:      JUDGE_MODEL=gpt-4o
#             LLM_AS_A_JUDGE_BASE=https://api.openai.com/v1
#             JUDGE_MODEL_API_KEY=<openai key>
#
#   Gemini:   JUDGE_MODEL=gemini-2.0-flash
#             LLM_AS_A_JUDGE_BASE=https://generativelanguage.googleapis.com/v1beta/openai
#             JUDGE_MODEL_API_KEY=<gemini key>
#
#   DeepSeek: JUDGE_MODEL=deepseek-chat
#             LLM_AS_A_JUDGE_BASE=https://api.deepseek.com/v1
#             JUDGE_MODEL_API_KEY=<deepseek key>
#
#   Local:    JUDGE_MODEL=<model-name-on-server>
#             LLM_AS_A_JUDGE_BASE=http://<host>:8000/v1
#             JUDGE_MODEL_API_KEY=EMPTY
export JUDGE_MODEL=${JUDGE_MODEL:?'JUDGE_MODEL is required (e.g. gpt-4o / gemini-2.0-flash / deepseek-chat)'}
export LLM_AS_A_JUDGE_BASE=${LLM_AS_A_JUDGE_BASE:?'LLM_AS_A_JUDGE_BASE is required'}
export JUDGE_MODEL_API_KEY=${JUDGE_MODEL_API_KEY:?'JUDGE_MODEL_API_KEY is required'}

export WANDB_API_KEY=${WANDB_API_KEY:-""}
export WANDB_MODE=offline

export MAX_CONCURRENT_JUDGE_REQUESTS=${MAX_CONCURRENT_JUDGE_REQUESTS:-6}
export JUDGE_MIN_INTERVAL_SEC=0.005
export DEBUG_JUDGE_OUTPUT=1
export JUDGE_OUTPUT_MAX_CHARS=4000
export JUDGE_DEBUG_JSONL_PATH=${JUDGE_DEBUG_JSONL_PATH:-"${BASEDIR}/judge_verify_debug.jsonl"}

# Multi-node: connect to the already-running Ray cluster on the head node.
# Single-node: leave unset so ray.init() starts a fresh local cluster.
if [ "${NNODES}" -gt 1 ]; then
    export RAY_ADDRESS=${RAY_ADDRESS:-"auto"}
fi

PROJECT_NAME="rubrics_rl"
EXPERIMENT_NAME="RubricsRL-wethink-geothought-virl39k"
ENGINE=${1:-vllm}

# ── Training ───────────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo "Training on Ray cluster: ${NNODES} nodes × ${N_GPUS_PER_NODE} GPUs"
echo "════════════════════════════════════════"
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATASET_TRAIN}" \
    data.val_files="${DATASET_VAL}" \
    data.train_batch_size=512 \
    data.max_prompt_length=5200 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path="${REF_MODEL_PATH}" \
    +actor_rollout_ref.model.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="wandb" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=4 \
    custom_reward_function.path=recipe/rubrics_rl/rubrics_rl.py \
    custom_reward_function.name=compute_score \
    +data.custom_cls.path=recipe/rubrics_rl/rubrics_rl.py \
    +data.custom_cls.name=RubricsRLHFDataset \
    +trainer.rollout_data_dir="${BASEDIR}/rollout_dump_rubrics-rl-multidata-3ds" \
    $@
