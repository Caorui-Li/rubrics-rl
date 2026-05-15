#!/bin/bash

set -euo pipefail
set -x

# ── Paths ──────────────────────────────────────────────────────────────────
BASEDIR=$(cd "$(dirname "$0")/../.." && pwd)

# DATA_ROOT: root directory for all downloaded datasets and extracted images.
# Override to point to a large-storage mount when local disk is limited.
# Example: export DATA_ROOT=/mnt/storage/data
export DATA_ROOT=${DATA_ROOT:-"${BASEDIR}/data"}

WETHINK_JSONL="${DATA_ROOT}/wethink_rubrics/wethink_rubrics_20k_processed.jsonl"
WETHINK_IMAGES="${DATA_ROOT}/wethink_rubrics/images"

GEOTHOUGHT_JSONL="${DATA_ROOT}/geothought_rubrics/geothought_rubrics_processed.jsonl"
GEOTHOUGHT_IMAGES="${DATA_ROOT}/geothought_rubrics/images"

VIRL39K_RUBRICS_JSONL="${DATA_ROOT}/virl39k_rubrics/virl39k_rubrics_processed.jsonl"
VIRL39K_RUBRICS_IMAGES="${DATA_ROOT}/virl39k_rubrics/images"

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

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MAX_CONCURRENT_JUDGE_REQUESTS=${MAX_CONCURRENT_JUDGE_REQUESTS:-6}
export JUDGE_MIN_INTERVAL_SEC=0.005
export DEBUG_JUDGE_OUTPUT=1
export JUDGE_OUTPUT_MAX_CHARS=4000
export JUDGE_DEBUG_JSONL_PATH=${JUDGE_DEBUG_JSONL_PATH:-"${BASEDIR}/judge_verify_debug.jsonl"}

PROJECT_NAME="rubrics_rl"
EXPERIMENT_NAME="RubricsRL-wethink-geothought-virl39k"
ENGINE=${1:-vllm}

# ── Data prep (Steps 1-4) ──────────────────────────────────────────────────
# Skip when the final parquets already exist (idempotent re-runs) and only
# run on NODE_RANK=0 to avoid racing writers on shared FS during multinode.
if [[ -f "${DATASET_TRAIN}" && -f "${DATASET_VAL}" ]]; then
    echo "Found ${DATASET_TRAIN} and ${DATASET_VAL}, skipping data prep."
elif [[ "${NODE_RANK:-0}" != "0" ]]; then
    echo "NODE_RANK=${NODE_RANK} > 0, waiting for master to finish data prep ..."
    while [[ ! -f "${DATASET_TRAIN}" || ! -f "${DATASET_VAL}" ]]; do
        sleep 10
    done
    echo "Master finished data prep, continuing."
else
    echo "════════════════════════════════════════"
    echo "Step 1: Extract wethink images"
    echo "════════════════════════════════════════"
    python3 "${BASEDIR}/data/extract_wethink_images.py"

    echo "════════════════════════════════════════"
    echo "Step 2: Extract geothought images"
    echo "════════════════════════════════════════"
    python3 "${BASEDIR}/data/extract_geothought_images.py"

    echo "════════════════════════════════════════"
    echo "Step 3: Extract virl39k rubrics images"
    echo "════════════════════════════════════════"
    python3 "${BASEDIR}/data/extract_virl39k_rubrics_images.py"

    echo "════════════════════════════════════════"
    echo "Step 4: Convert to verl parquet"
    echo "════════════════════════════════════════"
    python3 "${BASEDIR}/recipe/rubrics_rl/rubrics_gen/convert_wethink_to_verl.py" \
        --input_jsonl \
            "${WETHINK_JSONL}" \
            "${GEOTHOUGHT_JSONL}" \
            "${VIRL39K_RUBRICS_JSONL}" \
        --output_parquet "${DATASET_DIR}/mixed.parquet" \
        --image_base_paths \
            "${WETHINK_IMAGES}" \
            "${GEOTHOUGHT_IMAGES}" \
            "${VIRL39K_RUBRICS_IMAGES}" \
        --train_val_split 0.9 \
        --seed 42
fi

# ── Step 5: Train ──────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo "Step 5: Training"
echo "════════════════════════════════════════"
python3 -m verl.trainer.main_ppo \
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
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
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
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8} \
    trainer.nnodes=${NNODES:-1} \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=4 \
    custom_reward_function.path=recipe/rubrics_rl/rubrics_rl.py \
    custom_reward_function.name=compute_score \
    +data.custom_cls.path=recipe/rubrics_rl/rubrics_rl.py \
    +data.custom_cls.name=RubricsRLHFDataset \
    +trainer.rollout_data_dir="${BASEDIR}/rollout_dump_rubrics-rl-multidata-3ds" \
    $@
