#!/bin/bash

set -x

# export NO_PROXY="localhost,127.0.0.1,100.98.129.8"
# export no_proxy="localhost,127.0.0.1,100.98.129.8"

export JUDGE_MODEL=${JUGDE_MODEL:-"/mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/models/QwenLM/Qwen2.5-32B-Instruct"}
export LLM_AS_A_JUDGE_BASE=${LLM_AS_A_JUDGE_BASE:-"http://100.99.11.26:8000/v1"}
export JUDGE_MODEL_API_KEY="EMPTY"

export WANDB_API_KEY=${WANDB_API_KEY:-wandb_*****************************************}
export WANDB_MODE=offline

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MAX_CONCURRENT_JUDGE_REQUESTS=${MAX_CONCURRENT_JUDGE_REQUESTS:-6}
export JUDGE_MIN_INTERVAL_SEC=0.005

export DEBUG_JUDGE_OUTPUT=1
export JUDGE_OUTPUT_MAX_CHARS=4000
export JUDGE_DEBUG_JSONL_PATH=${JUDGE_DEBUG_JSONL_PATH:-./judge_verify_debug_0309.jsonl}
export MAX_CONCURRENT_JUDGE_REQUESTS=2
export JUDGE_MIN_INTERVAL_SEC=0.2



PROJECT_NAME="rubrics_rl"
EXPERIMENT_NAME="Baseline-RubricsRL-pipeline-all"

BASEDIR=${PWD}
SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-${BASEDIR}/verl_checkpoints}
DATASET_TRAIN=${BASEDIR}/dataset/ViRL39K/rubrics_pipeline_all_train.parquet
DATASET_VAL=${BASEDIR}/dataset/ViRL39K/rubrics_pipeline_all_val.parquet

REF_MODEL_PATH=${REF_MODEL_PATH:-/mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/models/QwenVL/Qwen2.5-VL-7B-Instruct}
set -x
ENGINE=${1:-vllm}


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=512 \
    data.max_prompt_length=5200 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$REF_MODEL_PATH \
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
    actor_rollout_ref.rollout.name=$ENGINE \
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
    trainer.project_name='verl_vlm_rubrics_rl' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=4 \
    custom_reward_function.path=recipe/rubrics_rl/rubrics_rl.py \
    custom_reward_function.name=compute_score \
    +data.custom_cls.path=recipe/rubrics_rl/rubrics_rl.py \
    +trainer.rollout_data_dir="./rollout_dump_rubrics-rl-pipeline-all" \
    +data.custom_cls.name=RubricsRLHFDataset $@

