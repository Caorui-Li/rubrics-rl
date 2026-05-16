#!/bin/bash

set -x

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

export WANDB_API_KEY=wandb_**********************************
export WANDB_MODE=offline
export WANDB_DIR=/mnt/shared-storage-user/colab-share/liujiaheng/workspace/caoruili/rubrics/verl-exp/verl_wandb

# ln -s /mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/datasets/ViRL39K/images \
#     /mnt/shared-storage-user/colab-share/liujiaheng/workspace/caoruili/rubrics/verl-exp/images

PROJECT_NAME="virl39k_simple"
EXPERIMENT_NAME="Baseline_total"

BASEDIR=${PWD}
SAVE_CHECKPOINT_DIR=${BASEDIR}/verl_checkpoints
DATASET_TRAIN=/mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/datasets/ViRL39K/virl39k_train.parquet
DATASET_VAL=/mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/datasets/ViRL39K/virl39k_val.parquet

REF_MODEL_PATH=/mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/models/QwenVL/Qwen2.5-VL-7B-Instruct
ENGINE=${1:-vllm}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=128 \
    data.max_prompt_length=40960 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.truncation=left \
    data.prompt_key=question \
    data.image_key=image \
    actor_rollout_ref.model.path=$REF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl-exp-virl39k-simple' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    custom_reward_function.path=recipe/rubrics_rl/reward_simple_virl39k.py \
    custom_reward_function.name=compute_score \
    +data.custom_cls.path=recipe/rubrics_rl/reward_simple_virl39k.py \
    +trainer.rollout_data_dir="./rollout_dump_virl39k" \
    +data.image_root=/mnt/shared-storage-user/colab-share/liujiaheng/pjlab-oss/datasets/ViRL39K \
    +data.custom_cls.name=ViRL39KSimpleDataset $@
