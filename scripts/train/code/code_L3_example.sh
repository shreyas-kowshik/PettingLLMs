#!/bin/bash
# Launch script for Example Agent + Code Agent Training (L3 - Full Model Specialization)
#
# This script trains two agents with SEPARATE, INDEPENDENT models:
# 1. Example Agent: Has its own model for generating solved examples
# 2. Code Agent: Has its own model for generating code
#
# Reward Structure:
# - Code conditions on example: Both agents get 0 reward
# - Code does NOT condition: Both agents get test pass ratio reward
#
# This encourages independent problem-solving.

set -x

# ============================================================================
# Verbose logging control (optional)
# ============================================================================
# By default, debug logs are suppressed for cleaner output
# Uncomment next line to enable verbose debug logging:
# export PETTINGLLMS_VERBOSE=1

# ============================================================================
# Environment variables for CUDA and VLLM
# ============================================================================
export CUDA_VISIBLE_DEVICES=0,1  # Using 2 GPUs (1 per model)
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

# ============================================================================
# GPU Configuration
# ============================================================================
GPU_num=1  # GPUs per model (2 total GPUs, 1 per model)

# ============================================================================
# Model Resource Configuration
# ============================================================================
# Define resource overrides for BOTH models (model_0 and model_1)

# Model 0: Example Generator
model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=2 \
  $model_0_config_path.trainer.n_gpus_per_node=$GPU_num \
  $model_0_config_path.trainer.nnodes=1 \
  $model_0_config_path.trainer.n_training_gpus_per_node=$GPU_num \
  $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_num"

# Model 1: Code with Example Agent
model_1_config_path="models.model_1.ppo_trainer_config"
model_1_resource="$model_1_config_path.trainer.n_gpus_per_node=$GPU_num \
  $model_1_config_path.trainer.nnodes=1 \
  $model_1_config_path.trainer.n_training_gpus_per_node=$GPU_num \
  $model_1_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_num"

# ============================================================================
# Training Launch
# ============================================================================
python3 -m pettingllms.trainer.train \
    --config-path ../config/code \
    --config-name code_L3_example \
    $model_0_resource \
    $model_1_resource \
    base_models.policy_0.path=Qwen/Qwen3-1.7B \
    base_models.policy_1.path=Qwen/Qwen3-1.7B \
    training.experiment_name=code_1.7B_L3_example \
    training.total_training_steps=200 \
    training.epoch_size=20 \
    training.train_batch_size=4 \
    training.train_sample_num=2 \
    training.validate_sample_num=2 \
    training.max_prompt_length=2048 \
    training.max_response_length=1024 \
    training.val_freq=10 \
    training.resample_freq=3 \
    training.num_workers=100 \
    env.dataset=apps \
    env.benchmark=livecodebench \
    env.max_turns=1

