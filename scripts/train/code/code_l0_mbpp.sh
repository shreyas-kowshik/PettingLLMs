#!/bin/bash
# Launch script for MBPP Single Agent Training (L0 - GRPO Training)
#
# This script trains a single agent using GRPO for MBPP dataset.
# The agent generates code without using examples.
# Reward is the fraction of test cases passed.

# Disable bash command echoing for cleaner output
# set -x  # Commented out to reduce logging

# ============================================================================
# Logging Control - MINIMAL LOGGING
# ============================================================================
# Suppress verbose debug logs
export PETTINGLLMS_VERBOSE=0

# Set Python logging to WARNING level (only show warnings and errors)
export PYTHONWARNINGS="ignore"
export LOGLEVEL="WARNING"

# ============================================================================
# Environment variables for CUDA and VLLM
# ============================================================================
export CUDA_VISIBLE_DEVICES=0  # Using 1 GPU for single agent
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

# WANDB #
WANDB_ENTITY=Deep-RL-10703
WANDB_PROJECT=deeprl_project
WANDB_EXPERIMENT_NAME=mbpp_l0_single_agent

# ============================================================================
# GPU Configuration
# ============================================================================
GPU_num=1  # GPUs for single model

# ============================================================================
# Model Resource Configuration
# ============================================================================
# Define resource overrides for single model

model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num \
  $model_0_config_path.trainer.n_gpus_per_node=$GPU_num \
  $model_0_config_path.trainer.nnodes=1 \
  $model_0_config_path.trainer.n_training_gpus_per_node=$GPU_num \
  $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_num"

# ============================================================================
# Training Launch
# ============================================================================
# Rollouts calculation #
# Train batch size * train sample num rollouts for training #
python3 -m pettingllms.trainer.train \
    --config-path ../config/code \
    --config-name code_mbpp_L0 \
    $model_0_resource \
    base_models.policy_0.path=Qwen/Qwen3-1.7B \
    training.experiment_name=mbpp_l0_single_agent \
    training.total_training_steps=3000 \
    training.epoch_size=200 \
    training.train_batch_size=1 \
    training.val_batch_size=1 \
    training.train_sample_num=5 \
    training.validate_sample_num=1 \
    training.max_prompt_length=512 \
    training.max_response_length=512 \
    training.val_freq=10 \
    training.resample_freq=100 \
    training.num_workers=100 \
    env.dataset=mbpp \
    env.benchmark=mbpp \
    env.max_turns=1 2>&1 | tee logs/mbpp_l0_single_agent/code_l0_mbpp.log

