#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=0,1
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
export WANDB_API_KEY="19dc57e0b460eccee9fa05d621e17413df4f7f2f"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

GPU_num=2

model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num  $model_0_config_path.trainer.n_gpus_per_node=1 $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=1"

model_1_config_path="models.model_1.ppo_trainer_config"
model_1_resource="$model_1_config_path.trainer.n_gpus_per_node=1 $model_1_config_path.trainer.nnodes=1 $model_1_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=1"

python3 -m pettingllms.trainer.train --config-path ../config/code --config-name code_L3_model \
    $model_0_resource \
    $model_1_resource \
    base_models.policy_0.path=Qwen/Qwen2.5-Coder-1.5B-Instruct \
    base_models.policy_1.path=Qwen/Qwen2.5-Coder-1.5B-Instruct \
    training.experiment_name=code_1.5B_L3_full_model \
    training.total_training_steps=200 \
    training.epoch_size=20 \
    training.train_batch_size=4 \
    training.train_sample_num=2 \
    training.validate_sample_num=1 \
    training.max_prompt_length=400 \
    training.max_response_length=400 \
    training.val_freq=10 \
    training.resample_freq=3 \
    env.dataset=code_contests \
    env.benchmark=livecodebench
    
echo "Training completed!"