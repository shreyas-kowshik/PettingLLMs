# Evaluate Module Documentation

## Overview

The `evaluate/` module provides evaluation capabilities for trained multi-agent models. It allows running inference on test datasets without training, using pre-trained checkpoints.

**Total Lines of Code**: ~256 lines across 2 files

## Architecture

```
evaluate/
├── evaluate.py (256 lines) - Main evaluation logic
└── vllm_id_token_proxy.py - VLLM proxy utilities
```

---

## Files

### 1. `evaluate.py`

#### Purpose
- Load trained models and run evaluation on test sets
- Support multi-agent evaluation with LoRA adapters
- Compute success metrics per agent
- No training - inference only

#### Key Functions

##### `init_agent_execution_engine(config, address)`

Creates the execution engine for evaluation mode.

```python
def init_agent_execution_engine(config: DictConfig, address: str):
    # Initialize tokenizers for all models
    tokenizer_dict = {}
    for model_key, model_config in config.models.items():
        tokenizer = hf_tokenizer(model_path, trust_remote_code)
        tokenizer_dict[model_name] = tokenizer
    
    # Setup server addresses
    server_address_dict = {model_name: [address]}
    
    # Detect LoRA differ mode
    lora_differ_mode = False
    if config.specialization == "lora" and lora_rank > 0:
        lora_differ_mode = True
        # Create LoRA adapter mappings
        for agent_name in agent_policy_mapping:
            lora_id = f"agent_{agent_name}_lora_{idx}"
            agent_lora_mapping[agent_name] = lora_id
    
    # Create execution engine
    return MultiAgentsExecutionEngine(
        config=config,
        tokenizer_dict=tokenizer_dict,
        server_address_dict=server_address_dict,
        agent_policy_mapping=agent_policy_mapping,
        lora_differ_mode=lora_differ_mode,
        agent_lora_mapping=agent_lora_mapping
    )
```

**Key Features**:
- Reuses `MultiAgentsExecutionEngine` from training
- Loads LoRA adapters if in LoRA mode
- Maps agents to policies from config

##### `validate(config, address)`

Main evaluation function.

```python
def validate(config: DictConfig, address: str):
    # 1. Initialize execution engine
    agent_execution_engine = init_agent_execution_engine(config, address)
    
    # 2. Initialize agents and envs in validate mode
    agent_execution_engine.init_agents_and_envs(mode="validate")
    
    # 3. Generate rollouts (async)
    gen_batch_output = asyncio.run(
        agent_execution_engine.generate_multiple_rollouts_concurrent(env_idx_list)
    )
    
    # 4. Compute success metrics
    total_rollout_num = len(agent_execution_engine.rollout_idx_list)
    success_rollout_rate_dict = {}
    
    for agent_name in agent_execution_engine.turn_order:
        success_rollout_num = len(
            agent_execution_engine.success_rollout_idx_list_dict[agent_name]
        )
        success_rate = success_rollout_num / total_rollout_num
        success_rollout_rate_dict[agent_name] = success_rate
    
    return agent_execution_engine, success_rollout_idx_list_dict, success_rollout_rate_dict
```

**Execution Flow**:
```
1. Load config
2. Create execution engine
3. Init envs (validate mode)
4. Run rollouts (all async)
5. Compute success rates
6. Return results
```

##### `test(config, address)`

Simple test function to verify model connectivity.

```python
def test(config: DictConfig, address: str):
    # Test prompt
    prompt = "Hello, who are you?"
    
    # Load tokenizer
    tokenizer = hf_tokenizer(model_path, trust_remote_code)
    
    # Convert to DataProto
    prompt_dpr = convert_prompt_to_dpr(
        tokenizer=tokenizer,
        prompts={"text": prompt, "image": None},
        max_prompt_length=config.data.max_prompt_length
    )
    
    # Generate response
    response = asyncio.run(
        llm_async_generate(
            prompts={"text": prompt, "image": None},
            address=address,
            tokenizer=tokenizer,
            mode="validate"
        )
    )
    
    print(f"Response: {response}")
```

**Usage**:
```bash
# Quick connectivity test
python -m pettingllms.evaluate.evaluate test \
    --config-name math_L1_prompt \
    --address "10.1.1.92:44589"
```

---

## Usage Examples

### Evaluate L1 Model (Prompt Mode)

```bash
python -m pettingllms.evaluate.evaluate validate \
    --config-path pettingllms/config/math \
    --config-name math_L1_prompt \
    base_models.policy_0.path=Qwen/Qwen3-1.7B \
    --address "10.1.1.92:44589"
```

### Evaluate L2 Model (LoRA Mode)

```bash
python -m pettingllms.evaluate.evaluate validate \
    --config-path pettingllms/config/math \
    --config-name math_L2_lora \
    base_models.policy_0.path=Qwen/Qwen3-1.7B \
    lora_checkpoint_path=checkpoints/math_1.7B_L2/lora_adapters \
    --address "10.1.1.92:44589"
```

**LoRA Checkpoint Structure**:
```
checkpoints/math_1.7B_L2/lora_adapters/
├── lora_adapter_agent_reasoning_generator_lora_0/
│   ├── adapter_config.json
│   └── adapter_model.bin
└── lora_adapter_agent_tool_generator_lora_1/
    ├── adapter_config.json
    └── adapter_model.bin
```

### Evaluate L3 Model (Full Model Mode)

```bash
python -m pettingllms.evaluate.evaluate validate \
    --config-path pettingllms/config/math \
    --config-name math_L3_fresh \
    base_models.policy_0.path=Qwen/Qwen3-1.7B \
    base_models.policy_1.path=Qwen/Qwen3-1.7B \
    address_map.reasoning_generator_model="10.1.1.92:44589" \
    address_map.tool_generator_model="10.1.1.92:44590"
```

**Note**: L3 requires separate server addresses for each model (or load both on same server).

---

## Configuration for Evaluation

### Key Differences from Training Config

| Parameter | Training | Evaluation |
|-----------|----------|------------|
| `mode` | "train" | "validate" |
| `train_sample_num` | Used | Ignored |
| `validate_sample_num` | Used | Used (1 typically) |
| `train_batch_size` | Used | Ignored (always 1) |
| `epoch_size` | Used | Ignored |

### Required Config Fields

```yaml
# Model paths
base_models:
  policy_0:
    path: "path/to/model"
    name: "model_name"

# Agent mappings
agent_policy_configs:
  agent_configs:
    agent_0:
      name: "agent_name"
      policy_name: "model_name"

# Environment
env:
  name: "math_env"  # or "code_env", etc.
  dataset: "polaris"
  benchmark: "AIME24"
  max_turns: 5

# Data limits
data:
  max_prompt_length: 512
  max_response_length: 512
```

---

## Output Metrics

### Success Rate Per Agent

```python
success_rollout_rate_dict = {
    "reasoning_generator": 0.65,  # 65% success rate
    "tool_generator": 0.58,       # 58% success rate
}
```

### Success Rollout Lists

```python
success_rollout_idx_list_dict = {
    "reasoning_generator": [0, 2, 5, 7, ...],  # Rollout IDs that succeeded
    "tool_generator": [1, 2, 6, 8, ...],
}
```

### Average Turns to Success

```python
success_ave_turn_dict = {
    "reasoning_generator": 2.3,  # Average turns for successful rollouts
    "tool_generator": 3.1,
}
```

---

## Evaluation vs Training

| Aspect | Training | Evaluation |
|--------|----------|------------|
| **Purpose** | Update model parameters | Measure performance |
| **PPO Trainer** | Required | Not required |
| **Rollout Engine** | Created by trainer | Standalone server |
| **Batch Size** | Configurable (4-32) | Always 1 |
| **Sample Num** | Multiple (2-8) | Usually 1 |
| **Metrics** | Loss, advantages, KL | Success rate only |
| **Checkpointing** | Saves models | Loads models |
| **Logging** | Wandb, console | Console only |

---

## Starting VLLM Server for Evaluation

Before running evaluation, start a VLLM server:

```bash
# For L1/L2 (single model)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --port 44589 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8

# For L2 with LoRA adapters
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --port 44589 \
    --enable-lora \
    --lora-modules \
        agent_reasoning_generator_lora_0=checkpoints/.../lora_adapter_agent_reasoning_generator_lora_0 \
        agent_tool_generator_lora_1=checkpoints/.../lora_adapter_agent_tool_generator_lora_1 \
    --max-lora-rank 16

# For L3 (two models on different ports)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --port 44589 &  # Model 0

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --port 44590 &  # Model 1
```

---

## Tips

1. **Single Sample Evaluation**: Set `validate_sample_num=1` for faster eval
2. **Batch Evaluation**: Set `validate_sample_num>1` for multiple trials per problem
3. **GPU Memory**: Reduce `--gpu-memory-utilization` if OOM during server startup
4. **Timeout**: Increase `VLLM_ENGINE_ITERATION_TIMEOUT_S` for long problems
5. **LoRA Loading**: Ensure LoRA checkpoint paths are correct and accessible

---

This concludes the Evaluate module documentation!

