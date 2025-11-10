# PettingLLMs: Overall Architecture and Data Flow

## Introduction

**PettingLLMs** is a research framework for training multi-agent systems using **Proximal Policy Optimization (PPO)**. It supports **four specialization levels** (L0, L1, L2, L3) that control how agents share or separate their underlying language models.

**Key Innovation**: Agents can collaborate on complex tasks (math, coding, search) while learning through reinforcement learning.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration (Hydra + YAML)                  │
│  - Model paths, agent configs, training params                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        train.py (Entry)                          │
│  - Ray initialization                                            │
│  - Resource pool creation (GPU allocation)                       │
│  - Tokenizer loading                                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MultiAgentsPPOTrainer                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Per-Model Components (1 or 2+ models):                  │  │
│  │  ┌─────────────────────┐  ┌─────────────────────────┐   │  │
│  │  │  RayPPOTrainer #0   │  │  RayPPOTrainer #1       │   │  │
│  │  │  - Actor            │  │  - Actor                │   │  │
│  │  │  - Critic           │  │  - Critic               │   │  │
│  │  │  - Ref Policy       │  │  - Ref Policy           │   │  │
│  │  │  - Rollout Manager  │  │  - Rollout Manager      │   │  │
│  │  └─────────────────────┘  └─────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │       MultiAgentsExecutionEngine                          │  │
│  │  - Agent-environment interactions                         │  │
│  │  - Rollout generation (async)                             │  │
│  │  - Reward computation                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               Environment & Agent Domains                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   Math   │  │   Code   │  │  Search  │  │ Stateful │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: End-to-End Training Step

### Overview

```
1. Init Environments & Agents
2. Generate Rollouts (Async, Concurrent)
   ├─ For each environment (in parallel):
   │   └─ Multi-turn agent-env interaction
   │       ├─ Agent creates prompt
   │       ├─ LLM generates response
   │       ├─ Agent extracts action
   │       ├─ Environment steps
   │       └─ Compute reward
3. Aggregate Rollouts by Policy
4. Filter Samples (UID-based)
5. Update Model Parameters (PPO)
   ├─ Compute log probs (old, ref)
   ├─ Compute values (critic)
   ├─ Compute advantages (GAE)
   ├─ Update critic
   └─ Update actor
6. Log Metrics & Save Checkpoint
7. Validation (periodic)
```

### Detailed Flow

#### Step 1: Initialization

```python
# In MultiAgentsPPOTrainer.fit()
self.agent_execution_engine.init_agents_and_envs(mode="train", step_idx=i)
```

**Actions**:
- Load `epoch_size` problems from dataset (e.g., 20 math problems)
- Create `train_sample_num` environments per problem (e.g., 2 samples); MAYBE FOR TWO AGENTS, ONE PER AGENT?
- Total: `20 * 2 = 40` environments
- Initialize agents with policy mappings

**Example**:
```yaml
training:
  epoch_size: 20  # Problems per step
  train_sample_num: 2  # Samples per problem
  train_batch_size: 4  # Concurrent rollouts

# Creates 40 envs total, processes 4 at a time
```

---

#### Step 2: Rollout Generation

```python
# Wake up LLM servers
for rollout_engine in self.rollout_engine_dict.values():
    rollout_engine.wake_up()

# Generate rollouts (async)
gen_batch_output_per_policy = asyncio.run(
    self.agent_execution_engine.generate_multiple_rollouts_concurrent(env_idx_list)
)

# Sleep LLM servers
for rollout_engine in self.rollout_engine_dict.values():
    rollout_engine.sleep()
```

**Async Execution**:
```python
async def generate_multiple_rollouts_concurrent(env_idx_list):
    # Create task per environment
    tasks = [
        self._generate_single_rollout(env_idx)
        for env_idx in env_idx_list
    ]
    
    # Execute all concurrently
    results = await asyncio.gather(*tasks)
    
    # Aggregate by policy
    batch_per_policy = aggregate_by_policy(results)
    return batch_per_policy
```

**Single Rollout Loop**:
```python
async def _generate_single_rollout(env_idx):
    env = self.env_batch.env_list[env_idx]
    rollout_data = []
    
    for turn_idx in range(max_turns):
        for agent_idx, agent_name in enumerate(turn_order):
            # 1. Agent updates prompt from environment
            env = agent.update_from_env(turn_idx, env)
            
            # 2. Get policy for this agent
            policy_name = self.agent_policy_mapping[agent_name]
            server_address = self.server_address_dict[policy_name][0]
            
            # 3. Generate response (async API call)
            response = await llm_async_generate(
                prompts=agent.current_prompt,
                address=server_address,
                tokenizer=self.tokenizer_dict[policy_name],
                temperature=agent_config.train_temperature,
                max_new_tokens=max_response_length,
                sample_num=train_sample_num,
                lora_adapter_id=lora_id if lora_differ_mode else None
            )
            
            # 4. Agent processes response → action
            env = agent.update_from_model(response)
            
            # 5. Agent steps in environment
            await agent.step(env, env_worker)
            
            # 6. Compute reward
            agent.calculate_reward(env)
            
            # 7. Store trajectory
            rollout_data.append({
                "env_idx": env_idx,
                "turn_idx": turn_idx,
                "agent_idx": agent_idx,
                "agent_name": agent_name,
                "prompt": agent.current_prompt["text"],
                "response": response,
                "action": agent.current_action,
                "reward": agent.agent_reward,
                "done": env.done,
            })
            
            # 8. Check termination
            if env.done:
                break
        
        if env.done:
            break
    
    # Group by policy
    data_by_policy = {}
    for data in rollout_data:
        policy_name = self.agent_policy_mapping[data["agent_name"]]
        data_by_policy.setdefault(policy_name, []).append(data)
    
    return data_by_policy
```

**Key Points**:
- All environments run **concurrently** (async)
- Each environment has **multi-turn interaction**
- Each turn has **multiple agents** (based on `turn_order`)
- Trajectory data grouped by **policy name**

**Example Trajectory**:
```python
{
    "reasoning_generator_model": [
        # Rollout 0
        {"env_idx": 0, "turn_idx": 0, "agent_idx": 1, "prompt": "...", "response": "...", "reward": 0.0},
        {"env_idx": 0, "turn_idx": 1, "agent_idx": 1, "prompt": "...", "response": "...", "reward": 1.0},
        # Rollout 1
        {"env_idx": 1, "turn_idx": 0, "agent_idx": 1, "prompt": "...", "response": "...", "reward": 0.0},
        ...
    ],
    "tool_generator_model": [
        # Rollout 0
        {"env_idx": 0, "turn_idx": 0, "agent_idx": 0, "prompt": "...", "response": "...", "reward": 0.0},
        {"env_idx": 0, "turn_idx": 1, "agent_idx": 0, "prompt": "...", "response": "...", "reward": 0.0},
        # Rollout 1
        {"env_idx": 1, "turn_idx": 0, "agent_idx": 0, "prompt": "...", "response": "...", "reward": 1.0},
        ...
    ]
}
```

---

#### Step 3: UID Assignment & Filtering

```python
for model_name, trainer in self.ppo_trainer_dict.items():
    batch = batch_per_trainer[model_name]
    
    # Assign UIDs to group samples
    batch = self._assign_consistent_uids(
        batch,
        filter_ratio=trainer.config.filter_ratio,  # e.g., 0.5 = remove 50%
        mode=trainer.config.filter_method,  # "mean", "std", "dapo", "uid"
        sample_num=train_sample_num
    )
```

**UID Assignment**:
```python
# Create UID per (env_idx // sample_num, turn_idx, agent_idx)
# This groups multiple samples from same problem together
uid = f"{env_idx // sample_num}_{turn_idx}_{agent_idx}"
```

**Filtering Modes**:

1. **`mode="mean"`**: Remove UIDs with lowest average reward
2. **`mode="std"`**: Remove UIDs with lowest variance (least diverse)
3. **`mode="dapo"`**: Remove UIDs with zero variance (no diversity)
4. **`mode="uid"`**: Within each UID, remove outliers

**Example (mode="mean", filter_ratio=0.5)**:
```
UID 0: rewards=[0.0, 0.0, 0.0]  → mean=0.0  ← REMOVE (bottom 50%)
UID 1: rewards=[1.0, 0.0, 0.0]  → mean=0.33 ← REMOVE (bottom 50%)
UID 2: rewards=[1.0, 1.0, 0.0]  → mean=0.67 ← KEEP (top 50%)
UID 3: rewards=[1.0, 1.0, 1.0]  → mean=1.0  ← KEEP (top 50%)
```

---

#### Step 4: PPO Update

```python
for model_name, trainer in self.ppo_trainer_dict.items():
    batch = batch_per_trainer[model_name]
    
    # Update parameters
    updated_batch = self._update_parameters(batch, trainer, timing_raw)
```

**PPO Update Flow** (in `_update_parameters`):

```
1. Prepare Batch
   ├─ Pad prompts (left padding)
   ├─ Pad responses (right padding)
   ├─ Create attention masks
   └─ Compute position IDs

2. Compute Log Probabilities
   ├─ old_log_prob: π_old(a|s)  (current policy)
   └─ ref_log_prob: π_ref(a|s)  (reference policy, frozen)

3. Compute Values (if critic enabled)
   └─ values: V(s)

4. Compute Advantages
   └─ Use GAE (Generalized Advantage Estimation)
       A(s,a) = Σ (γλ)^t * δ_t
       where δ_t = r_t + γV(s_{t+1}) - V(s_t)

5. Update Critic (if enabled)
   └─ Minimize: MSE(V(s), returns)

6. Update Actor
   └─ Maximize: min(
         ratio * A,
         clip(ratio, 1-ε, 1+ε) * A
       ) - β * KL(π_old || π_ref)
       
       where ratio = π(a|s) / π_old(a|s)
```

**LoRA Differ Mode** (L2 with multiple agents):
```python
if self.lora_differ_mode:
    # Split batch by agent
    for agent_name in unique_agents:
        agent_batch = batch.select_idxs(agent_indices)
        
        # Update this agent's LoRA adapter
        actor_output = ppo_trainer.actor_rollout_wg.update_actor(
            agent_batch,
            lora_id=self.agent_lora_mapping[agent_name]
        )
```

---

#### Step 5: Logging & Checkpointing

```python
# Compute metrics
metrics.update({
    "training/global_step": self.global_steps,
    "model_0_actor/loss": actor_loss,
    "model_0_critic/loss": critic_loss,
    "model_0_rollout/mean_reward": mean_reward,
    ...
})

# Log to wandb/console
logger.log(data=metrics, step=self.global_steps)

# Save checkpoint (periodic)
if self.global_steps % save_freq == 0:
    for model_name, trainer in self.ppo_trainer_dict.items():
        trainer._save_checkpoint()
```

---

#### Step 6: Validation

```python
if self.global_steps % val_freq == 0:
    val_metrics = self._validate()
    metrics.update(val_metrics)
```

**Validation Flow**:
```python
def _validate(self):
    # 1. Init validation mode (different dataset split)
    self.agent_execution_engine.init_agents_and_envs(mode="validate")
    
    # 2. Generate validation rollouts
    gen_batch_output = asyncio.run(
        self.agent_execution_engine.generate_multiple_rollouts_concurrent(...)
    )
    
    # 3. Compute success rates
    for agent_name in turn_order:
        success_rollouts = self.agent_execution_engine.success_rollout_idx_list_dict[agent_name]
        success_rate = len(success_rollouts) / total_rollouts
        val_metrics[f"validation/agent_{agent_name}/success_rate"] = success_rate
    
    return val_metrics
```

---

## Specialization Levels: L0, L1, L2, L3

### Comparison Table

| Level | Name | Agents Share | Parameters | LoRA | Config Example |
|-------|------|--------------|------------|------|----------------|
| **L0** | Single Agent | N/A (1 agent) | 1 model | No | 1 model, 1 agent |
| **L1** | Prompt Specialization | Same model | 1 model | No | 1 model, 2+ agents, different prompts |
| **L2** | LoRA Specialization | Base model | 1 base + N LoRA | Yes | 1 model + LoRA rank > 0 |
| **L3** | Full Model Specialization | Nothing | N models | No | N models, each agent has own model |

---

### L0: Single Agent

**Concept**: Single agent solves the task alone.

**Configuration**:
```yaml
specialization: "prompt"  # or omit

agent_policy_configs:
  num_agents: 1
  policy_list: ["reasoning_generator"]
  agent_configs:
    agent_0:
      name: "reasoning_generator"
      policy_name: "shared_model"

base_models:
  policy_0:
    path: "Qwen/Qwen3-1.7B"
    name: "shared_model"

models:
  model_0:
    path: ${base_models.policy_0.path}
    name: ${base_models.policy_0.name}

multi_agent_interaction:
  turn_order: ["reasoning_generator"]
  num_interacting_agents: 1
```

**Data Flow**:
```
┌─────────────────────┐
│   Environment       │
│   (Math Problem)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Reasoning Agent    │
│  → Model 0          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Environment Step   │
│  → Compute Reward   │
└─────────────────────┘
```

**Pros**:
- Simple
- Fast training

**Cons**:
- No collaboration
- Limited to single perspective

---

### L1: Prompt Specialization

**Concept**: Multiple agents share the **same model**, differentiated only by their **system prompts**.

**Configuration**:
```yaml
specialization: "prompt"

agent_policy_configs:
  num_agents: 2
  policy_list: ["reasoning_generator", "tool_generator"]
  agent_configs:
    agent_0:
      name: "reasoning_generator"
      policy_name: "shared_model"  # Same model!
    agent_1:
      name: "tool_generator"
      policy_name: "shared_model"  # Same model!

base_models:
  policy_0:
    path: "Qwen/Qwen3-1.7B"
    name: "shared_model"

models:
  model_0:  # Only 1 model
    path: ${base_models.policy_0.path}
    name: ${base_models.policy_0.name}

multi_agent_interaction:
  turn_order: ["tool_generator", "reasoning_generator"]
  num_interacting_agents: 2
```

**Data Flow**:
```
┌──────────────────────────────────────────────┐
│            Environment                        │
│            (Math Problem)                     │
└──────────┬───────────────────────────────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
┌─────────┐  ┌─────────┐
│ Tool    │  │Reasoning│
│ Agent   │  │ Agent   │
│ Prompt 1│  │Prompt 2 │
└────┬────┘  └────┬────┘
     │            │
     └─────┬──────┘
           ▼
    ┌─────────────┐
    │   Model 0   │
    │  (Shared)   │
    └─────────────┘
```

**Agent Differentiation**:
- **Tool Agent**: "You are a code generator. Write Python code to solve the problem."
- **Reasoning Agent**: "You are a reasoning assistant. Think step-by-step and output the answer."

**Training**:
- Both agents' trajectories train the **same model**
- Model learns to respond appropriately based on prompt

**Pros**:
- Minimal memory overhead (1 model)
- Fast training
- Agents share knowledge

**Cons**:
- Agents may not specialize well
- Prompt engineering required

**Use Cases**:
- Quick prototyping
- Limited GPU memory
- Tasks where agents have similar capabilities

---

### L2: LoRA Specialization

**Concept**: Multiple agents share the **same base model** but have **separate LoRA adapters** trained independently.

**Configuration**:
```yaml
specialization: "lora"

training:
  lora_rank: 16
  lora_alpha: 32

agent_policy_configs:
  num_agents: 2
  agent_configs:
    agent_0:
      name: "reasoning_generator"
      policy_name: "shared_model"  # Same base model
    agent_1:
      name: "tool_generator"
      policy_name: "shared_model"  # Same base model

base_models:
  policy_0:
    path: "Qwen/Qwen3-1.7B"
    name: "shared_model"

models:
  model_0:  # Only 1 base model
    path: ${base_models.policy_0.path}
    name: ${base_models.policy_0.name}
    ppo_trainer_config:
      actor_rollout_ref:
        model:
          lora_rank: 16  # Enable LoRA
          lora_alpha: 32

multi_agent_interaction:
  turn_order: ["tool_generator", "reasoning_generator"]
  num_interacting_agents: 2
```

**Architecture**:
```
┌─────────────────────────────────────────────┐
│           Base Model (Frozen)                │
│         Qwen/Qwen3-1.7B                      │
└──────────┬────────────────────┬──────────────┘
           │                    │
      ┌────▼────┐          ┌────▼────┐
      │ LoRA    │          │ LoRA    │
      │Adapter 0│          │Adapter 1│
      │(Tool)   │          │(Reason) │
      └────┬────┘          └────┬────┘
           │                    │
      ┌────▼────┐          ┌────▼────┐
      │  Tool   │          │Reasoning│
      │  Agent  │          │  Agent  │
      └─────────┘          └─────────┘
```

**LoRA Adapters**:
- **Low-Rank Adaptation**: Only trains small matrices (rank 16)
- **Per-Agent**: Each agent has its own LoRA adapter
- **Independent Updates**: Agents' gradients update different adapters

**Training Flow**:
```python
# In _update_parameters():
if self.lora_differ_mode:
    # Split batch by agent
    for agent_name in ["tool_generator", "reasoning_generator"]:
        agent_batch = batch.select_idxs(agent_indices)
        lora_id = self.agent_lora_mapping[agent_name]  # e.g., "agent_tool_generator_lora_0"
        
        # Update this agent's LoRA adapter
        actor_output = ppo_trainer.actor_rollout_wg.update_actor(
            agent_batch,
            lora_id=lora_id
        )
```

**Checkpoint Structure**:
```
checkpoints/math_1.7B_L2/
├── model_0/
│   └── lora_adapters/
│       ├── lora_adapter_agent_tool_generator_lora_0/
│       │   ├── adapter_config.json
│       │   └── adapter_model.bin  (only ~20MB)
│       └── lora_adapter_agent_reasoning_generator_lora_1/
│           ├── adapter_config.json
│           └── adapter_model.bin  (only ~20MB)
└── base_model/  (NOT saved, use HuggingFace path)
```

**Pros**:
- **Memory Efficient**: Only 1 base model loaded
- **Specialized**: Each agent has independent parameters
- **Fast**: LoRA adapters are small

**Cons**:
- **Limited Capacity**: LoRA has less expressive power than full models
- **Shared Base**: Base model is frozen, limits specialization

**Use Cases**:
- **Medium GPU memory** (e.g., 1-2 A100s)
- **Strong base model** that needs task-specific tuning
- **Quick iteration** on agent specialization

---

### L3: Full Model Specialization

**Concept**: Each agent has a **completely separate model** with **independent parameters**.

**Configuration**:
```yaml
specialization: "full"

agent_policy_configs:
  num_agents: 2
  agent_configs:
    agent_0:
      name: "reasoning_generator"
      policy_name: "reasoning_generator_model"  # Different!
    agent_1:
      name: "tool_generator"
      policy_name: "tool_generator_model"  # Different!

base_models:
  policy_0:
    path: "Qwen/Qwen3-1.7B"
    name: "reasoning_generator_model"
  policy_1:
    path: "Qwen/Qwen3-1.7B"  # Can be different model!
    name: "tool_generator_model"

models:
  model_0:  # First independent model
    path: ${base_models.policy_0.path}
    name: ${base_models.policy_0.name}
    ppo_trainer_config:
      # Full PPO config for model 0
      ...
  model_1:  # Second independent model
    path: ${base_models.policy_1.path}
    name: ${base_models.policy_1.name}
    ppo_trainer_config:
      # Full PPO config for model 1
      ...

multi_agent_interaction:
  turn_order: ["tool_generator", "reasoning_generator"]
  num_interacting_agents: 2

resource:
  n_gpus_per_node: 2  # Allocate GPUs
```

**Architecture**:
```
┌─────────────────────────────────────────────┐
│          GPU 0                GPU 1          │
│  ┌──────────────────┐  ┌──────────────────┐ │
│  │    Model 0       │  │    Model 1       │ │
│  │ (Tool Generator) │  │(Reasoning Gen)   │ │
│  │  Qwen3-1.7B      │  │  Qwen3-1.7B      │ │
│  └────────┬─────────┘  └────────┬─────────┘ │
└───────────┼─────────────────────┼───────────┘
            │                     │
       ┌────▼────┐           ┌────▼────┐
       │  Tool   │           │Reasoning│
       │  Agent  │           │  Agent  │
       └─────────┘           └─────────┘
```

**GPU Allocation**:
```python
# In train.py:
n_gpus_per_model = n_gpus_per_node // model_num
# For 2 GPUs, 2 models: 2 // 2 = 1 GPU per model

resource_pool_spec = {
    "global_pool_model_0": [1],  # Model 0 gets GPU 0
    "global_pool_model_1": [1],  # Model 1 gets GPU 1
}
```

**Training Flow**:
```python
# In fit():
for model_name, trainer in self.ppo_trainer_dict.items():
    batch = batch_per_trainer[model_name]
    
    # Each model trains independently
    self._update_parameters(batch, trainer, timing_raw)
```

**Parallel Updates**:
```python
# In fit():
if len(self.ppo_trainer_dict) > 1:
    # Update models in parallel (threading)
    with ThreadPoolExecutor(max_workers=len(self.ppo_trainer_dict)) as executor:
        futures = []
        for model_name, trainer in self.ppo_trainer_dict.items():
            future = executor.submit(
                self._update_parameters,
                batch_per_trainer[model_name],
                trainer,
                timing_raw
            )
            futures.append(future)
        
        # Wait for all updates
        for future in as_completed(futures):
            result = future.result()
```

**Checkpoint Structure**:
```
checkpoints/math_1.7B_L3/
├── reasoning_generator_model/
│   ├── pytorch_model.bin  (full model, ~3GB)
│   ├── config.json
│   └── tokenizer/
└── tool_generator_model/
    ├── pytorch_model.bin  (full model, ~3GB)
    ├── config.json
    └── tokenizer/
```

**Pros**:
- **Maximum Specialization**: Each agent has completely independent parameters
- **Different Models**: Can use different architectures per agent
- **Full Expressiveness**: No LoRA limitations

**Cons**:
- **High Memory**: N models loaded simultaneously
- **Slow Training**: More parameters to update
- **Expensive**: Requires multiple GPUs

**Use Cases**:
- **High GPU memory** (e.g., 4+ A100s)
- **Maximum performance** requirements
- **Heterogeneous agents** (e.g., small code model + large reasoning model)

---

## Comparison: L1 vs L2 vs L3

### Training Speed

| Metric | L1 | L2 | L3 |
|--------|----|----|-----|
| **Rollout Speed** | Fast | Fast | Medium |
| **Update Speed** | Fast | Fast | Slow |
| **Memory** | Low | Low | High |
| **Scalability** | Best | Good | Poor |

### Performance

| Metric | L1 | L2 | L3 |
|--------|----|----|-----|
| **Specialization** | Low | Medium | High |
| **Final Performance** | Good | Better | Best |
| **Convergence** | Fast | Medium | Slow |

### Resource Requirements

| Setup | GPUs | VRAM | Training Time |
|-------|------|------|---------------|
| **L1** (2 agents) | 1 | 40GB | 10 hrs |
| **L2** (2 agents) | 1 | 40GB | 12 hrs |
| **L3** (2 agents) | 2 | 2×40GB | 20 hrs |

---

## Configuration Inheritance

### Hydra Defaults

PettingLLMs uses **Hydra** for hierarchical configuration.

**Example**: `math_L3_fresh.yaml`

```yaml
defaults:
  - ../ppo_trainer@models.model_0.ppo_trainer_config: eval
  - ../ppo_trainer@models.model_1.ppo_trainer_config: eval
  - _self_

# This inherits from ppo_trainer/eval.yaml for both models
```

**Inheritance Chain**:
```
config/math/math_L3_fresh.yaml
  ├─ Inherits: config/ppo_trainer/eval.yaml → models.model_0.ppo_trainer_config
  ├─ Inherits: config/ppo_trainer/eval.yaml → models.model_1.ppo_trainer_config
  └─ Overrides: Local configs take precedence
```

### Base PPO Config

**File**: `config/ppo_trainer/eval.yaml`

```yaml
filter_method: mean
filter_ratio: 0.5

data:
  max_prompt_length: 512
  max_response_length: 512

actor_rollout_ref:
  model:
    path: ???  # Must be overridden
  rollout:
    temperature: 1.0
    prompt_length: ${data.max_prompt_length}
    response_length: ${data.max_response_length}
    tensor_model_parallel_size: 1
  trainer:
    n_gpus_per_node: 1
    nnodes: 1

algorithm:
  adv_estimator: gae
  gamma: 0.99
  lam: 0.95
```

### Override via Shell Script

**File**: `scripts/train/math/math_L3_fresh.sh`

```bash
python3 -m pettingllms.trainer.train \
    --config-path ../config/math \
    --config-name math_L3_fresh \
    resource.n_gpus_per_node=2 \
    models.model_0.ppo_trainer_config.trainer.n_gpus_per_node=2 \
    models.model_1.ppo_trainer_config.trainer.n_gpus_per_node=2 \
    base_models.policy_0.path=Qwen/Qwen3-1.7B \
    base_models.policy_1.path=Qwen/Qwen3-1.7B \
    training.total_training_steps=200
```

---

## Common Issues & Solutions

### Issue 1: Too Many Ray Workers

**Symptom**:
```
WARNING: 1800 PYTHON worker processes have been started
```

**Cause**: Default `num_workers=1800` is too high.

**Solution**:
```yaml
# In config or command line:
training:
  num_workers: 150  # Reduce to 150-300
```

---

### Issue 2: L3 Config Errors

**Symptom**:
```
ConfigKeyError: Key 'train_batch_size' is not in struct
```

**Cause**: Incomplete `ppo_trainer_config` for model_1 in L3.

**Solution**: Ensure **both** models have full `ppo_trainer_config`:

```yaml
defaults:
  - ../ppo_trainer@models.model_0.ppo_trainer_config: eval  # ✓
  - ../ppo_trainer@models.model_1.ppo_trainer_config: eval  # ✓ (don't forget!)
  - _self_
```

---

### Issue 3: Agent-Policy Mismatch

**Symptom**:
```
KeyError: 'tool_generator_model' not found in server_address_dict
```

**Cause**: Agent's `policy_name` doesn't match any model's `name`.

**Solution**: Ensure consistency:

```yaml
agent_policy_configs:
  agent_configs:
    agent_1:
      policy_name: "tool_generator_model"  # Must match ↓

models:
  model_1:
    name: "tool_generator_model"  # ✓ Matches
```

---

### Issue 4: LoRA Not Loading

**Symptom**: LoRA mode enabled but adapters not used.

**Cause**: `lora_rank` not set in model config.

**Solution**:
```yaml
models:
  model_0:
    ppo_trainer_config:
      actor_rollout_ref:
        model:
          lora_rank: 16  # Must be > 0
          lora_alpha: 32
```

---

## Summary

### Key Takeaways

1. **Specialization Levels**:
   - L1: Prompt-based (same model, different prompts)
   - L2: LoRA-based (same base, different adapters)
   - L3: Full model (separate models)

2. **Data Flow**:
   - Init envs → Generate rollouts (async) → Filter samples → Update models (PPO) → Validate

3. **Training Loop**:
   - Concurrent rollout generation (all envs in parallel)
   - Multi-turn agent-env interaction
   - Grouped by policy for updates

4. **Configuration**:
   - Hydra-based inheritance
   - Override via YAML or command line
   - Ensure agent-policy mapping consistency

5. **Performance**:
   - L1: Fastest, lowest memory
   - L2: Balanced (memory + specialization)
   - L3: Best performance, highest cost

---

This concludes the comprehensive PettingLLMs documentation! 🎉

