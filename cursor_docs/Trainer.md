# Trainer Module Documentation

## Overview

The `trainer/` module is the **core training orchestration system** of PettingLLMs. It coordinates multi-agent PPO training with reinforcement learning, handling everything from data collection through model updates, checkpoint management, and distributed training across multiple GPUs.

**Total Lines of Code**: ~2,906 lines across 6 files

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                      train.py (Entry Point)                   │
│  - Hydra config loading                                       │
│  - Ray initialization                                         │
│  - Resource pool management                                   │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                 MultiAgentsPPOTrainer                         │
│  - Orchestrates PPO training loop                             │
│  - Manages multiple policy trainers                           │
│  - Handles L1/L2/L3 specialization modes                      │
│  - Coordinates data collection & parameter updates            │
└─────┬──────────────────────────────┬─────────────────────────┘
      │                              │
      ▼                              ▼
┌─────────────────────┐    ┌──────────────────────────────────┐
│  RayPPOTrainer      │    │ MultiAgentsExecutionEngine       │
│  (per model)        │    │  - Agent-environment interaction │
│  - Actor updates    │    │  - Rollout generation           │
│  - Critic updates   │    │  - Environment stepping         │
│  - Ref policy       │    │  - Reward computation           │
└─────────────────────┘    └──────────────────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  async_generate.py   │
                           │  - LLM API calls     │
                           │  - Token processing  │
                           │  - Batch management  │
                           └──────────────────────┘
```

---

## File-by-File Breakdown

### 1. `train.py` (169 lines)

**Purpose**: Main entry point for training. Initializes Ray, loads configs, creates resource pools, and launches training.

#### Key Functions

##### `main(config: DictConfig)`
- **Purpose**: Hydra entry point decorated with `@hydra.main`
- **Responsibilities**:
  - Sets default values for `lora_rank` and `lora_alpha`
  - Converts config to YAML
  - Calls `run_ppo(config)`
- **Configuration Loading**: Uses Hydra to load from `config_path="config"`, `config_name="ppo_trainer"`

##### `run_ppo(config)`
```python
def run_ppo(config):
    # Ray initialization with GPU allocation
    # Creates temporary directories for Ray
    # Initializes distributed training environment
```

**Key Operations**:
1. **Ray Initialization**:
   ```python
   ray.init(
       num_gpus=n_gpus_per_node,
       runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true"}},
       _temp_dir=ray_tmp_dir,
       _system_config=system_config
   )
   ```

2. **GPU Validation**: Checks `CUDA_VISIBLE_DEVICES` and adjusts GPU count
3. **Temp Directory Management**: Creates `/tmp/verl_ray_{pid}` and `/tmp/verl_spill_{pid}`
4. **Remote Execution**: Wraps `train_multi_agents()` as Ray remote function

##### `train_multi_agents(config)`
```python
def train_multi_agents(config):
    # Multi-model setup
    # Tokenizer initialization
    # Resource pool creation
    # Trainer initialization
    # Training loop execution
```

**Key Workflow**:
```
1. Parse config → Extract model configs
2. Load tokenizers → One per model (from HuggingFace)
3. Create resource pools → GPU allocation per model
4. Initialize MultiAgentsPPOTrainer
5. Init workers → Distributed actor/critic/ref workers
6. Init execution engine → Agent-environment interface
7. Start training → trainer.fit()
```

**Resource Pool Management**:
```python
# For L3 with 2 models on 2 GPUs:
n_gpus_per_model = 2 // 2 = 1  # Each model gets 1 GPU

resource_pool_spec = {
    "global_pool_model_0": [1],  # 1 GPU for model_0
    "global_pool_model_1": [1],  # 1 GPU for model_1
}
```

---

### 2. `multi_agents_ppo_trainer.py` (1,007 lines)

**Purpose**: Main training orchestrator. Manages the complete PPO training loop with multi-agent support.

#### Class: `MultiAgentsPPOTrainer`

##### Constructor `__init__(config, tokenizer_dict, role_worker_mapping, resource_pool_manager, ...)`

**Initialization Flow**:
```python
def __init__(...):
    # 1. Set lora defaults
    # 2. Detect specialization mode (L1/L2/L3)
    # 3. Build agent_policy_mapping
    # 4. Create PPO trainers (one per model)
    # 5. Setup LoRA differ mode if needed
```

**Agent-Policy Mapping**:
```python
# From config.agent_policy_configs.agent_configs
for agent_key, agent_config in config.agent_policy_configs.agent_configs.items():
    agent_name = agent_config.name  # e.g., "reasoning_generator"
    policy_name = agent_config.policy_name  # e.g., "reasoning_generator_model"
    self.agent_policy_mapping[agent_name] = policy_name
```

**LoRA Differ Mode Detection**:
```python
if config.specialization == "lora" and len(config.models) == 1:
    self.lora_differ_mode = True
    # Create separate LoRA adapters for each agent
    for agent_idx, agent_name in enumerate(self.agent_policy_mapping.keys()):
        lora_id = f"agent_{agent_name}_lora_{agent_idx}"
        self.agent_lora_mapping[agent_name] = lora_id
```

##### Method: `init_workers()`

**Purpose**: Initialize Ray distributed workers for actor, critic, and reference policy.

**Worker Types**:
- **Actor**: Performs forward passes and gradient updates
- **Critic**: Value function estimation (if enabled)
- **Reference Policy**: KL divergence computation for PPO

**Parallelization**:
```python
# For single model or LoRA mode: Sequential
# For multi-model (L3): Sequential (one model at a time)
for idx, (model_name, trainer) in enumerate(self.ppo_trainer_dict.items()):
    if self.lora_differ_mode:
        trainer.init_workers(lora_num=self.lora_num, agent_lora_mapping=self.agent_lora_mapping)
    else:
        trainer.init_workers(lora_num=1)
```

##### Method: `init_multi_agent_sys_execution_engine()`

**Purpose**: Creates the `MultiAgentsExecutionEngine` which handles agent-environment interactions.

```python
def init_multi_agent_sys_execution_engine(self):
    # Extract rollout engines from each PPO trainer
    for model_name, trainer in self.ppo_trainer_dict.items():
        self.rollout_engine_dict[model_name] = trainer.async_rollout_manager
        self.server_address_dict[model_name] = rollout_engine.server_addresses
    
    # Create execution engine
    self.agent_execution_engine = MultiAgentsExecutionEngine(
        config=self.config,
        ppo_trainer_config_dict=self.ppo_trainer_config_dict,
        tokenizer_dict=self.tokenizer_dict,
        server_address_dict=self.server_address_dict,
        agent_policy_mapping=self.agent_policy_mapping,
        lora_differ_mode=self.lora_differ_mode,
        agent_lora_mapping=self.agent_lora_mapping,
    )
```

##### Method: `fit()` - Main Training Loop

**Training Loop Structure**:
```python
def fit(self):
    # Initialize logger (wandb, console)
    logger = self._initialize_logger_safely()
    
    # Progress bar
    progress_bar = tqdm(range(self.total_training_steps))
    
    for step in progress_bar:
        # 1. Initial validation (step 0 only)
        if step == 0:
            val_metrics = self._validate()
        
        # 2. Collect trajectories
        with simple_timer("collect_trajectory"):
            self.agent_execution_engine.init_agents_and_envs(mode="train")
            
            # Wake up rollout engines
            for rollout_engine in self.rollout_engine_dict.values():
                rollout_engine.wake_up()
            
            # Generate rollouts (async)
            gen_batch_output_per_policy = asyncio.run(
                self.agent_execution_engine.generate_multiple_rollouts_concurrent(...)
            )
            
            # Sleep rollout engines
            for rollout_engine in self.rollout_engine_dict.values():
                rollout_engine.sleep()
        
        # 3. Update parameters
        with simple_timer("update_parameters"):
            for model_name, trainer in self.ppo_trainer_dict.items():
                # Apply filtering
                batch = self._assign_consistent_uids(batch, filter_ratio, filter_method)
                
                # Update model parameters
                updated_batch = self._update_parameters(batch, trainer, timing_raw)
        
        # 4. Save checkpoint
        if step % save_freq == 0:
            for model_name, trainer in self.ppo_trainer_dict.items():
                trainer._save_checkpoint()
        
        # 5. Validation
        if step % val_freq == 0:
            val_metrics = self._validate()
        
        # 6. Log metrics
        logger.log(data=metrics, step=self.global_steps)
```

##### Method: `_update_parameters(batch, ppo_trainer, timing_raw)`

**Purpose**: Core PPO update logic. Computes advantages, updates actor and critic.

**PPO Update Flow**:
```
1. Prepare batch
   ├─ Pad prompts (left padding)
   ├─ Pad responses (right padding)
   ├─ Create attention masks
   └─ Compute position IDs

2. Compute log probabilities
   ├─ Old log probs (from current policy)
   └─ Ref log probs (from reference policy, for KL)

3. Compute values (if critic enabled)
   └─ Value function V(s)

4. Compute advantages
   ├─ Use GAE (Generalized Advantage Estimation)
   ├─ Or use GRPO (if adv_estimator="grpo")
   └─ Normalize advantages

5. Update critic (if enabled)
   └─ Minimize value loss

6. Update actor
   ├─ PPO clipped loss
   ├─ KL penalty
   └─ Entropy bonus (optional)
```

**Token-Level Rewards**:
```python
# Place reward at the last valid token position
reward_tensor = torch.zeros_like(responses_batch, dtype=torch.float32)
for i, reward_val in enumerate(rewards):
    last_valid_pos = valid_token_counts[i] - 1
    reward_tensor[i, last_valid_pos] = reward_val
```

**LoRA Differ Mode Handling**:
```python
if self.lora_differ_mode:
    # Split batch by agent
    for agent_name in unique_agents:
        agent_indices = [i for i, name in enumerate(agent_names) if name == agent_name]
        agent_batch = batch.select_idxs(agent_indices)
        
        # Update each agent's LoRA adapter separately
        agent_output = ppo_trainer.actor_rollout_wg.update_actor(agent_batch)
else:
    # Standard update (all agents together)
    actor_output = ppo_trainer.actor_rollout_wg.update_actor(batch)
```

##### Method: `_validate()`

**Purpose**: Run validation rollouts and compute success metrics.

```python
def _validate(self):
    # Init validation mode
    self.agent_execution_engine.init_agents_and_envs(mode="validate")
    
    # Generate validation rollouts
    gen_batch_output = asyncio.run(
        self.agent_execution_engine.generate_multiple_rollouts_concurrent(...)
    )
    
    # Compute success rates per agent
    for agent_name in self.agent_execution_engine.turn_order:
        success_rollout_num = len(
            self.agent_execution_engine.success_rollout_idx_list_dict[agent_name]
        )
        success_rate = success_rollout_num / total_rollout_num
        
        validation_metrics[f"validation/agent_{agent_name}/success_rate"] = success_rate
    
    return validation_metrics
```

##### Method: `_assign_consistent_uids(data_proto, filter_ratio, mode, sample_num)`

**Purpose**: Assign UIDs to trajectories and optionally filter low-quality samples.

**UID Assignment**:
```python
# Create unique ID per (env_idx, turn_idx, agent_idx) tuple
key = (rollout_indices[i] // sample_num, turn_indices[i], agent_indices[i])
if key not in uid_mapping:
    uid_mapping[key] = str(uuid.uuid4())
uid = uid_mapping[key]
```

**Filtering Modes**:

1. **`mode="mean"`**: Remove UIDs with lowest mean reward
   ```python
   uid_means = {uid: np.mean(rewards_in_group) for uid, rewards_in_group in ...}
   num_to_remove = int(total_uids * filter_ratio)
   uids_to_remove = sorted_by_mean[:num_to_remove]
   ```

2. **`mode="std"`**: Remove UIDs with lowest variance (least diverse)
   ```python
   variance = np.var(rewards_in_group, ddof=0) / (range ** 2)
   uids_to_remove = sorted_by_variance[:num_to_remove]
   ```

3. **`mode="dapo"`**: Remove UIDs with zero variance (no diversity)
   ```python
   if range_normalized_variance(rewards) == 0:
       remove_uid
   ```

4. **`mode="uid"`**: Filter within each UID (remove outliers)
   ```python
   group_mean = np.mean(rewards_in_group)
   samples_with_deviation = [(idx, abs(reward - group_mean)) for ...]
   remove_top_k_by_deviation
   ```

---

### 3. `multi_agents_execution_engine.py` (856 lines)

**Purpose**: Handles multi-agent environment interactions, rollout generation, and reward computation.

#### Class: `MultiAgentsExecutionEngine`

##### Constructor

```python
def __init__(
    self,
    config,
    ppo_trainer_config_dict=None,
    tokenizer_dict=None,
    processor_dict=None,
    server_address_dict=None,
    agent_policy_mapping=None,
    max_workers=1000,  # Default, typically overridden by config
    lora_differ_mode=False,
    agent_lora_mapping=None,
):
```

**Initialization Steps**:
```
1. Load environment and agent classes from registry
2. Setup turn order and agent configs
3. Create Ray docker workers (for code execution)
4. Initialize logging system
5. Setup agent-policy routing
```

**Key Attributes**:
- `env_name`: Environment type (e.g., "math_env", "code_env")
- `turn_order`: List of agent names in execution order
- `agent_config_dict`: Agent-specific configs
- `num_workers`: Number of Ray docker workers (default 1800, should be ~150-300)
- `env_workers`: Ray actors for isolated code execution

##### Method: `init_agents_and_envs(mode="train", step_idx=0)`

**Purpose**: Initialize environments and agents for a rollout.

```python
def init_agents_and_envs(mode="train", step_idx=0):
    # Set mode parameters
    if mode == "validate":
        self.sample_num = config.training.validate_sample_num
        self.gen_batch_size = 1
    else:  # train
        self.sample_num = config.training.train_sample_num
        self.gen_batch_size = config.training.train_batch_size
    
    # Create environment batch
    epoch_size = config.training.epoch_size
    self.env_idx_list = list(range(epoch_size))
    self.env_batch = ENV_BATCH_CLASS[env_name](
        env_idx_list=self.env_idx_list,
        rollout_idx=list(range(len(self.env_idx_list))),
        max_turns=config.env.max_turns,
        config=config
    )
    
    # Initialize agents
    for agent_name in self.turn_order:
        agent_class = AGENT_CLASS_MAPPING[agent_name]
        self.agent_dict[agent_name] = agent_class()
```

##### Method: `generate_multiple_rollouts_concurrent(env_idx_list, rollout_mode="no_tree")`

**Purpose**: Main async function to generate rollouts for all environments concurrently.

**Async Execution Flow**:
```python
async def generate_multiple_rollouts_concurrent(env_idx_list):
    # Create tasks for each environment
    tasks = []
    for env_idx in env_idx_list:
        task = asyncio.create_task(
            self._generate_single_rollout_with_semaphore(env_idx)
        )
        tasks.append(task)
    
    # Execute all rollouts concurrently
    results = await asyncio.gather(*tasks)
    
    # Aggregate results by policy
    batch_per_policy = {}
    for result in results:
        for policy_name, data in result.items():
            if policy_name not in batch_per_policy:
                batch_per_policy[policy_name] = []
            batch_per_policy[policy_name].append(data)
    
    # Convert to DataProto
    for policy_name, data_list in batch_per_policy.items():
        batch_per_policy[policy_name] = DataProto.from_list(data_list)
    
    return batch_per_policy
```

**Semaphore Control**:
```python
# Limit concurrent environment executions
semaphore = asyncio.Semaphore(max_concurrent_envs)

async def _generate_single_rollout_with_semaphore(env_idx):
    async with semaphore:
        return await self._generate_single_rollout(env_idx)
```

##### Method: `_generate_single_rollout(env_idx)` (Async)

**Purpose**: Generate a complete rollout for one environment (multi-turn interaction).

**Rollout Loop**:
```python
async def _generate_single_rollout(env_idx):
    env = self.env_batch.env_list[env_idx]
    rollout_data = []
    
    for turn_idx in range(max_turns):
        for agent_idx, agent_name in enumerate(self.turn_order):
            # 1. Get agent's policy
            policy_name = self.agent_policy_mapping[agent_name]
            
            # 2. Get server address for this policy
            server_address = self.server_address_dict[policy_name][selected_idx]
            
            # 3. Agent updates prompt from environment
            env = agent.update_from_env(turn_idx, env)
            
            # 4. Generate response via LLM
            response = await llm_async_generate(
                prompts=agent.current_prompt,
                address=server_address,
                tokenizer=tokenizer,
                mode=mode,
                temperature=temperature,
                ...
            )
            
            # 5. Agent processes model response
            env = agent.update_from_model(turn_idx, env, response)
            
            # 6. Environment steps with agent's action
            next_obs, reward, done, truncated, info = env.step(agent.current_action)
            
            # 7. Store trajectory data
            rollout_data.append({
                "env_idx": env_idx,
                "turn_idx": turn_idx,
                "agent_idx": agent_idx,
                "agent_name": agent_name,
                "prompt": agent.current_prompt,
                "response": response,
                "action": agent.current_action,
                "reward": reward,
                "done": done,
            })
            
            # 8. Check termination
            if done or truncated:
                break
        
        if done or truncated:
            break
    
    # Group rollout data by policy
    data_by_policy = {}
    for data in rollout_data:
        policy_name = self.agent_policy_mapping[data["agent_name"]]
        if policy_name not in data_by_policy:
            data_by_policy[policy_name] = []
        data_by_policy[policy_name].append(data)
    
    return data_by_policy
```

**LoRA Mode Routing**:
```python
if self.lora_differ_mode:
    # Include LoRA adapter ID in request
    lora_id = self.agent_lora_mapping[agent_name]
    response = await llm_async_generate(
        ...,
        lora_adapter_id=lora_id  # Server loads this LoRA
    )
```

---

### 4. `async_generate.py` (802 lines)

**Purpose**: Async LLM API interaction, token processing, and batch management.

#### Key Functions

##### `llm_async_generate(...)`

**Purpose**: Async function to generate responses from LLM server.

```python
async def llm_async_generate(
    prompts: Dict[str, Any],  # {"text": str, "image": Optional}
    address: str,  # Server address (e.g., "10.1.1.92:44589")
    tokenizer,
    processor=None,
    mode: str = "train",  # "train" or "validate"
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    max_new_tokens: int = 512,
    sample_num: int = 1,  # Number of responses to sample
    enable_thinking: bool = False,
    rollout_idx: int = 0,
    turn_idx: int = 0,
    agent_idx: int = 0,
    lora_adapter_id: Optional[str] = None,
    ...
) -> str:
```

**Execution Flow**:
```
1. Convert prompt to DataProto
   └─ Tokenize text, process images (if multimodal)

2. Create API request payload
   ├─ Format as OpenAI-compatible completion request
   └─ Include LoRA adapter ID if specified

3. Submit to LLM server (async)
   ├─ Use aiohttp for non-blocking I/O
   ├─ Retry on failure (max 3 attempts)
   └─ Handle timeouts

4. Parse response
   ├─ Extract generated tokens
   └─ Decode to string

5. Post-process
   ├─ Apply thinking delimiters if enabled
   └─ Return final response string
```

**API Request Format**:
```python
payload = {
    "model": model_name,
    "prompt": prompt_text,  # Tokenized prompt
    "max_tokens": max_new_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "n": sample_num,  # Number of completions
    "stream": False,
    "logprobs": 1,  # Return log probabilities
}

if lora_adapter_id:
    payload["adapter_id"] = lora_adapter_id
```

**Timeout Handling**:
```python
try:
    async with asyncio.timeout(timeout_seconds):
        response = await submit_completions(...)
except asyncio.TimeoutError:
    print(f"[TIMEOUT] Request timed out after {timeout_seconds}s")
    # Retry with exponential backoff
```

##### `submit_completions(...)`

**Purpose**: Submit completion request with retry logic.

```python
async def submit_completions(
    address: str,
    model: str,
    prompt: str,
    max_retries: int = 3,
    initial_retry_delay: float = 1.0,
    timeout: Optional[float] = None,
    **kwargs
):
    for attempt in range(max_retries):
        try:
            result = await poll_completions_openai(
                address=address,
                model=model,
                prompt=prompt,
                timeout=timeout,
                **kwargs
            )
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Exponential backoff
            retry_delay = initial_retry_delay * (2 ** attempt)
            await asyncio.sleep(retry_delay)
```

##### `poll_completions_openai(...)`

**Purpose**: Make actual HTTP POST request to LLM server.

```python
async def poll_completions_openai(
    address: str,
    model: str,
    prompt: str,
    timeout: Optional[float] = None,
    **kwargs
):
    url = f"http://{address}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        **kwargs
    }
    
    async with get_shared_session() as session:
        async with session.post(url, json=payload, timeout=timeout) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                raise Exception(f"API error: {error_text}")
```

**Connection Pooling**:
```python
# Global aiohttp session (reused across requests)
async def get_shared_session():
    global _shared_session
    
    if _shared_session is None or _shared_session.closed:
        connector = aiohttp.TCPConnector(
            limit=200,  # Max total connections
            limit_per_host=100,  # Max per host
            ttl_dns_cache=300,
        )
        
        timeout = aiohttp.ClientTimeout(
            total=300,  # 5 minutes total
            connect=30,  # 30s to establish connection
        )
        
        _shared_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
    
    return _shared_session
```

##### `convert_prompt_to_dpr(...)`

**Purpose**: Convert prompt dict to `DataProto` format for processing.

```python
def convert_prompt_to_dpr(
    tokenizer,
    processor,
    prompts: Dict[str, Any],  # {"text": str, "image": Optional}
    max_prompt_length: int,
    multi_modal: bool = False,
) -> DataProto:
    # Tokenize text
    if multi_modal and prompts.get("image") is not None:
        # Use processor for multimodal
        inputs = processor(
            text=prompts["text"],
            images=prompts["image"],
            return_tensors="pt"
        )
    else:
        # Text-only tokenization
        inputs = tokenizer(
            prompts["text"],
            return_tensors="pt",
            max_length=max_prompt_length,
            truncation=True
        )
    
    # Create DataProto
    batch = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    
    if "position_ids" not in inputs:
        # Compute position IDs from attention mask
        position_ids = (torch.cumsum(inputs["attention_mask"], dim=1) - 1) * inputs["attention_mask"]
        batch["position_ids"] = position_ids
    
    return DataProto.from_dict({"batch": batch})
```

##### `postprocess_batch(...)`

**Purpose**: Post-process generated responses into DataProto batch.

```python
def postprocess_batch(
    batch: DataProto,
    response_ids: List[List[int]],  # Generated token IDs
    n: int,  # Number of samples
    pad_token_id: int,
    eos_token_id: int,
    max_response_length: int,
    max_prompt_length: int
) -> DataProto:
    # Flatten response_ids
    response = []
    for r_ids in response_ids:
        for r in r_ids:
            if r is None or len(r) == 0:
                response.append([eos_token_id])  # Fallback
            elif len(r) > max_response_length:
                response.append(r[:max_response_length])  # Truncate
            else:
                response.append(r)
    
    # Pad prompts (left padding)
    prompts_batch = pad_sequence(prompts, left_pad=True, padding_value=pad_token_id)
    
    # Pad responses (right padding)
    responses_batch = pad_sequence(response, left_pad=False, padding_value=pad_token_id)
    
    # Concatenate to get full sequence
    input_ids_batch = torch.cat([prompts_batch, responses_batch], dim=1)
    
    # Create attention mask (1 for non-padding, 0 for padding)
    attention_mask_batch = (input_ids_batch != pad_token_id).long()
    
    # Compute position IDs
    position_ids = (torch.cumsum(attention_mask_batch, dim=1) - 1) * attention_mask_batch
    
    # Update batch
    batch.batch.update({
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "position_ids": position_ids,
        "prompts": prompts_batch,
        "responses": responses_batch,
    })
    
    return batch
```

---

### 5. `multiagentssys_register.py` (72 lines)

**Purpose**: Registry mapping environment and agent names to their implementation classes.

#### Mappings

**`ENV_CLASS_MAPPING`**:
```python
{
    "math_env": MathEnv,
    "code_env": CodeEnv,
    "search_env": SearchEnv,
    "stateful_env": StatefulEnv,
    "base_env": BaseEnv,
}
```

**`ENV_BATCH_CLASS_MAPPING`**:
```python
{
    "math_env": MathEnvBatch,
    "code_env": CodeEnvBatch,
    "search_env": SearchEnvBatch,
    "stateful_env": StatefulEnvBatch,
}
```

**`AGENT_CLASS_MAPPING`**:
```python
{
    # Math agents
    "reasoning_generator": ReasoningAgent,
    "tool_generator": ToolAgent,
    
    # Code agents
    "code_generator": CodeGenerationAgent,
    "test_generator": UnitTestGenerationAgent,
    
    # Search agents
    "search_reasoning_agent": ReasoningAgent,
    "web_search_agent": WebSearchAgent,
    
    # Stateful agents
    "plan_agent": PlanAgent,
    "tool_call_agent": ToolAgent,
}
```

**`ENV_WORKER_CLASS_MAPPING`**:
```python
{
    "code_env": get_ray_docker_worker_cls,  # For code execution
    "math_env": get_ray_docker_worker_cls,  # For math verification
    "search_env": get_ray_docker_worker_cls,
}
```

**Usage in Code**:
```python
# Get environment class dynamically
env_name = config.env.name  # e.g., "math_env"
ENV_CLASS = ENV_CLASS_MAPPING[env_name]  # MathEnv class
env = ENV_CLASS(env_idx, max_turns, config)

# Get agent class dynamically
agent_name = "reasoning_generator"
AGENT_CLASS = AGENT_CLASS_MAPPING[agent_name]  # ReasoningAgent class
agent = AGENT_CLASS()
```

---

## Data Flow

### Complete Training Step

```
┌────────────────────────────────────────────────────────────────┐
│ 1. fit() calls collect_trajectory                              │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. init_agents_and_envs(mode="train")                          │
│    - Create env_batch (epoch_size environments)                │
│    - Initialize all agents                                     │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. generate_multiple_rollouts_concurrent(env_idx_list)         │
│    - Create async task per environment                         │
│    - asyncio.gather() runs all concurrently                    │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. For each env: _generate_single_rollout(env_idx)             │
│    ┌──────────────────────────────────────────────────────┐   │
│    │ For turn in range(max_turns):                        │   │
│    │   For agent in turn_order:                           │   │
│    │     1. agent.update_from_env(env) → get prompt       │   │
│    │     2. llm_async_generate() → get response          │   │
│    │     3. agent.update_from_model(response) → action    │   │
│    │     4. env.step(action) → next_obs, reward, done     │   │
│    │     5. Store trajectory data                         │   │
│    │     6. Check done/truncated                          │   │
│    └──────────────────────────────────────────────────────┘   │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 5. Aggregate results by policy_name                            │
│    batch_per_policy[policy_name] = DataProto(trajectories)     │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 6. _assign_consistent_uids(batch, filter_ratio, filter_method) │
│    - Assign UIDs to group samples                              │
│    - Filter low-quality samples                                │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 7. _update_parameters(batch, ppo_trainer, timing_raw)          │
│    ┌──────────────────────────────────────────────────────┐   │
│    │ 1. Prepare batch (pad, create masks)                 │   │
│    │ 2. Compute old_log_prob (current policy)             │   │
│    │ 3. Compute ref_log_prob (reference policy)           │   │
│    │ 4. Compute values (critic, if enabled)               │   │
│    │ 5. Compute advantages (GAE or GRPO)                  │   │
│    │ 6. Update critic (if enabled)                        │   │
│    │ 7. Update actor (PPO loss)                           │   │
│    └──────────────────────────────────────────────────────┘   │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│ 8. Compute metrics and log                                     │
│    - Data metrics (rewards, advantages)                        │
│    - Timing metrics (rollout time, update time)                │
│    - Log to wandb/console                                      │
└────────────────────────────────────────────────────────────────┘
```

---

## Specialization Modes (L0/L1/L2/L3)

### L0: Single Agent
```python
agent_policy_configs:
  num_agents: 1
  turn_order: ["reasoning_generator"]

models:
  model_0:
    name: "shared_model"
```

### L1: Prompt Specialization
```python
specialization: "prompt"

agent_policy_configs:
  agent_0:
    name: "reasoning_generator"
    policy_name: "shared_model"  # Same model
  agent_1:
    name: "tool_generator"
    policy_name: "shared_model"  # Same model

models:
  model_0:  # Only one model
    name: "shared_model"
```

### L2: LoRA Specialization
```python
specialization: "lora"
lora_rank: 16
lora_alpha: 32

agent_policy_configs:
  agent_0:
    policy_name: "shared_model"  # Same base model
  agent_1:
    policy_name: "shared_model"  # Same base model

models:
  model_0:  # Only one base model
    ppo_trainer_config:
      actor_rollout_ref:
        model:
          lora_rank: 16  # Enable LoRA

# System automatically creates separate LoRA adapters per agent
```

### L3: Full Model Specialization
```python
specialization: "full"

agent_policy_configs:
  agent_0:
    policy_name: "reasoning_generator_model"  # Different
  agent_1:
    policy_name: "tool_generator_model"  # Different

models:
  model_0:  # First independent model
    name: "reasoning_generator_model"
  model_1:  # Second independent model
    name: "tool_generator_model"
```

---

## Important APIs

### Public APIs

#### `MultiAgentsPPOTrainer`
```python
trainer = MultiAgentsPPOTrainer(
    config=config,
    tokenizer_dict=tokenizer_dict,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
)

# Initialize distributed workers
trainer.init_workers()

# Initialize agent-environment interface
trainer.init_multi_agent_sys_execution_engine()

# Start training
trainer.fit()
```

#### `MultiAgentsExecutionEngine`
```python
engine = MultiAgentsExecutionEngine(
    config=config,
    tokenizer_dict=tokenizer_dict,
    server_address_dict=server_address_dict,
    agent_policy_mapping=agent_policy_mapping,
    lora_differ_mode=False,
)

# Initialize environments and agents
engine.init_agents_and_envs(mode="train", step_idx=0)

# Generate rollouts (async)
batch_per_policy = await engine.generate_multiple_rollouts_concurrent(env_idx_list)
```

#### `llm_async_generate`
```python
response = await llm_async_generate(
    prompts={"text": prompt_text, "image": None},
    address="10.1.1.92:44589",
    tokenizer=tokenizer,
    mode="train",
    temperature=1.0,
    max_new_tokens=512,
    sample_num=1,
)
```

---

## Configuration Parameters

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_training_steps` | 200 | Total number of training iterations |
| `train_batch_size` | 4-32 | Number of environments per step |
| `train_sample_num` | 2-8 | Responses sampled per prompt |
| `validate_sample_num` | 2-5 | Samples for validation |
| `epoch_size` | 20 | Environments per epoch |
| `val_freq` | 10 | Validation every N steps |
| `save_freq` | 40 | Save checkpoint every N steps |
| `num_workers` | 150-300 | Ray docker workers (default 1800 is too high!) |
| `max_prompt_length` | 512-4096 | Max prompt tokens |
| `max_response_length` | 512-2048 | Max response tokens |

### Environment Parameters

| Parameter | Description |
|-----------|-------------|
| `env.name` | Environment type: "math_env", "code_env", etc. |
| `env.dataset` | Dataset name: "polaris", "apps", etc. |
| `env.benchmark` | Benchmark: "AIME24", "LiveCodeBench", etc. |
| `env.max_turns` | Maximum turns per episode |

---

## Performance Tips

1. **Reduce `num_workers`**: Default 1800 is way too high. Use 150-300.
2. **Batch size tuning**: Larger batch = faster training, more memory
3. **Sample num tuning**: More samples = better gradients, slower rollouts
4. **Async rollouts**: All environments run concurrently for speed
5. **Connection pooling**: Reuse aiohttp sessions for API calls
6. **LoRA differ mode**: Train separate LoRA adapters without loading multiple full models

---

## Error Handling

### Common Errors

1. **Out of Memory (OOM)**:
   - Reduce `train_batch_size`
   - Reduce `max_prompt_length` or `max_response_length`
   - Enable gradient checkpointing

2. **Ray Worker Explosion**:
   - Reduce `training.num_workers` (150-300 recommended)

3. **API Timeout**:
   - Increase timeout in `llm_async_generate`
   - Check server is running and accessible

4. **LoRA Config Errors**:
   - Ensure `lora_rank > 0` when `specialization="lora"`
   - Check LoRA adapters are properly saved/loaded

---

This concludes the comprehensive Trainer module documentation!

