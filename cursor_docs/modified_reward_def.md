# Modified Reward Function: Cooperative Credit Assignment

## Problem Statement

**Current Behavior**: Each agent receives rewards independently based on their own correctness:
- **Tool Agent** (code generator): Reward = 1 if code answer is correct, 0 otherwise
- **Reasoning Agent**: Reward = 1 if reasoning answer is correct, 0 otherwise

**Desired Behavior**: Cooperative reward structure where both agents share the final outcome:
- **Reasoning Agent**: Reward = 1 if final answer is correct, 0 otherwise
- **Tool Agent**: Reward = Same as reasoning agent (credit for enabling the final answer)

**Rationale**: Since the reasoning agent conditions on the tool agent's output, the tool agent should be credited/penalized based on how useful its code execution was for the final answer.

---

## Current Implementation Analysis

### Turn Order
```yaml
multi_agent_interaction:
  turn_order: ["tool_generator", "reasoning_generator"]
```

**Execution Flow**:
```
Turn 0:
  1. Tool Agent → Generates code → Executes code → Gets answer X
     - Sets self.agent_reward = 1.0 (if code correct) or 0.0 (if incorrect)
  
  2. Reasoning Agent → Sees code result X → Generates reasoning → Gets answer Y
     - Sets self.agent_reward = 1 (if reasoning correct) or 0 (if incorrect)

Both rewards are stored independently in trajectories.
```

### Current Reward Calculation Location

**File**: `pettingllms/trainer/multi_agents_execution_engine.py`

```python
# Line 326-346: After all agents in a turn complete
for agent_output in agent_outputs:
    agent_name = agent_output['agent_name']
    current_agent = agent_output['current_agent']
    
    # Calculate reward using agent's calculate_reward method
    if hasattr(current_agent, 'calculate_reward'):
        current_agent.calculate_reward(env)  # Each agent calculates own reward
    
    # Store reward in trajectory
    if output_dpr is not None:
        output_dpr.non_tensor_batch["reward"] = np.array([current_agent.agent_reward])
```

### Current Agent Implementations

**File**: `pettingllms/multi_agent_env/math/agents/reasoning_agent.py`
```python
def calculate_reward(self, env_data: Env):
    self.agent_reward = int(env_data.state.reasoning_is_correct)
    self.reward_history.append(self.agent_reward)
```

**File**: `pettingllms/multi_agent_env/math/agents/tool_agent.py`
```python
async def step(self, env_data: Env, env_worker=None):
    # ... code execution ...
    if is_correct:
        self.agent_reward = 1.0
    else:
        self.agent_reward = 0.0

def calculate_reward(self, env_data: Env):
    self.agent_reward = self.agent_reward  # Keep existing reward
    self.reward_history.append(self.agent_reward)
```

---

## Solution Approaches

### **Approach 1: Shared Reward in Environment State (RECOMMENDED)**

#### Concept
Store a "final_reward" in the environment state that both agents can access. The tool agent gets the reasoning agent's reward retroactively.

#### Implementation

##### Step 1: Modify Environment State

**File**: `pettingllms/multi_agent_env/math/math_env.py`

```python
@dataclass
class MathEnvState:
    problem: str = None
    ground_truth_answer: str = None
    
    # Existing fields...
    reasoning_is_correct: bool = None
    code_is_correct: bool = False
    
    # NEW: Add shared reward field
    final_reward: float = None  # Set by reasoning agent, used by tool agent
    reward_mode: str = "independent"  # "independent" or "shared"
```

##### Step 2: Modify Reasoning Agent

**File**: `pettingllms/multi_agent_env/math/agents/reasoning_agent.py`

```python
async def step(self, env_data: Env, env_worker=None):
    # ... existing code ...
    
    # Verify correctness
    is_correct = verify(extracted_answer, parse(ground_truth))
    env_data.state.reasoning_is_correct = is_correct
    
    # NEW: Set final reward for both agents to share
    if hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared":
        env_data.state.final_reward = 1.0 if is_correct else 0.0
    
    if is_correct:
        self.success = True
        env_data.done = True

def calculate_reward(self, env_data: Env):
    # Check if using shared reward mode
    if hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared":
        # Use final reward from state
        self.agent_reward = env_data.state.final_reward if env_data.state.final_reward is not None else 0.0
    else:
        # Independent reward (original behavior)
        self.agent_reward = int(env_data.state.reasoning_is_correct)
    
    self.reward_history.append(self.agent_reward)
```

##### Step 3: Modify Tool Agent

**File**: `pettingllms/multi_agent_env/math/agents/tool_agent.py`

```python
async def step(self, env_data: Env, env_worker=None):
    # ... existing code execution ...
    
    # Check correctness of code output
    is_correct = verify(parse(code_execution_output), parse(ground_truth_answer))
    
    if is_correct:
        self.success = True
        env_data.state.code_is_correct = True
        # DON'T set self.agent_reward here if using shared mode
    else:
        env_data.state.code_is_correct = False

def calculate_reward(self, env_data: Env):
    # Check if using shared reward mode
    if hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared":
        # Use final reward from reasoning agent
        if env_data.state.final_reward is not None:
            self.agent_reward = env_data.state.final_reward
        else:
            # Reasoning agent hasn't set reward yet (shouldn't happen in correct turn order)
            self.agent_reward = 0.0
    else:
        # Independent reward (original behavior)
        # Keep the reward already set in step()
        pass
    
    self.reward_history.append(self.agent_reward)
```

##### Step 4: Add Configuration Parameter

**File**: `pettingllms/config/math/math_L1_prompt.yaml` (or your config)

```yaml
env:
  name: math_env
  dataset: "polaris"
  benchmark: "AIME24"
  max_turns: 5
  reward_mode: "shared"  # NEW: "independent" or "shared"
```

##### Step 5: Initialize Reward Mode in Environment

**File**: `pettingllms/multi_agent_env/math/math_env.py`

```python
class MathEnvBatch:
    def __init__(self, env_idx_list, rollout_idx_list, samples, max_turns, config, mode="train"):
        # ... existing code ...
        
        for i, problem in enumerate(self.problem_list):
            state = MathEnvState(
                problem=problem["question"],
                ground_truth_answer=problem["solution"],
            )
            
            # NEW: Set reward mode from config
            if hasattr(config, 'env') and hasattr(config.env, 'reward_mode'):
                state.reward_mode = config.env.reward_mode
            else:
                state.reward_mode = "independent"  # Default
            
            for s in range(samples):
                env = MathEnv(...)
                env.state = copy.deepcopy(state)
                self.env_list.append(env)
```

---

### **Approach 2: Post-Processing Rewards in Execution Engine**

#### Concept
After all agents in a turn complete, redistribute rewards based on the final outcome.

#### Implementation

**File**: `pettingllms/trainer/multi_agents_execution_engine.py`

```python
# After line 326 (after all agents complete their step):
# Line 326-378: Calculate rewards section

for agent_output in agent_outputs:
    agent_idx = agent_output['agent_idx']
    agent_name = agent_output['agent_name']
    current_agent = agent_output['current_agent']
    
    # Calculate reward using agent's calculate_reward method
    if hasattr(current_agent, 'calculate_reward'):
        current_agent.calculate_reward(env)

# NEW: Post-process rewards for shared reward mode
if hasattr(env.state, 'reward_mode') and env.state.reward_mode == "shared":
    # Find the reasoning agent's reward (final outcome)
    reasoning_reward = None
    for agent_output in agent_outputs:
        if agent_output['agent_name'] == 'reasoning_generator':
            reasoning_reward = agent_output['current_agent'].agent_reward
            break
    
    # If reasoning agent acted this turn, share its reward with all agents
    if reasoning_reward is not None:
        for agent_output in agent_outputs:
            agent_output['current_agent'].agent_reward = reasoning_reward
            # Update the last entry in reward history
            if agent_output['current_agent'].reward_history:
                agent_output['current_agent'].reward_history[-1] = reasoning_reward

# Continue with existing code to store rewards in trajectory
for agent_output in agent_outputs:
    # ... existing code ...
    if output_dpr is not None:
        output_dpr.non_tensor_batch["reward"] = np.array([current_agent.agent_reward])
```

**Pros**:
- Centralized logic
- No need to modify agent classes
- Easy to add different reward sharing strategies

**Cons**:
- Less intuitive
- Harder to debug
- Reward logic split between agents and engine

---

### **Approach 3: Delayed Reward Assignment**

#### Concept
Don't assign rewards to tool agent until reasoning agent completes. Store tool agent's trajectory, then update reward retroactively.

#### Implementation

**File**: `pettingllms/trainer/multi_agents_execution_engine.py`

```python
# Modify the reward storage logic around line 345-378

# After all agents complete:
for agent_output in agent_outputs:
    agent_name = agent_output['agent_name']
    current_agent = agent_output['current_agent']
    output_dpr = agent_output['output_dpr']
    policy_name = agent_output['policy_name']
    
    # Calculate reward
    if hasattr(current_agent, 'calculate_reward'):
        current_agent.calculate_reward(env)
    
    # Store reward in trajectory
    if output_dpr is not None:
        # NEW: Check if this is tool agent in shared reward mode
        if (hasattr(env.state, 'reward_mode') and 
            env.state.reward_mode == "shared" and 
            agent_name == 'tool_generator'):
            
            # Mark as pending - will be updated after reasoning agent
            output_dpr.non_tensor_batch["reward_pending"] = np.array([True])
            output_dpr.non_tensor_batch["reward"] = np.array([0.0])  # Placeholder
        else:
            # Normal reward assignment
            output_dpr.non_tensor_batch["reward"] = np.array([current_agent.agent_reward])
        
        output_dpr.non_tensor_batch["agent_name"] = np.array([agent_name], dtype=object)

# NEW: After all agents, update pending rewards
if hasattr(env.state, 'reward_mode') and env.state.reward_mode == "shared":
    # Find reasoning agent's reward
    reasoning_reward = None
    for agent_output in agent_outputs:
        if agent_output['agent_name'] == 'reasoning_generator':
            reasoning_reward = agent_output['current_agent'].agent_reward
            break
    
    # Update tool agent's reward
    if reasoning_reward is not None:
        for agent_output in agent_outputs:
            if agent_output['agent_name'] == 'tool_generator':
                output_dpr = agent_output['output_dpr']
                if output_dpr is not None and "reward_pending" in output_dpr.non_tensor_batch:
                    # Update with reasoning agent's reward
                    output_dpr.non_tensor_batch["reward"] = np.array([reasoning_reward])
                    del output_dpr.non_tensor_batch["reward_pending"]
```

---

## Comparison of Approaches

| Aspect | Approach 1 (State) | Approach 2 (Post-Process) | Approach 3 (Delayed) |
|--------|-------------------|---------------------------|----------------------|
| **Clarity** | ⭐⭐⭐⭐⭐ Most intuitive | ⭐⭐⭐ Somewhat obscure | ⭐⭐ Complex |
| **Modularity** | ⭐⭐⭐⭐⭐ Agent-level | ⭐⭐⭐ Engine-level | ⭐⭐ Engine-level |
| **Debuggability** | ⭐⭐⭐⭐⭐ Easy to trace | ⭐⭐⭐ Moderate | ⭐⭐ Harder |
| **Extensibility** | ⭐⭐⭐⭐⭐ Easy to add modes | ⭐⭐⭐⭐ Flexible | ⭐⭐ Limited |
| **Code Changes** | Moderate (4 files) | Small (1 file) | Small (1 file) |
| **Backward Compat** | ⭐⭐⭐⭐⭐ Fully compatible | ⭐⭐⭐⭐⭐ Fully compatible | ⭐⭐⭐⭐ Mostly |

**Recommendation**: **Approach 1 (Shared Reward in Environment State)**

---

## Complete Implementation Guide (Approach 1)

### Files to Modify

1. ✅ `pettingllms/multi_agent_env/math/math_env.py`
2. ✅ `pettingllms/multi_agent_env/math/agents/reasoning_agent.py`
3. ✅ `pettingllms/multi_agent_env/math/agents/tool_agent.py`
4. ✅ `pettingllms/config/math/math_L1_prompt.yaml` (or your config)

### Step-by-Step Changes

#### Change 1: Add Fields to MathEnvState

**File**: `pettingllms/multi_agent_env/math/math_env.py`

**Location**: Line 14-28 (in `MathEnvState` dataclass)

**Change**:
```python
@dataclass
class MathEnvState:
    problem: str = None
    ground_truth_answer: str = None
    reasoning_generated_solution: str = None
    code_generated_solution: str = None
    reasoning_extracted_answer: str = None
    code_extracted_answer: str = None
    reasoning_is_correct: bool = None
    code_is_correct: bool = False
    code_reasoning_aligned: bool = False
    reasoning_generated_solution_history: List = field(default_factory=list)
    code_generated_solution_history: List = field(default_factory=list)
    reasoning_extracted_answer_history: List = field(default_factory=list)
    code_extracted_answer_history: List = field(default_factory=list)
    
    # NEW FIELDS FOR SHARED REWARD
    final_reward: float = None  # Set by reasoning agent, used by tool agent
    reward_mode: str = "independent"  # "independent" or "shared"
```

#### Change 2: Initialize Reward Mode in EnvBatch

**File**: `pettingllms/multi_agent_env/math/math_env.py`

**Location**: Line 63-86 (in `MathEnvBatch.__init__`)

**Change**:
```python
class MathEnvBatch:
    def __init__(self, env_idx_list, env_indices, rollout_idx_list, samples, max_turns, config, mode="train", *, env_workers=None):
        # ... existing code ...
        
        for i, problem in enumerate(self.problem_list):
            state = MathEnvState(
                problem=problem["question"],
                ground_truth_answer=problem["solution"],
            )
            
            # NEW: Initialize reward mode from config
            if hasattr(config, 'env') and hasattr(config.env, 'reward_mode'):
                state.reward_mode = config.env.reward_mode
            else:
                state.reward_mode = "independent"  # Default to original behavior
            
            for s in range(samples):
                env = MathEnv(env_idx=i, rollout_idx=rollout_idx_list[i*samples+s], max_turns=max_turns, config=None)
                env.state = copy.deepcopy(state)
                self.env_list.append(env)
```

#### Change 3: Modify Reasoning Agent Step Method

**File**: `pettingllms/multi_agent_env/math/agents/reasoning_agent.py`

**Location**: Line 92-126 (in `step` method)

**Change**:
```python
async def step(self, env_data: Env, env_worker: Any = None):
    """
    Process the generated reasoning solution and evaluate it against the ground truth.
    """
    env_data.state.reasoning_generated_solution = truncatefn(self.current_action)
    env_data.state.reasoning_extracted_answer = parse(self.current_action)
    env_data.state.reasoning_generated_solution_history.append(env_data.state.reasoning_generated_solution)
    env_data.state.reasoning_extracted_answer_history.append(env_data.state.reasoning_extracted_answer)
    self.answer_history.append(env_data.state.reasoning_extracted_answer)
    self.action_history.append(self.current_action)
    extracted_answer = env_data.state.reasoning_extracted_answer
    ground_truth_answer = env_data.state.ground_truth_answer
    is_correct = False
    
    if extracted_answer is not None and ground_truth_answer is not None:
        is_correct = verify(extracted_answer, parse(ground_truth_answer))
        env_data.state.reasoning_is_correct = is_correct
        
        # NEW: Set final reward for shared reward mode
        if hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared":
            env_data.state.final_reward = 1.0 if is_correct else 0.0
        
        if is_correct:
            self.success = True
            env_data.state.reasoning_is_correct = True
        else:
            self.success = False
            env_data.state.reasoning_is_correct = False
    
    # ... rest of existing code ...
```

#### Change 4: Modify Reasoning Agent calculate_reward Method

**File**: `pettingllms/multi_agent_env/math/agents/reasoning_agent.py`

**Location**: Line 129-131 (in `calculate_reward` method)

**Change**:
```python
def calculate_reward(self, env_data: Env):
    # Check if using shared reward mode
    if hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared":
        # Use final reward from state (which we just set in step())
        self.agent_reward = env_data.state.final_reward if env_data.state.final_reward is not None else 0.0
    else:
        # Independent reward (original behavior)
        self.agent_reward = int(env_data.state.reasoning_is_correct)
    
    self.reward_history.append(self.agent_reward)
```

#### Change 5: Modify Tool Agent Step Method

**File**: `pettingllms/multi_agent_env/math/agents/tool_agent.py`

**Location**: Line 96-135 (in `step` method)

**Change**:
```python
async def step(self, env_data: Env, env_worker: Any = None):
    """
    Process the generated code solution and evaluate it against the ground truth.
    """
    generated_solution = self.current_action
    env_data.state.code_generated_solution = generated_solution
    ground_truth_answer = env_data.state.ground_truth_answer
    is_correct = False
    code_execution_output = None
    
    try:
        # execute the code (through ray worker)
        code_execution_output = await get_code_execution_output(
            generated_solution,
            timeout=20.0,
            ray_actor=env_worker,
        )
        env_data.state.code_extracted_answer = parse(code_execution_output)
        # Update history records
        env_data.state.code_generated_solution_history.append(env_data.state.code_generated_solution)
        env_data.state.code_extracted_answer_history.append(env_data.state.code_extracted_answer)
        self.answer_history.append(env_data.state.code_extracted_answer)
        self.action_history.append(self.current_action)
        
        if code_execution_output is None:
            # Don't set reward here in shared mode
            if not (hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared"):
                self.agent_reward = -1
            return
        
        is_correct = verify(parse(code_execution_output), parse(ground_truth_answer))
        
        if is_correct:
            self.success = True
            env_data.state.code_is_correct = True
            # Only set reward in independent mode
            if not (hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared"):
                self.agent_reward = 1.0
        else:
            self.success = False
            env_data.state.code_is_correct = False
            # Only set reward in independent mode
            if not (hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared"):
                self.agent_reward = 0.0
                
    except Exception as e:
        code_execution_output = f"error: {e}"
        env_data.state.code_extracted_answer = code_execution_output
        # Only set reward in independent mode
        if not (hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared"):
            self.agent_reward = -1
```

#### Change 6: Modify Tool Agent calculate_reward Method

**File**: `pettingllms/multi_agent_env/math/agents/tool_agent.py`

**Location**: Line 139-141 (in `calculate_reward` method)

**Change**:
```python
def calculate_reward(self, env_data: Env):
    # Check if using shared reward mode
    if hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared":
        # Use final reward from reasoning agent
        if hasattr(env_data.state, 'final_reward') and env_data.state.final_reward is not None:
            self.agent_reward = env_data.state.final_reward
        else:
            # Reasoning agent hasn't set reward yet
            # This shouldn't happen with correct turn order (tool before reasoning)
            # But set to 0 as fallback
            self.agent_reward = 0.0
    else:
        # Independent reward (original behavior)
        # Keep the reward already set in step()
        # self.agent_reward is already set, do nothing
        pass
    
    self.reward_history.append(self.agent_reward)
```

#### Change 7: Add Configuration Parameter

**File**: `pettingllms/config/math/math_L1_prompt.yaml` (or create new config)

**Location**: In the `env:` section

**Change**:
```yaml
env:
  name: math_env
  dataset: "polaris"
  benchmark: "AIME24"
  max_turns: 5
  resolve: false
  multi_modal: false
  batched_init: true
  reward_mode: "shared"  # NEW: Set to "shared" for cooperative rewards
```

---

## Testing the Implementation

### Test Case 1: Both Correct (Shared Mode)

**Setup**:
```yaml
env:
  reward_mode: "shared"
```

**Scenario**:
- Tool agent generates code → Execution result: "42" (correct)
- Reasoning agent sees code result → Final answer: "42" (correct)

**Expected Rewards**:
- Tool agent: `1.0` (shares reasoning success)
- Reasoning agent: `1.0` (correct answer)

### Test Case 2: Tool Wrong, Reasoning Correct (Shared Mode)

**Scenario**:
- Tool agent generates code → Execution result: "40" (wrong)
- Reasoning agent sees code result → Ignores it, finds correct answer: "42" (correct)

**Expected Rewards**:
- Tool agent: `1.0` (shares reasoning success, even though code was wrong)
- Reasoning agent: `1.0` (correct answer)

### Test Case 3: Both Wrong (Shared Mode)

**Scenario**:
- Tool agent generates code → Execution result: "40" (wrong)
- Reasoning agent sees code result → Final answer: "40" (wrong, followed code)

**Expected Rewards**:
- Tool agent: `0.0` (shares reasoning failure)
- Reasoning agent: `0.0` (wrong answer)

### Test Case 4: Independent Mode (Backward Compatibility)

**Setup**:
```yaml
env:
  reward_mode: "independent"  # or omit (defaults to independent)
```

**Scenario**:
- Tool agent generates code → Execution result: "40" (wrong)
- Reasoning agent ignores code → Final answer: "42" (correct)

**Expected Rewards**:
- Tool agent: `0.0` (code was wrong)
- Reasoning agent: `1.0` (reasoning was correct)

---

## Verification Steps

1. **Add Logging**:
```python
# In reasoning_agent.py calculate_reward():
def calculate_reward(self, env_data: Env):
    if hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared":
        self.agent_reward = env_data.state.final_reward if env_data.state.final_reward is not None else 0.0
        print(f"[ReasoningAgent] Shared reward mode: final_reward={env_data.state.final_reward}")
    else:
        self.agent_reward = int(env_data.state.reasoning_is_correct)
        print(f"[ReasoningAgent] Independent mode: reward={self.agent_reward}")
    self.reward_history.append(self.agent_reward)

# In tool_agent.py calculate_reward():
def calculate_reward(self, env_data: Env):
    if hasattr(env_data.state, 'reward_mode') and env_data.state.reward_mode == "shared":
        if hasattr(env_data.state, 'final_reward') and env_data.state.final_reward is not None:
            self.agent_reward = env_data.state.final_reward
            print(f"[ToolAgent] Shared reward mode: received final_reward={env_data.state.final_reward}")
        else:
            self.agent_reward = 0.0
            print(f"[ToolAgent] Shared reward mode: final_reward not set yet, using 0.0")
    else:
        print(f"[ToolAgent] Independent mode: keeping reward={self.agent_reward}")
    self.reward_history.append(self.agent_reward)
```

2. **Check Trajectories**:
```python
# In multi_agents_execution_engine.py, after line 346:
print(f"[Trajectory] Agent: {agent_name}, Reward: {current_agent.agent_reward}")
```

3. **Monitor Logs**:
```bash
# Look for reward values in logs
grep "reward" logs/math_1.7B_L1/*/train/0/0/env_agent.log
```

---

## Considerations & Trade-offs

### Advantages of Shared Reward

1. **Credit Assignment**: Tool agent gets credit when its code helps reasoning agent
2. **Cooperation**: Encourages tool agent to generate useful code, not just correct code
3. **Alignment**: Both agents work toward same goal (final answer correctness)

### Potential Issues

1. **Delayed Feedback**: Tool agent doesn't know immediately if its code was good
2. **Less Informative**: Tool agent loses signal about code correctness
3. **Dependency**: Tool agent's learning depends entirely on reasoning agent's performance

### Mitigation Strategies

#### Option 1: Weighted Combination

Combine both rewards:
```python
# In tool_agent.py calculate_reward():
if env_data.state.reward_mode == "shared":
    code_reward = 1.0 if env_data.state.code_is_correct else 0.0
    final_reward = env_data.state.final_reward if env_data.state.final_reward is not None else 0.0
    
    # Weighted combination: 30% own correctness, 70% final outcome
    self.agent_reward = 0.3 * code_reward + 0.7 * final_reward
```

#### Option 2: Bonus for Code Correctness

```python
# In tool_agent.py calculate_reward():
if env_data.state.reward_mode == "shared":
    base_reward = env_data.state.final_reward if env_data.state.final_reward is not None else 0.0
    
    # Add bonus if code was also correct
    if env_data.state.code_is_correct:
        self.agent_reward = base_reward + 0.5  # Bonus for code correctness
    else:
        self.agent_reward = base_reward
```

#### Option 3: Penalty for Code Errors

```python
# In tool_agent.py calculate_reward():
if env_data.state.reward_mode == "shared":
    final_reward = env_data.state.final_reward if env_data.state.final_reward is not None else 0.0
    
    # Penalize if code execution failed (error = -1)
    if hasattr(self, 'code_execution_failed') and self.code_execution_failed:
        self.agent_reward = final_reward - 0.5
    else:
        self.agent_reward = final_reward
```

---

## Alternative Configurations

### Configuration 1: Asymmetric Rewards

```yaml
env:
  name: math_env
  reward_mode: "asymmetric"  # New mode
  tool_reward_weight: 0.7     # Tool gets 70% of final reward
  reasoning_reward_weight: 1.0 # Reasoning gets full reward
```

### Configuration 2: Turn-Based Decay

```yaml
env:
  reward_mode: "temporal_decay"
  tool_decay_factor: 0.9  # Tool reward decays by 10% each turn
```

---

## Summary

**Recommended Implementation**: Approach 1 (Shared Reward in Environment State)

**Key Changes**:
1. Add `final_reward` and `reward_mode` fields to `MathEnvState`
2. Reasoning agent sets `final_reward` in its `step()` method
3. Both agents check `reward_mode` in `calculate_reward()`
4. Tool agent uses `final_reward` instead of own correctness
5. Add `reward_mode: "shared"` to config file

**Benefits**:
- Clean, modular design
- Backward compatible (defaults to independent)
- Easy to extend with other reward modes
- Intuitive and debuggable

**Testing**: Verify with logging that tool agent receives reasoning agent's reward in shared mode.

---

**Implementation Complexity**: Medium (4 files to modify)  
**Estimated Time**: 1-2 hours  
**Risk Level**: Low (backward compatible, can toggle via config)


