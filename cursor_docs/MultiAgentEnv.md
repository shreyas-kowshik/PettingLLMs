# Multi-Agent Environment Module Documentation

## Overview

The `multi_agent_env/` module contains all environment and agent implementations for different tasks (math, code, search, stateful). This is the **domain logic layer** where task-specific prompting, reward computation, and action parsing happen.

**Structure**: 
- 5 environment domains (math, code, search, stateful, base)
- 10+ agent types
- Shared base classes (`Env`, `Agent`)

---

## Architecture

```
multi_agent_env/
├── base/                 # Abstract base classes
│   ├── env.py           # Base environment interface
│   └── agent.py         # Base agent interface
├── math/                # Mathematical problem solving
│   ├── math_env.py      # Math environment
│   ├── math_utils.py    # Dataset loading, verification
│   ├── math_worker.py   # Ray docker worker for execution
│   └── agents/
│       ├── reasoning_agent.py  # Reasoning agent
│       └── tool_agent.py       # Tool (code) agent
├── code/                # Code generation
│   ├── code_env.py      # Code environment
│   ├── code_utils.py    # Test evaluation
│   ├── code_worker.py   # Ray docker worker
│   └── agents/
│       ├── code_agent.py       # Code generator
│       └── unit_test_agent.py  # Test generator
├── search/              # Web search QA
│   ├── search_env.py    # Search environment
│   ├── search_utils.py  # Web API utils
│   └── agents/
│       ├── reasoning_agent.py     # Question analyzer
│       └── web_search_agent.py    # Search executor
├── stateful/            # Stateful tool usage
│   ├── stateful_env.py  # Stateful environment
│   ├── env_state.py     # State management
│   └── agents/
│       ├── plan_agent.py   # Planning agent
│       └── tool_agent.py   # Tool executor
└── math_aggretion/      # Math with aggregation (experimental)
    └── ...
```

---

## Base Classes

### `base/env.py`

#### Class: `Env`

Abstract base class for all environments.

```python
class Env:
    def __init__(self, env_idx: int, rollout_idx: int, max_turns: int, config: dict):
        self.env_idx = env_idx          # Environment ID
        self.rollout_idx = rollout_idx  # Rollout ID (for sampling)
        self.max_turns = max_turns      # Max interaction turns
        self.config = config
        self.history = []
        self.current_turn = 0
        self.done = False
        self.state = None  # Task-specific state
    
    @abstractmethod
    def step(self, action):
        """
        Take a step based on agent's action.
        
        Returns:
            next_observation, reward, terminated, truncated, info
        """
        pass
```

**Key Attributes**:
- `env_idx`: Which environment (problem) this is
- `rollout_idx`: Which sample/rollout of this environment
- `max_turns`: Maximum agent-env interaction steps
- `state`: Domain-specific state (e.g., `MathEnvState`, `CodeEnvState`)
- `done`: Whether episode is finished

#### Class: `EnvBatch`

Container for multiple environments (one per problem, possibly multiple samples per problem).

```python
class EnvBatch:
    def __init__(self, env_idx_list, rollout_idx_list, max_turns):
        self.env_list = []
        for env_idx in env_idx_list:
            env = Env(env_idx, max_turns)
            self.env_list.append(env)
```

---

### `base/agent.py`

#### DataClass: `AgentData`

```python
@dataclass
class AgentData:
    current_prompt: Optional[Dict[str, Any]] = {"text": None, "image": None}
    current_action: Optional[Any] = None
    agent_reward: Optional[float] = 0.0
    success: bool = False
    done: bool = False
    answer_history: List[Any] = field(default_factory=list)
    action_history: List[Any] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
```

#### Class: `Agent`

Abstract base class for all agents.

```python
class Agent(AgentData):
    @abstractmethod
    def update_from_env(self, env_data: Env, **kwargs) -> Env:
        """
        Update agent's prompt based on environment state.
        
        Returns updated env_data.
        """
        pass
    
    @abstractmethod
    def update_from_model(self, env_data: Env, **kwargs) -> Env:
        """
        Process model's response and extract action.
        
        Returns updated env_data.
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset agent for new episode."""
        pass
```

**Agent Lifecycle**:
```
1. update_from_env(env)  → Create prompt from env state
2. [LLM generates response]
3. update_from_model(response) → Extract action from response
4. env.step(action) → Environment processes action
5. agent.calculate_reward(env) → Compute agent reward
```

---

## Math Environment

### Problem

Solve mathematical problems (competition math, olympiad problems).

**Datasets**: AIME, AMC, MATH, Polaris

### `math/math_env.py`

#### State: `MathEnvState`

```python
@dataclass
class MathEnvState:
    problem: str  # Problem statement
    ground_truth_answer: str  # Correct answer
    
    # Reasoning agent outputs
    reasoning_generated_solution: str
    reasoning_extracted_answer: str
    reasoning_is_correct: bool
    
    # Tool agent outputs
    code_generated_solution: str  # Python code
    code_extracted_answer: str    # Code execution result
    code_is_correct: bool
    
    # Alignment check
    code_reasoning_aligned: bool  # Do both agents agree?
    
    # Histories (for multi-turn refinement)
    reasoning_generated_solution_history: List
    code_generated_solution_history: List
    reasoning_extracted_answer_history: List
    code_extracted_answer_history: List
```

#### Class: `MathEnv`

```python
class MathEnv(Env):
    def __init__(self, env_idx, rollout_idx, max_turns, config):
        super().__init__(...)
        self.state = MathEnvState()
    
    def reset(self):
        # Clear all state fields
        self.state.reasoning_generated_solution = None
        self.state.code_generated_solution = None
        # ... reset all fields
```

**Note**: `MathEnv` itself doesn't implement `step()`. Instead, agents implement `agent.step(env)` which modifies the env state.

#### Class: `MathEnvBatch`

```python
class MathEnvBatch:
    def __init__(self, env_idx_list, rollout_idx_list, samples, max_turns, config, mode="train"):
        # Load problems from dataset
        self.problem_list = load_math_problem_batch(env_idx_list, mode, config)
        
        # Create env for each (problem, sample) pair
        self.env_list = []
        for i, problem in enumerate(self.problem_list):
            state = MathEnvState(
                problem=problem["question"],
                ground_truth_answer=problem["solution"]
            )
            for s in range(samples):
                env = MathEnv(env_idx=i, rollout_idx=rollout_idx_list[i*samples+s], ...)
                env.state = copy.deepcopy(state)
                self.env_list.append(env)
```

**Key Points**:
- Loads problems from `math_datasets/`
- Creates `samples` environments per problem
- Each env gets its own copy of the state

---

### Math Agents

#### `math/agents/reasoning_agent.py`

**Purpose**: Generate step-by-step reasoning solutions.

##### Method: `update_from_env(turn_idx, env_data)`

```python
def update_from_env(self, turn_idx: int, env_data: Env):
    state = env_data.state
    problem = state.problem
    
    # Get history
    reasoning_history = state.reasoning_generated_solution_history
    code_history = state.code_generated_solution_history
    
    # Construct prompt
    if turn_idx == 0:
        # First turn: direct problem solving
        prompt = (
            f"Problem:\n{problem}\n\n"
            f"Please think step by step and output the final answer in \\boxed{{}} format.\n"
            f"Example: \\boxed{{123}}\n"
        )
    else:
        # Later turns: refine based on history
        prompt = (
            f"You are a helpful assistant that refines mathematical solutions.\n\n"
            f"Problem:\n{problem}\n\n"
        )
        # Add history of previous attempts
        for i, (reasoning, answer) in enumerate(zip(reasoning_history, ...)):
            prompt += f"Turn {i+1} reasoning: {reasoning}\n"
            prompt += f"Turn {i+1} answer: {answer}\n"
        
        prompt += (
            f"Please select or refine the best solution.\n"
            f"Output in \\boxed{{<answer>}} format.\n"
        )
    
    self.current_prompt = {"text": prompt, "image": None}
```

**Prompt Strategy**:
- **Turn 0**: Direct problem solving
- **Turn >0**: Show history, ask to refine or select best

##### Method: `update_from_model(response)`

```python
def update_from_model(self, response: str):
    self.current_action = response
    return self.current_action
```

##### Method: `step(env_data, env_worker)`

```python
async def step(self, env_data: Env, env_worker=None):
    # Extract answer from response (parse \boxed{...})
    env_data.state.reasoning_generated_solution = truncate(self.current_action)
    env_data.state.reasoning_extracted_answer = parse(self.current_action)
    
    # Update history
    env_data.state.reasoning_generated_solution_history.append(...)
    env_data.state.reasoning_extracted_answer_history.append(...)
    
    # Verify correctness
    extracted_answer = env_data.state.reasoning_extracted_answer
    ground_truth = env_data.state.ground_truth_answer
    
    is_correct = verify(extracted_answer, parse(ground_truth))
    env_data.state.reasoning_is_correct = is_correct
    
    if is_correct:
        self.success = True
        env_data.done = True  # Can terminate early
    
    # Check alignment with code agent
    if env_data.state.code_extracted_answer is not None:
        is_aligned = verify(
            env_data.state.code_extracted_answer,
            env_data.state.reasoning_extracted_answer
        )
        env_data.state.code_reasoning_aligned = is_aligned
        if is_aligned:
            env_data.done = True  # Both agree, done!
```

**Reward**:
```python
def calculate_reward(self, env_data: Env):
    self.agent_reward = int(env_data.state.reasoning_is_correct)
    self.reward_history.append(self.agent_reward)
```

---

#### `math/agents/tool_agent.py`

**Purpose**: Generate Python code to solve the problem computationally.

##### Method: `update_from_env(turn_idx, env_data)`

```python
def update_from_env(self, turn_idx: int, env_data: Env):
    state = env_data.state
    problem = state.problem
    
    # Get history
    code_history = state.code_generated_solution_history
    reasoning_history = state.reasoning_generated_solution_history
    
    if turn_idx == 0:
        # First turn: generate code
        prompt = (
            f"Problem:\n{problem}\n\n"
            f"Please write Python code to solve this problem.\n"
            f"Print the final answer at the end.\n"
            f"Wrap your code in ```python\\n...\\n```\n"
        )
    else:
        # Later turns: fix code based on errors or reasoning
        prompt = (
            f"Problem:\n{problem}\n\n"
            f"Previous attempts:\n"
        )
        for i, code in enumerate(code_history):
            result = state.code_extracted_answer_history[i]
            prompt += f"Turn {i+1} code:\n```python\n{code}\n```\n"
            prompt += f"Turn {i+1} result: {result}\n"
        
        # Show reasoning agent's answer if available
        if reasoning_history:
            prompt += f"\nReasoning agent's answer: {state.reasoning_extracted_answer}\n"
        
        prompt += (
            f"Please fix the code or write a better solution.\n"
            f"Wrap in ```python\\n...\\n```\n"
        )
    
    self.current_prompt = {"text": prompt, "image": None}
```

##### Method: `step(env_data, env_worker)`

```python
async def step(self, env_data: Env, env_worker):
    # Extract code from response
    code = extract_code_from_response(self.current_action)
    env_data.state.code_generated_solution = code
    
    # Execute code in Docker sandbox
    result = await execute_code_in_worker(code, env_worker)
    env_data.state.code_extracted_answer = parse(result)
    
    # Update history
    env_data.state.code_generated_solution_history.append(code)
    env_data.state.code_extracted_answer_history.append(result)
    
    # Verify correctness
    is_correct = verify(
        env_data.state.code_extracted_answer,
        parse(env_data.state.ground_truth_answer)
    )
    env_data.state.code_is_correct = is_correct
    
    if is_correct:
        self.success = True
        env_data.done = True
    
    # Check alignment with reasoning agent
    if env_data.state.reasoning_extracted_answer is not None:
        is_aligned = verify(
            env_data.state.code_extracted_answer,
            env_data.state.reasoning_extracted_answer
        )
        env_data.state.code_reasoning_aligned = is_aligned
        if is_aligned:
            env_data.done = True
```

**Code Execution**:
- Code runs in isolated Ray Docker worker
- Captures stdout as answer
- Parses numerical/symbolic result

---

### `math/math_utils.py`

#### Function: `load_math_problem_batch(...)`

```python
def load_math_problem_batch(env_indices, mode, config, benchmark_name):
    """
    Load math problems from dataset.
    
    Args:
        env_indices: List of problem indices to load
        mode: "train" or "validate"
        config: Config object
        benchmark_name: "AIME24", "AIME", "AMC", "MATH", "polaris"
    
    Returns:
        List of problem dicts with keys:
            - "question": Problem statement
            - "solution": Ground truth answer
    """
    dataset_path = f"math_datasets/{benchmark_name}/{mode}.jsonl"
    problems = []
    
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if i in env_indices:
                problem = json.loads(line)
                problems.append({
                    "question": problem["problem"],
                    "solution": problem["answer"]
                })
    
    return problems
```

#### Functions: `parse(response)` and `verify(answer1, answer2)`

```python
def parse(response: str) -> str:
    """
    Extract answer from \\boxed{...} format.
    
    Example: "The answer is \\boxed{42}" → "42"
    """
    import re
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1).strip()
    return None

def verify(predicted, ground_truth) -> bool:
    """
    Compare two answers (handles numerical, symbolic, string).
    
    Returns True if answers are equivalent.
    """
    # Try numerical comparison
    try:
        return abs(float(predicted) - float(ground_truth)) < 1e-6
    except:
        pass
    
    # Try symbolic comparison (using sympy)
    try:
        from sympy import simplify, sympify
        return simplify(sympify(predicted) - sympify(ground_truth)) == 0
    except:
        pass
    
    # Fallback: string comparison
    return predicted.strip() == ground_truth.strip()
```

---

## Code Environment

### Problem

Generate code to solve coding problems, with test case validation.

**Datasets**: APPS, LiveCodeBench

### `code/code_env.py`

#### State: `CodeEnvState`

```python
@dataclass
class CodeEnvState:
    problem: str  # Problem description
    golden_code: str  # Reference solution (if available)
    
    # Code agent outputs
    generated_code: str
    generated_code_history: List[str]
    
    # Test agent outputs
    generated_test_input: List[str]
    generated_test_output: List[str]
    
    # Ground truth tests
    ground_truth_test_input: List[str]
    ground_truth_test_output: List[str]
    
    # Execution results
    exe_code_generated_test_output: List[str]  # Code on generated tests
    exe_code_ground_truth_test_output: List[str]  # Code on GT tests
    
    # Evaluation metrics
    ground_truth_test_vs_generated_code_match_ratio: float
    generated_test_vs_generated_code_match_ratio: float
    generated_test_vs_golden_code_match_ratio: float
    
    # Mismatch cases (for debugging)
    ground_truth_test_vs_generated_code_mismatch_cases: List[Dict]
```

#### Class: `CodeEnv`

```python
class CodeEnv(Env):
    def __init__(self, env_idx, rollout_idx, max_turns, config):
        super().__init__(...)
        self.state = CodeEnvState()
        self.backend = "ray_docker"  # Isolated execution
```

#### Class: `CodeEnvBatch`

```python
class CodeEnvBatch:
    def __init__(self, env_idx_list, rollout_idx_list, samples, max_turns, config, mode="train"):
        # Load problems
        if mode == "train":
            self.problem_list = load_problem_batch(
                env_idx_list,
                benchmark_name="train",
                mode="train",
                difficulty="difficult"
            )
        else:
            benchmark_name = config.env.benchmark  # "LiveCodeBench", "APPS", etc.
            self.problem_list = load_problem_batch(
                env_idx_list,
                mode="validate",
                benchmark_name=benchmark_name
            )
        
        # Create envs
        self.env_list = []
        for i, problem in enumerate(self.problem_list):
            state = CodeEnvState(
                problem=problem["question"],
                ground_truth_test_input=problem["test_input"],
                ground_truth_test_output=problem["test_output"]
            )
            for s in range(samples):
                env = CodeEnv(...)
                env.state = copy.deepcopy(state)
                self.env_list.append(env)
```

---

### Code Agents

#### `code/agents/code_agent.py` - `CodeGenerationAgent`

**Purpose**: Generate code solutions.

##### Method: `update_from_env(turn_idx, env_data)`

```python
def update_from_env(self, turn_idx: int, env_data: Env):
    state = env_data.state
    problem = state.problem
    
    if turn_idx == 0:
        # First turn: generate code
        prompt = (
            f"Write a Python function to solve:\n\n"
            f"{problem}\n\n"
            f"Requirements:\n"
            f"- Implement the required function\n"
            f"- Handle edge cases\n"
            f"- Return the result\n\n"
            f"Wrap your code in ```python\\n...\\n```\n"
        )
    else:
        # Later turns: fix based on test failures
        code_history = state.generated_code_history
        mismatch_cases = state.ground_truth_test_vs_generated_code_mismatch_cases
        
        prompt = (
            f"Your previous solution failed some tests.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Previous code:\n```python\n{code_history[-1]}\n```\n\n"
            f"Failed test cases:\n"
        )
        for case in mismatch_cases[:5]:  # Show first 5 failures
            prompt += f"  Input: {case['input']}\n"
            prompt += f"  Expected: {case['expected_output']}\n"
            prompt += f"  Got: {case['actual_output']}\n\n"
        
        prompt += f"Fix the code to pass all tests.\n"
    
    self.current_prompt = {"text": prompt, "image": None}
```

##### Method: `step(env_data, env_worker)`

```python
async def step(self, env_data: Env, env_worker):
    # Extract code
    code = extract_code_from_response(self.current_action)
    env_data.state.generated_code = code
    env_data.state.generated_code_history.append(code)
    
    # Evaluate against ground truth tests
    match_ratio, match_cases, mismatch_cases = await evaluate_code_against_tests(
        code=code,
        test_inputs=env_data.state.ground_truth_test_input,
        test_outputs=env_data.state.ground_truth_test_output,
        env_worker=env_worker
    )
    
    env_data.state.ground_truth_test_vs_generated_code_match_ratio = match_ratio
    env_data.state.ground_truth_test_vs_generated_code_mismatch_cases = mismatch_cases
    
    # Success if all tests pass
    if match_ratio >= 1.0:
        self.success = True
        env_data.done = True
```

**Reward**:
```python
def calculate_reward(self, env_data: Env):
    # Reward = test pass rate
    self.agent_reward = env_data.state.ground_truth_test_vs_generated_code_match_ratio
```

---

#### `code/agents/unit_test_agent.py` - `UnitTestGenerationAgent`

**Purpose**: Generate test cases to verify code.

##### Method: `update_from_env(turn_idx, env_data)`

```python
def update_from_env(self, turn_idx: int, env_data: Env):
    state = env_data.state
    problem = state.problem
    generated_code = state.generated_code
    
    prompt = (
        f"Generate unit tests for this problem:\n\n"
        f"{problem}\n\n"
        f"Code to test:\n```python\n{generated_code}\n```\n\n"
        f"Generate 5-10 diverse test cases covering:\n"
        f"- Normal cases\n"
        f"- Edge cases\n"
        f"- Corner cases\n\n"
        f"Format as JSON:\n"
        f"```json\n"
        f"[\n"
        f"  {{\"input\": [...], \"expected_output\": ...}},\n"
        f"  ...\n"
        f"]\n"
        f"```\n"
    )
    
    self.current_prompt = {"text": prompt, "image": None}
```

##### Method: `step(env_data, env_worker)`

```python
async def step(self, env_data: Env, env_worker):
    # Parse test cases from response
    tests = extract_test_cases_from_response(self.current_action)
    
    env_data.state.generated_test_input = [t["input"] for t in tests]
    env_data.state.generated_test_output = [t["expected_output"] for t in tests]
    
    # Evaluate generated code against generated tests
    match_ratio, _, _ = await evaluate_code_against_tests(
        code=env_data.state.generated_code,
        test_inputs=env_data.state.generated_test_input,
        test_outputs=env_data.state.generated_test_output,
        env_worker=env_worker
    )
    
    env_data.state.generated_test_vs_generated_code_match_ratio = match_ratio
    
    # Also evaluate tests against golden code (if available)
    if env_data.state.golden_code:
        match_ratio_golden, _, _ = await evaluate_code_against_tests(
            code=env_data.state.golden_code,
            test_inputs=env_data.state.generated_test_input,
            test_outputs=env_data.state.generated_test_output,
            env_worker=env_worker
        )
        env_data.state.generated_test_vs_golden_code_match_ratio = match_ratio_golden
```

**Reward**:
```python
def calculate_reward(self, env_data: Env):
    # Reward based on test quality (how well tests match golden code)
    self.agent_reward = env_data.state.generated_test_vs_golden_code_match_ratio
```

---

### `code/code_utils.py`

#### Function: `evaluate_code_against_tests(...)`

```python
async def evaluate_code_against_tests(
    code: str,
    test_inputs: List,
    test_outputs: List,
    env_worker
) -> Tuple[float, List, List]:
    """
    Execute code on test inputs and compare with expected outputs.
    
    Returns:
        match_ratio: Fraction of tests passed (0.0 to 1.0)
        match_cases: List of passed test cases
        mismatch_cases: List of failed test cases
    """
    match_cases = []
    mismatch_cases = []
    
    for test_input, expected_output in zip(test_inputs, test_outputs):
        # Execute code with this input
        actual_output = await run_code_with_input(code, test_input, env_worker)
        
        if actual_output == expected_output:
            match_cases.append({
                "input": test_input,
                "expected_output": expected_output,
                "actual_output": actual_output
            })
        else:
            mismatch_cases.append({
                "input": test_input,
                "expected_output": expected_output,
                "actual_output": actual_output
            })
    
    total_tests = len(test_inputs)
    match_ratio = len(match_cases) / total_tests if total_tests > 0 else 0.0
    
    return match_ratio, match_cases, mismatch_cases
```

---

## Search Environment

### Problem

Answer questions requiring multi-hop web search (e.g., "Who is the spouse of the director of Movie X?").

**Datasets**: HotpotQA, 2WikiMultihopQA, Bamboogle, MuSiQue

### `search/search_env.py`

#### State: `SearchEnvState`

```python
@dataclass
class SearchEnvState:
    question: str  # Question to answer
    ground_truth_answer: str
    
    # Search agent outputs
    search_queries: List[str]  # Queries generated
    search_results: List[str]  # Search results
    
    # Reasoning agent outputs
    reasoning_steps: str  # Analysis of search results
    extracted_answer: str  # Final answer
    is_correct: bool
```

---

### Search Agents

#### `search/agents/web_search_agent.py` - `WebSearchAgent`

**Purpose**: Generate search queries and retrieve information.

```python
def update_from_env(self, turn_idx, env_data):
    question = env_data.state.question
    previous_results = env_data.state.search_results
    
    if turn_idx == 0:
        prompt = (
            f"Question: {question}\n\n"
            f"Generate a search query to find relevant information.\n"
            f"Output only the query, no explanation.\n"
        )
    else:
        prompt = (
            f"Question: {question}\n\n"
            f"Previous search results:\n{previous_results}\n\n"
            f"Generate a follow-up search query to find missing information.\n"
        )
    
    self.current_prompt = {"text": prompt, "image": None}

async def step(self, env_data, env_worker):
    query = self.current_action.strip()
    env_data.state.search_queries.append(query)
    
    # Execute search (using Google API, Bing API, etc.)
    results = await perform_web_search(query)
    env_data.state.search_results.append(results)
```

#### `search/agents/reasoning_agent.py` - `ReasoningAgent`

**Purpose**: Analyze search results and answer the question.

```python
def update_from_env(self, turn_idx, env_data):
    question = env_data.state.question
    search_results = env_data.state.search_results
    
    prompt = (
        f"Question: {question}\n\n"
        f"Search results:\n"
    )
    for i, result in enumerate(search_results):
        prompt += f"Result {i+1}:\n{result}\n\n"
    
    prompt += (
        f"Based on the search results, answer the question.\n"
        f"Output format: \\boxed{{answer}}\n"
    )
    
    self.current_prompt = {"text": prompt, "image": None}

async def step(self, env_data, env_worker):
    env_data.state.reasoning_steps = self.current_action
    env_data.state.extracted_answer = parse(self.current_action)
    
    is_correct = verify(
        env_data.state.extracted_answer,
        env_data.state.ground_truth_answer
    )
    env_data.state.is_correct = is_correct
    
    if is_correct:
        self.success = True
        env_data.done = True
```

---

## Stateful Environment

### Problem

Execute multi-step plans with stateful tool usage (e.g., file operations, database queries).

**Datasets**: Custom benchmarks with tool APIs

### `stateful/stateful_env.py`

#### State: `StatefulEnvState`

```python
@dataclass
class StatefulEnvState:
    task_description: str
    available_tools: List[str]  # ["read_file", "write_file", "search", ...]
    
    # Plan agent outputs
    plan: str  # High-level plan
    
    # Tool agent outputs
    tool_calls: List[Dict]  # [{"tool": "read_file", "args": {...}}, ...]
    tool_results: List[Any]  # Results from executing tools
    
    # Environment state
    file_system: Dict[str, str]  # {filename: content}
    database: Dict[str, Any]  # {table: rows}
    
    task_completed: bool
```

---

### Stateful Agents

#### `stateful/agents/plan_agent.py` - `PlanAgent`

**Purpose**: Create high-level plans.

```python
def update_from_env(self, turn_idx, env_data):
    task = env_data.state.task_description
    available_tools = env_data.state.available_tools
    previous_results = env_data.state.tool_results
    
    prompt = (
        f"Task: {task}\n\n"
        f"Available tools: {', '.join(available_tools)}\n\n"
    )
    
    if previous_results:
        prompt += f"Previous tool results:\n{previous_results}\n\n"
    
    prompt += (
        f"Create a step-by-step plan to complete the task.\n"
        f"Output as numbered steps.\n"
    )
    
    self.current_prompt = {"text": prompt, "image": None}
```

#### `stateful/agents/tool_agent.py` - `ToolAgent`

**Purpose**: Execute individual tool calls.

```python
def update_from_env(self, turn_idx, env_data):
    plan = env_data.state.plan
    tool_results = env_data.state.tool_results
    
    prompt = (
        f"Plan:\n{plan}\n\n"
        f"Previous tool calls:\n{tool_results}\n\n"
        f"What is the next tool call to execute?\n"
        f"Output as JSON:\n"
        f"{{\"tool\": \"tool_name\", \"args\": {{...}}}}\n"
    )
    
    self.current_prompt = {"text": prompt, "image": None}

async def step(self, env_data, env_worker):
    tool_call = json.loads(self.current_action)
    env_data.state.tool_calls.append(tool_call)
    
    # Execute tool
    result = await execute_tool(
        tool_name=tool_call["tool"],
        args=tool_call["args"],
        env_state=env_data.state
    )
    env_data.state.tool_results.append(result)
    
    # Check if task is complete
    if check_task_completion(env_data.state):
        self.success = True
        env_data.done = True
```

---

## Ray Docker Workers

For code execution and tool usage, PettingLLMs uses Ray actors running in Docker containers for isolation.

### `math/math_worker.py` and `code/code_worker.py`

```python
def get_ray_docker_worker_cls():
    """
    Returns Ray actor class for sandboxed code execution.
    """
    @ray.remote
    class RayDockerWorker:
        def __init__(self, worker_idx):
            self.worker_idx = worker_idx
            self.container = self._create_docker_container()
        
        def _create_docker_container(self):
            import docker
            client = docker.from_env()
            container = client.containers.run(
                image="python:3.9-slim",
                command="sleep infinity",
                detach=True,
                mem_limit="512m",
                network_mode="none"  # No internet access
            )
            return container
        
        def execute_code(self, code: str, timeout: int = 10):
            """Execute code in Docker container."""
            try:
                result = self.container.exec_run(
                    cmd=["python", "-c", code],
                    stdout=True,
                    stderr=True,
                    timeout=timeout
                )
                return result.output.decode()
            except Exception as e:
                return f"Error: {str(e)}"
        
        def cleanup(self):
            self.container.stop()
            self.container.remove()
    
    return RayDockerWorker
```

**Usage**:
```python
# In MultiAgentsExecutionEngine.__init__:
num_workers = config.training.num_workers  # 150-300
env_workers = [RayDockerWorker.remote(idx) for idx in range(num_workers)]

# In agent.step():
result = await env_worker.execute_code.remote(code)
```

---

## Turn Order and Multi-Agent Interaction

### Configuration

```yaml
multi_agent_interaction:
  turn_order: ["tool_generator", "reasoning_generator"]
  num_interacting_agents: 2
```

### Execution

```python
# In MultiAgentsExecutionEngine._generate_single_rollout():
for turn_idx in range(max_turns):
    for agent_idx, agent_name in enumerate(self.turn_order):
        # 1. Agent updates from environment
        env = agent.update_from_env(turn_idx, env)
        
        # 2. Generate response
        response = await llm_async_generate(...)
        
        # 3. Agent processes response
        env = agent.update_from_model(response)
        
        # 4. Agent steps in environment
        await agent.step(env, env_worker)
        
        # 5. Compute reward
        agent.calculate_reward(env)
        
        # 6. Check termination
        if env.done:
            break
    
    if env.done:
        break
```

**Turn Order Examples**:
- **Math**: `["tool_generator", "reasoning_generator"]` - Code first, then reasoning
- **Code**: `["code_generator", "test_generator"]` - Code first, then tests
- **Search**: `["web_search_agent", "reasoning_agent"]` - Search first, then analyze

---

## Summary Table

| Environment | Agents | Datasets | Key Features |
|-------------|--------|----------|--------------|
| **Math** | Reasoning, Tool | AIME, AMC, MATH, Polaris | Code execution, answer verification |
| **Code** | Code, UnitTest | APPS, LiveCodeBench | Test-driven development, sandboxed execution |
| **Search** | WebSearch, Reasoning | HotpotQA, 2Wiki, MuSiQue | Multi-hop search, web APIs |
| **Stateful** | Plan, Tool | Custom | Stateful tools, multi-step planning |

---

This concludes the comprehensive Multi-Agent Environment documentation!

