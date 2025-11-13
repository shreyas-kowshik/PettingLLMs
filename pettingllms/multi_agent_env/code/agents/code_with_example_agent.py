"""
Code Generation Agent with Example Conditioning

This agent generates code and can optionally condition on examples
provided by an example generation agent. The reward structure encourages
the agent to solve problems without relying on examples.
"""

import logging
import random
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.code.code_utils import evaluate_code_against_tests

logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    """Truncate string to specified length"""
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CodeWithExampleAgent(Agent):
    """
    Code generation agent that can optionally condition on examples.
    
    Reward Structure:
    - If conditioned on example: Both agents get reward = 0
    - If NOT conditioned on example: Both agents get reward = fraction of tests passed
    
    This encourages the agent to solve problems independently without relying
    on examples, making examples only useful during exploration.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """Initialize the Code Generation Agent"""
        super().__init__()
        self.rollout_idx = rollout_idx
        # Track whether this agent conditioned on example
        self.used_example = False
        # Accept other keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        self.used_example = False

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        Create prompt for code generation.
        
        Randomly decides whether to condition on the example or not.
        Since max_turns=1, turn_idx will always be 0.
        """
        # Save environment data
        self.env_data = env_data
        
        # Get problem and example from environment state
        state = getattr(env_data, "state", None)
        problem = getattr(state, "problem", None)
        generated_example = getattr(state, "generated_example", None)
        
        # Randomly decide whether to use example (50% chance)
        # This creates two training scenarios:
        # 1. With example (gets 0 reward) - discourages reliance
        # 2. Without example (gets test pass ratio) - encourages independence
        self.used_example = random.choice([True, False])
        
        # Store decision in state for reward calculation
        env_data.state.code_used_example = self.used_example
        
        # Build prompt based on whether we're using the example
        if self.used_example and generated_example:
            # Case 1: Condition on example (will get 0 reward)
            formatted_prompt = f"""You are a helpful assistant that generates python code to solve programming problems.

⚠️ Important: Your solution MUST read input using input() and write output using print().
The input values will be provided externally when the program runs, so do NOT hardcode or generate inputs yourself.

Here is a solved example to help you understand the problem:

{generated_example}

Now solve the following problem:

Problem:
{problem}

Please think step by step about the inputs needed and generate the code to solve the problem.
At the end, print the result.

Respond in the format:

**Code:**
```python
# your code here
```
"""
        else:
            # Case 2: Do NOT condition on example (will get test pass ratio reward)
            formatted_prompt = f"""You are a helpful assistant that generates python code to solve programming problems.

⚠️ Important: Your solution MUST read input using input() and write output using print().
The input values will be provided externally when the program runs, so do NOT hardcode or generate inputs yourself.

Now solve the following problem:

Problem:
{problem}

Please think step by step about the inputs needed and generate the code to solve the problem.
At the end, print the result.

Respond in the format:

**Code:**
```python
# your code here
```
"""
        
        # Log which strategy was chosen
        logger.info(f"Rollout {self.rollout_idx}: {'USED' if self.used_example else 'DID NOT USE'} example")
        # print("IN USED EXAMPLE STUB, USED EXAMPLE: {}".format(self.used_example))
        # print("FORMATTED PROMPT: {}".format(formatted_prompt))
        
        self.current_prompt = {"text": formatted_prompt, "image": None}

    def update_from_model(self, response: str):
        """
        Extract generated code from model response.
        """
        import re
        
        # Extract code from markdown code block
        code = ""
        matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
        if matches:
            code = matches[-1].strip()
        else:
            code = "# Could not extract code from output"
        # print("IN CODE WITH EXAMPLE AGENT UPDATE_FROM_MODEL, CODE: {}".format(code))
        
        self.current_action = code
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Execute the generated code against test cases and store results.
        """
        # Store generated code in environment state
        generated_code = self.current_action
        env_data.state.generated_code = generated_code
        
        # Get ground truth test cases
        ground_truth_test_input = env_data.state.ground_truth_test_input or []
        ground_truth_test_output = env_data.state.ground_truth_test_output or []
        
        # Evaluate code against tests
        passed_ratio = 0.0
        if isinstance(ground_truth_test_input, list) and isinstance(ground_truth_test_output, list) \
           and ground_truth_test_input and ground_truth_test_output:
            try:
                passed_ratio, passed_cases, failed_cases = await evaluate_code_against_tests(
                    generated_code, 
                    ground_truth_test_input, 
                    ground_truth_test_output, 
                    timeout=30.0, 
                    ray_actor=env_worker,
                    rollout_idx=self.rollout_idx
                )
            except Exception as e:
                logger.error(f"Failed to evaluate code: {e}")
                passed_ratio, passed_cases, failed_cases = 0.0, [], [f"error: {e}"]
            
            # Store evaluation results
            env_data.state.ground_truth_test_vs_generated_code_match_cases = passed_cases
            env_data.state.ground_truth_test_vs_generated_code_mismatch_cases = failed_cases
            env_data.state.ground_truth_test_vs_generated_code_match_ratio = passed_ratio
            
            # Determine success
            if passed_ratio >= 1.0 and len(ground_truth_test_input) > 0:
                self.success = True
                env_data.done = True
            else:
                self.success = False
            
            # Share success status with example agent
            env_data.state.example_agent_success = self.success
        
        # Store the raw test pass ratio (before applying conditioning penalty)
        env_data.state.raw_test_pass_ratio = passed_ratio
        
        logger.info(f"Rollout {self.rollout_idx}: Test pass ratio = {passed_ratio:.2f}, Used example = {self.used_example}")

    def calculate_reward(self, env_data: Env):
        raw_test_pass_ratio = getattr(env_data.state, "raw_test_pass_ratio", 0.0)
        used_example = getattr(env_data.state, "code_used_example", False)
        
        if not used_example:
            self.agent_reward = 0.0
            # env_data.state.example_agent_reward = 0.0
            # logger.info(f"Rollout {self.rollout_idx}: Conditioned on example → reward = 0.0")
        else:
            # Case 2: Did NOT condition on example → test pass ratio for both agents
            self.agent_reward = raw_test_pass_ratio
            # env_data.state.example_agent_reward = raw_test_pass_ratio
            # logger.info(f"Rollout {self.rollout_idx}: Did not use example → reward = {raw_test_pass_ratio:.2f}")
        
        self.reward_history.append(self.agent_reward)

    def reset(self):
        """Reset the agent's internal state for a new episode"""
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.agent_reward = None
        self.reward_history = []
        self.success = False
        self.used_example = False

