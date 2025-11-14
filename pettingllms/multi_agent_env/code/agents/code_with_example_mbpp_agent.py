"""
MBPP Code Generation Agent with Example Conditioning

This agent generates code for MBPP dataset and can optionally condition on examples
provided by an example generation agent. The reward structure encourages
the agent to solve problems without relying on examples.
"""

import logging
import random
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.code.code_utils import compute_mbpp_reward_fraction

logger = logging.getLogger(__name__)


class CodeWithExampleMBPPAgent(Agent):
    """
    MBPP code generation agent that can optionally condition on examples.
    
    Reward Structure:
    - If conditioned on example: Both agents get reward = 0
    - If NOT conditioned on example: Both agents get reward = fraction of tests passed
    
    This encourages the agent to solve problems independently without relying
    on examples, making examples only useful during exploration.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """Initialize the MBPP Code Generation Agent"""
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
        Create prompt for MBPP code generation.
        
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
            formatted_prompt = f"""You are a helpful Python coding assistant.
Given a task, output ONLY valid Python code that defines the required function(s).
Do not include explanations or markdown fences.
Do not include any docstrings or anything other than the python function(s).
Enclose the entire solution within ```python and ```.

Here is a solved example to help you understand the problem:

{generated_example}

Task:
{problem}

Constraints:
- Write clean, minimal Python and no other language.
- Define the function(s) exactly as implied by the tests.
- Do NOT print; just return values.
- Output ONLY code (no backticks, no explanations).
- Enclose the entire solution within ```python and ```.
"""
        else:
            # Case 2: Do NOT condition on example (will get test pass ratio reward)
            formatted_prompt = f"""You are a helpful Python coding assistant.
Given a task, output ONLY valid Python code that defines the required function(s).
Do not include explanations or markdown fences.
Do not include any docstrings or anything other than the python function(s).
Enclose the entire solution within ```python and ```.

Task:
{problem}

Constraints:
- Write clean, minimal Python and no other language.
- Define the function(s) exactly as implied by the tests.
- Do NOT print; just return values.
- Output ONLY code (no backticks, no explanations).
- Enclose the entire solution within ```python and ```.
"""
        
        # Log which strategy was chosen
        logger.info(f"Rollout {self.rollout_idx}: {'USED' if self.used_example else 'DID NOT USE'} example")
        
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
            # Try without language specifier
            matches = re.findall(r"```(.*?)```", response, re.DOTALL)
            if matches:
                code = matches[-1].strip()
            else:
                code = response.strip()
        
        self.current_action = code
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Execute the generated code against MBPP test cases and store results.
        """
        # Store generated code in environment state
        generated_code = self.current_action
        env_data.state.generated_code = generated_code
        
        # Get MBPP test cases
        mbpp_ground_truth = getattr(env_data.state, "mbpp_ground_truth", None) or []
        mbpp_setup = getattr(env_data.state, "mbpp_setup", "") or ""
        
        # Evaluate code against tests
        passed_ratio = 0.0
        if mbpp_ground_truth and isinstance(mbpp_ground_truth, list) and len(mbpp_ground_truth) > 0:
            try:
                passed_ratio = compute_mbpp_reward_fraction(
                    generated_code, 
                    mbpp_ground_truth, 
                    mbpp_setup
                )
            except Exception as e:
                logger.error(f"Failed to evaluate MBPP code: {e}")
                passed_ratio = 0.0
            
            # Store evaluation results
            env_data.state.mbpp_test_pass_ratio = passed_ratio
            
            # Determine success
            if passed_ratio >= 1.0:
                self.success = True
                env_data.done = True
            else:
                self.success = False
        else:
            logger.warning("No MBPP ground truth tests found")
            self.success = False
        
        # Store the raw test pass ratio (before applying conditioning penalty)
        env_data.state.raw_test_pass_ratio = passed_ratio
        
        logger.info(f"Rollout {self.rollout_idx}: Test pass ratio = {passed_ratio:.2f}, Used example = {self.used_example}")

    def calculate_reward(self, env_data: Env):
        """
        Calculate reward based on whether example was used.
        
        Reward Structure:
        - If conditioned on example: reward = 0 (discourages reliance)
        - If NOT conditioned on example: reward = fraction of tests passed (encourages independence)
        """
        raw_test_pass_ratio = getattr(env_data.state, "raw_test_pass_ratio", 0.0)
        used_example = getattr(env_data.state, "code_used_example", False)
        
        if not used_example:
            self.agent_reward = 0.0
        else:
            self.agent_reward = raw_test_pass_ratio
        
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

