"""
MBPP Code Generation Agent

This agent generates code for MBPP (Mostly Basic Python Problems) dataset.
It uses MBPP-specific reward computation that returns fraction of test cases passed.
"""

import logging
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.code.code_utils import compute_mbpp_reward_fraction

logger = logging.getLogger(__name__)


class CodeMBPPAgent(Agent):
    """
    Agent specialized for generating code to solve MBPP programming problems.
    Uses MBPP-specific reward computation (fraction of test cases passed).
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the MBPP Code Generation Agent.
        """
        super().__init__()
        self.rollout_idx = rollout_idx
        # Accept other unrelated keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def reset(self):
        super().reset()

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        Create prompt for MBPP code generation.
        """
        # Save environment data
        self.env_data = env_data

        # Support passing either the raw environment (with state) or a wrapped Env
        state = getattr(env_data, "state", None)
        
        question = getattr(state, "problem", None)
        
        if turn_idx == 0:
            # Generation mode
            formatted_prompt = (
                f"You are a helpful Python coding assistant.\n"
                f"Given a task, output ONLY valid Python code that defines the required function(s).\n"
                f"Do not include explanations or markdown fences.\n"
                f"Do not include any docstrings or anything other than the python function(s).\n"
                f"Enclose the entire solution within ```python and ```.\n\n"
                f"Task:\n{question}\n\n"
                f"Constraints:\n"
                f"- Write clean, minimal Python and no other language.\n"
                f"- Define the function(s) exactly as implied by the tests.\n"
                f"- Do NOT print; just return values.\n"
                f"- Output ONLY code (no backticks, no explanations).\n"
                f"- Enclose the entire solution within ```python and ```."
            )
        else:
            # Refinement mode (if needed)
            formatted_prompt = (
                f"You are a helpful Python coding assistant.\n"
                f"Given a task, output ONLY valid Python code that defines the required function(s).\n"
                f"Do not include explanations or markdown fences.\n"
                f"Do not include any docstrings or anything other than the python function(s).\n"
                f"Enclose the entire solution within ```python and ```.\n\n"
                f"Task:\n{question}\n\n"
                f"Constraints:\n"
                f"- Write clean, minimal Python and no other language.\n"
                f"- Define the function(s) exactly as implied by the tests.\n"
                f"- Do NOT print; just return values.\n"
                f"- Output ONLY code (no backticks, no explanations).\n"
                f"- Enclose the entire solution within ```python and ```."
            )

        print("IN UPDATE FROM ENV, FORMATTED PROMPT: {}".format(formatted_prompt))
        self.current_prompt = {"text": formatted_prompt, "image": None}
        
    def update_from_model(self, response: str):
        """
        Extract code from model response.
        """
        import re
        
        # Parse code
        code = ""
        
        # Try to match the code block in our prompt format
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

        print("IN UPDATE FROM MODEL, CODE: {}".format(code))

        # Update the agent's current action (environment expects a raw code string)
        self.current_action = code

        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Execute the generated code against MBPP test cases and compute reward.
        """
        # 1) Parse and update generated code
        gen_code = self.current_action
        env_data.state.generated_code = gen_code
        
        # 2) Evaluate generated code against MBPP test cases
        mbpp_ground_truth = getattr(env_data.state, "mbpp_ground_truth", None) or []
        mbpp_setup = getattr(env_data.state, "mbpp_setup", "") or ""
        
        print("IN STEP, GEN_CODE: {}".format(gen_code))
        print("IN STEP, MBPP_GROUND_TRUTH: {}".format(mbpp_ground_truth))
        print("IN STEP, MBPP_SETUP: {}".format(mbpp_setup))

        passed_ratio = 0.0
        if mbpp_ground_truth and isinstance(mbpp_ground_truth, list) and len(mbpp_ground_truth) > 0:
            try:
                passed_ratio = compute_mbpp_reward_fraction(
                    gen_code, 
                    mbpp_ground_truth, 
                    mbpp_setup
                )
            except Exception as e:
                logger.error(f"Warning: Failed to evaluate MBPP code against tests: {e}")
                passed_ratio = 0.0
            
            env_data.state.mbpp_test_pass_ratio = passed_ratio
            
            if passed_ratio >= 1.0:
                self.success = True
            else:
                self.success = False
        else:
            logger.warning("No MBPP ground truth tests found")
            self.success = False
    
    def calculate_reward(self, env_data: Env):
        """
        Calculate reward as fraction of MBPP test cases passed.
        """
        self.agent_reward = getattr(env_data.state, "mbpp_test_pass_ratio", 0.0)
        
        # BINARY 0/1 REWARD
        if self.agent_reward >= 0.5:
            self.agent_reward = 1.0
        else:
            self.agent_reward = 0.0
        
        self.reward_history.append(self.agent_reward)

    def reset(self):
        """
        Reset the agent's internal state for a new episode.
        """
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.success = False


