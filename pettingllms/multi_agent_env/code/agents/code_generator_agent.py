import logging
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.code.code_utils import evaluate_code_against_tests
import random
import re

logger = logging.getLogger(__name__)

def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]

class CodeGeneratorAgent(Agent):
    def __init__(self, rollout_idx: int | None = None, **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        self.condition_on_example = False
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)
    
    def reset(self):
        super().reset()
        self.current_action = None
        self.current_prompt = None
        self.success = False
        self.agent_reward = 0.0
        self.reward_history = []
        self.condition_on_example = False
    
    def update_from_env(self, turn_idx: int, env_data: Env):
        self.env_data = env_data
        assert turn_idx == 0, "CodeGeneratorAgent only supports single-turn generation."

        state = getattr(env_data, "state", None)
        question = getattr(state, "problem", None)
        example = getattr(state, "generated_example", None)
        self.condition_on_example = random.random() < 0.5

        if self.condition_on_example and example is not None:
            prompt = (
                f"You are a helpful coding assistant that generates clean, correct, and efficient Python code to solve programming problems. Please think step by step and then generate the code.\n"
                f"Given a programming problem and a solved example, your task is use the provided example as a guide to generate code that solves the problem correctly while handling edge cases and following best practices.\n"
                f"Important instructions:\n"
                f"• Carefully analyze the provided solved example and understand the approach taken to solve it.\n"
                f"• The program must read all inputs using input() and assume inputs are provided externally in the correct format when the program runs.\n"
                f"• The program must write all outputs using print().\n"
                f"• Do not hardcode any input values or generate inputs yourself.\n"
                f"• Do not include explanatory text outside the code block.\n\n"
                f"Here is a solved example to help you understand the problem and reasoning process:\n{example}\n\n"
                f"Now, using the insights from the example, generate Python code to solve the following problem:\n\nProblem:\n{question}\n\n"
                f"Please first think how many and the type of inputs you need, and write like x = int(input()),b=int(input()), and then generate the function to solve the problem, at last print the result.\n\n"
                f"Respond strictly in this format:\n\n"
                f"**Code:**\n```python\n# your code here\n```\n\n"
            )
            logger.info(f"CodeGeneratorAgent is conditioning on the provided example for code generation for Rollout {self.rollout_idx}.")
        else:
            prompt = (
                f"You are a helpful coding assistant that generates clean, correct, and efficient Python code to solve programming problems. Please think step by step and then generate the code.\n"
                f"Given a programming problem, your task is to generate Python code that solves the problem correctly while handling edge cases and following best practices.\n"
                f"Important instructions:\n"
                f"• The program must read all inputs using input() and assume inputs are provided externally in the correct format when the program runs.\n"
                f"• The program must write all outputs using print().\n"
                f"• Do not hardcode any input values or generate inputs yourself.\n"
                f"• Do not include explanatory text outside the code block.\n\n"
                f"Now, generate Python code to solve the following problem:\n\nProblem:\n{question}\n\n"
                f"Please first think how many and the type of inputs you need, and write like x = int(input()),b=int(input()), and then generate the function to solve the problem, at last print the result.\n\n"
                f"Respond strictly in this format:\n\n"
                f"**Code:**\n```python\n# your code here\n```\n\n"
            )
            logger.info(f"CodeGeneratorAgent is generating code without conditioning on an example for Rollout {self.rollout_idx}.")
        self.current_prompt = {"text": prompt, "image": None}
        env_data.state.code_conditioned_on_example = self.condition_on_example
    
    def update_from_model(self, response: str):
        matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
        if matches:
            code = matches[-1].strip()
        else:
            code = response.strip()
        self.current_action = code
        return self.current_action
    
    async def step(self, env_data: Env, env_worker:Any=None):
        generated_code = self.current_action
        env_data.state.generated_code = generated_code
        ground_truth_test_input = env_data.state.ground_truth_test_input or []
        ground_truth_test_output = env_data.state.ground_truth_test_output or []
        passed_ratio = 0.0

        if isinstance(ground_truth_test_input, list) and isinstance(ground_truth_test_output, list) and ground_truth_test_input and ground_truth_test_output:
            try:
                passed_ratio, _, _ = await evaluate_code_against_tests(
                    generated_code,
                    ground_truth_test_input,
                    ground_truth_test_output,
                    timeout=30.0,
                    ray_actor=env_worker,
                    rollout_idx=self.rollout_idx
                )
            except Exception as e:
                passed_ratio = 0.0
                logger.error(f"Error during code evaluation for Rollout {self.rollout_idx}: {e}")
            
            env_data.state.ground_truth_test_vs_generated_code_match_ratio = passed_ratio
            if passed_ratio == 1.0:
                self.success = True
            else:
                self.success = False
        else:
            self.success = False
            env_data.state.ground_truth_test_vs_generated_code_match_ratio = 0.0
    
    def calculate_reward(self, env_data: Env):
        passed_ratio = getattr(env_data.state, "ground_truth_test_vs_generated_code_match_ratio", 0.0)
        condition_on_example = getattr(env_data.state, "code_conditioned_on_example", False)

        if condition_on_example:
            self.agent_reward = passed_ratio
        else:
            self.agent_reward = 0.0
        self.reward_history.append(self.agent_reward)