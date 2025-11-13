"""
Example Generation Agent for Code Problems

This agent generates solved examples with step-by-step reasoning
to help illustrate how to approach a programming problem.
"""

import logging
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    """Truncate string to specified length"""
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class ExampleGenerationAgent(Agent):
    """
    Agent specialized for generating solved examples for programming problems.
    
    This agent does NOT generate code. Instead, it creates clear examples
    showing how to solve the problem step-by-step with toy values.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """Initialize the Example Generation Agent"""
        super().__init__()
        self.rollout_idx = rollout_idx
        # Accept other keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def reset(self):
        """Reset agent state for new episode"""
        super().reset()

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        Create prompt for generating an example.
        
        Since this agent only acts once (turn_idx=0), we only need
        to handle the initial generation case.
        """
        # Save environment data
        self.env_data = env_data
        
        # Get problem statement from environment state
        state = getattr(env_data, "state", None)
        problem = getattr(state, "problem", None)
        
        #         # Create prompt using the specified template
        #         formatted_prompt = f"""Given the following problem statement:
        # {problem}

        # Generate a clear and complete solved example for this problem.
        # 	•	Do not write any code.
        # 	•	Choose simple toy values to illustrate the process.
        # 	•	Show the step-by-step reasoning used to solve the example.
        # 	•	Clearly present the initial input, intermediate steps, and final output.
        # 	•	Format the solution neatly using bullet points or equations where appropriate.

        # Structure your response with the following sections:
        # 	1.	Problem Recap
        # 	2.	Example Input
        # 	3.	Step-by-Step Solution
        # 	4.	Final Answer
        # """

        formatted_prompt = f"""Given the following problem statement:
{problem}

Generate a clear and complete solved example for this problem.
	•	Do not write any code.
	•	Choose simple toy values to illustrate the process.
	•	Show the step-by-step reasoning used to solve the example.
	•	Clearly present the initial input, intermediate steps, and final output.
	•	Format the solution neatly using bullet points or equations where appropriate.

Structure your response with the following sections:
	1.	Problem Recap
	2.	Example Input
	3.	Step-by-Step Solution
	4.	Final Answer"
 
Example:
"""
        
        print("IN EXAMPLE AGENT UPDATE_FROM_ENV, FORMATTED PROMPT: {}".format(formatted_prompt))

        self.current_prompt = {"text": formatted_prompt, "image": None}

    def update_from_model(self, response: str):
        """
        Extract the generated example from model response.
        
        The example is just stored as-is (raw text).
        """
        # Store the entire response as the example
        print("IN EXAMPLE AGENT UPDATE_FROM_MODEL, RESPONSE: {}".format(response))

        self.current_action = response.strip()
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Store the generated example in environment state.
        
        Note: This agent doesn't have its own correctness metric.
        Its reward will be determined by how well the code agent performs.
        """
        # Store the generated example in environment state
        generated_example = self.current_action
        env_data.state.generated_example = generated_example
        
        # Log the example (truncated for readability)
        logger.info(f"Generated example (truncated): {truncatefn(generated_example, 200)}")
        
        # This agent has no intrinsic success criterion
        # Success is determined by the downstream code agent
        self.success = False

    def calculate_reward(self, env_data: Env):
        """
        Calculate reward for example agent.
        
        The reward is the same as the code agent's reward (test pass ratio),
        but it depends on whether the code agent used the example or not.
        This is set externally in the environment/execution logic.
        """
        # Reward will be set by the environment based on:
        # - If code agent conditions on example: reward = 0 (both agents)
        # - If code agent doesn't condition: reward = test_pass_ratio (both agents)
        # This is handled in the code agent's calculate_reward method
        
        # For now, initialize to 0 - will be updated later
        # if not hasattr(env_data.state, "example_agent_reward"):
        #     raise ValueError("Example agent reward not set")
        # else:
        #     self.agent_reward = env_data.state.example_agent_reward

        # Rewards are calculated in turn_order, so this agent will be called first
        # Get information from env state to get reward
        used_example = getattr(env_data.state, "code_used_example", False)
        raw_test_pass_ratio = getattr(env_data.state, "raw_test_pass_ratio", 0.0)
        if not used_example:
            self.agent_reward = 0.0
        else:
            self.agent_reward = raw_test_pass_ratio
        
        print(f"Example agent reward: {self.agent_reward}")
        
        self.reward_history.append(self.agent_reward)

    def reset(self):
        """Reset the agent's internal state for a new episode"""
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.agent_reward = None
        self.reward_history = []
        self.success = False

