import logging
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)

def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]

class ExampleGeneratorAgent(Agent):
    def __init__(self, rollout_idx: int | None = None, **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)
        
    def reset(self):
        super().reset()
        self.current_action = None
        self.current_prompt = None
        self.success = False
        self.agent_reward = 0.0
        self.reward_history = []
        
    def update_from_env(self, turn_idx: int, env_data: Env):
        self.env_data = env_data
        assert turn_idx == 0, "ExampleGeneratorAgent only supports single-turn generation."

        state = getattr(env_data, "state", None)
        question = getattr(state, "problem", None)
        prompt = (
            f"You are a helpful assistant that generates clear and challenging solved examples for programming problems.\n"
            f"Given a programming problem, instead of solving it directly, your task is to generate some new examples that illustrate how to solve the problem step-by-step.\n"
            f"This is the programming problem:\n{question}\n\n"
            f"Before providing the solved example, think carefully about how to break down the problem into manageable steps. Reason through the solution process and outline the key steps needed to arrive at the final answer.\n"
            f"Then, provide a clear, step-by-step solved example that demonstrates how to solve the problem. Make sure to explain each step thoroughly so that someone learning to code can follow along and understand the reasoning behind each part of the solution.\n"
            f"• Do not write any code for the original problem.\n"
            f"• Focus on clarity and educational value in your example.\n"
            f"• Use toy values in your example to illustrate the concepts effectively.\n"
            f"• Show all your reasoning steps before presenting the final solved example.\n"
            f"• Ensure the example is relevant to the original problem and covers key concepts that would help someone understand how to approach similar coding challenges.\n"
            f"• Finally, present the input, intermediate steps, and the final output of your solved example clearly.\n"
            f"Generate one comprehensive solved example following these guidelines."
        )
        self.current_prompt = {"text": prompt, "image": None}
    
    def update_from_model(self, response: str):
        self.current_action = response.strip()
        return self.current_action
    
    async def step(self, env_data: Env, env_worker:Any=None):
        env_data.state.generated_example = self.current_action
        self.success = False
    
    def calculate_reward(self, env_data: Env):
        # TODO: implement llm-as-judge evaluation to check if the example doesnt give code directly
        self.agent_reward = getattr(env_data.state, "example_quality_reward", 0.0)
        self.reward_history.append(self.agent_reward)