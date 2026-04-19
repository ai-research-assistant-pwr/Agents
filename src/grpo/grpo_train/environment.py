import os
import torch
import json
import re
import yaml
from openrlhf.utils.agent import AgentInstanceBase
from .rewards import calculate_step_reward

class MockHypothesisEnv(AgentInstanceBase):
    async def __init__(self, *args, **kwargs):
        config_path = os.getenv("MARTI_CONFIG_PATH", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.max_turns = self.config["environment"]["max_turns"]
        self.reward_cfg = self.config["rewards"]
        self.current_turn = 0

    async def reset(self, states: dict, **kwargs):
        self.current_turn = 0
        obs = states.get("observation", states.get("visible_chunks", ""))
        
        if isinstance(obs, list):
            obs = "\n\n".join(obs)
            
        return {"observation": obs}

    def _extract_json(self, text: str):
        try:
            json_str = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(json_str.group(0)) if json_str else None
        except:
            return None

    async def step(self, states: dict, **kwargs) -> dict:
        action_text = states["action_text"]
        expected_action = states["label"] 
        
        if "<REQUEST>" in action_text:
            action = "ASK"
        elif "<THOUGHT>" in action_text:
            action = "GENERATE"
        else:
            action = None

        reward, reason = calculate_step_reward(
            parsed_json={"content": action_text},
            action=action,
            expected_action=expected_action,
            current_turn=self.current_turn,
            reward_cfg=self.reward_cfg
        )

        done = (action == "GENERATE") or (self.current_turn >= self.max_turns) or (action is None)
        
        env_feedback = ""
        if action == "ASK" and not done:
            hidden_chunks = states.get("hidden_chunks", [])
            if isinstance(hidden_chunks, str):
                hidden_chunks = [hidden_chunks]
                
            half_idx = max(1, len(hidden_chunks) // 2)

            if self.current_turn == 0:
                current_context = hidden_chunks[:half_idx] if hidden_chunks else ["No additional data found."]
            else:
                current_context = hidden_chunks[half_idx:] if hidden_chunks else ["No additional data found."]

            context_str = "\n\n".join(current_context)
            
            env_feedback = f"<|im_end|>\n<|im_start|>user\n{context_str}<|im_end|>\n<|im_start|>assistant\n"
            
            self.current_turn += 1

        return {
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "scores": torch.tensor(reward, dtype=torch.float32),
            "environment_feedback": env_feedback,
            "done": done,
            "extra_logs": {"reward_reason": reason, "turn": self.current_turn}
        }