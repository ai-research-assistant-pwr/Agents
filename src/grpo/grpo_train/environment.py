import os
import torch
import json
import re
import yaml
from openrlhf.utils.agent import AgentInstanceBase
from .rewards import calculate_step_reward

class MockHypothesisEnv(AgentInstanceBase):
    def __init__(self, *args, **kwargs):
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

    def _detect_action(self, text: str):
        if "<REQUEST>" in text:
            return "ASK"
        else:
            return "GENERATE"

    async def step(self, states: dict, **kwargs) -> dict:
        action_text = states["action_text"]

        expected_action = states.get("label") or states.get("labels")

        action = self._detect_action(action_text)

        reward, reason = calculate_step_reward(
            parsed_json=None,
            action=action,
            expected_action=expected_action,
            current_turn=self.current_turn,
            reward_cfg=self.reward_cfg
        )

        done = (
            action == "GENERATE"
            or self.current_turn >= self.max_turns
        )

        env_feedback = ""

        if action == "ASK" and not done:
            hidden_chunks = states.get("hidden_chunks", [])

            if isinstance(hidden_chunks, str):
                hidden_chunks = [hidden_chunks]

            if len(hidden_chunks) == 0:
                current_context = ["No additional data found."]
            else:
                half_idx = max(1, len(hidden_chunks) // 2)

                if self.current_turn == 0:
                    current_context = hidden_chunks[:half_idx]
                else:
                    current_context = hidden_chunks[half_idx:] or ["No additional data found."]

            context_str = "\n\n".join(current_context)

            env_feedback = (
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{context_str}"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            self.current_turn += 1

        return {
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "scores": torch.tensor(reward, dtype=torch.float32),
            "environment_feedback": env_feedback,
            "done": done,
            "extra_logs": {
                "reward_reason": reason,
                "turn": self.current_turn
            }
        }