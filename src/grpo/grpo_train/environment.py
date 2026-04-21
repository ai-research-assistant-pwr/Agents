import os
import sys
import torch
import yaml
import re
import json
import time
from typing import Dict, Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)

from openrlhf.utils.agent import MultiTurnAgentExecutor, AgentInstanceBase

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from rewards import calculate_step_reward

class MockHypothesisEnvInstance(AgentInstanceBase):
    def __init__(self, *args, **kwargs):
        config_path = os.getenv("MARTI_CONFIG_PATH", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.max_turns = self.config["environment"]["max_turns"]
        self.reward_cfg = self.config["rewards"]

    # =========================
    # LOGGING HELPER
    # =========================
    def _log_to_file(self, data: dict):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "..", "..", "data", "eval_results")
        log_dir = os.path.abspath(log_dir)

        os.makedirs(log_dir, exist_ok=True)

        run_id = os.getenv("SLURM_JOB_ID", str(int(time.time())))
        log_path = os.path.join(log_dir, f"debug_rollouts_{run_id}.jsonl")

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            print("LOGGING ERROR:", e)

    async def reset(self, states: dict, **kwargs):
        states["current_turn"] = 0
        states["episode_id"] = str(time.time())
        raw_obs = states.get("observation", states.get("prompt", ""))

        hidden_match = re.search(r"<HIDDEN_CHUNKS>(.*?)</HIDDEN_CHUNKS>", raw_obs, re.DOTALL)
        if hidden_match:
            try:
                states["hidden_chunks"] = json.loads(hidden_match.group(1))
            except:
                states["hidden_chunks"] = []
            obs = re.sub(r"<HIDDEN_CHUNKS>.*?</HIDDEN_CHUNKS>", "", raw_obs, flags=re.DOTALL).strip()
        else:
            states["hidden_chunks"] = []
            obs = raw_obs

        states["initial_observation"] = obs

        return {"observation": obs}

    def _detect_action(self, text: str):
        if "<REQUEST>" in text or "ASK" in text.upper():
            return "ASK"
        return "GENERATE"

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        action_text = states["action_text"]
        current_turn = states.get("current_turn", 0)
        expected_action = states.get("expected_action")

        action = self._detect_action(action_text)

        reward, reason = calculate_step_reward(
            parsed_json=None,
            action=action,
            expected_action=expected_action,
            current_turn=current_turn,
            reward_cfg=self.reward_cfg
        )

        done = (action == "GENERATE" or current_turn >= self.max_turns)
        env_feedback = ""

        if action == "ASK" and not done:
            hidden_chunks = states.get("hidden_chunks", [])

            if not hidden_chunks:
                context_str = "No additional data found."
            else:
                half = max(1, len(hidden_chunks) // 2)
                current_context = hidden_chunks[:half] if current_turn == 0 else hidden_chunks[half:]
                context_str = "\n\n".join(current_context)

            env_feedback = (
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{context_str}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            states["current_turn"] = current_turn + 1

        sampling_params = states.get("sampling_params")

        if sampling_params is None:
            sampling_params = {"stop": ["<|im_end|>"], "stop_token_ids": [151645]}
        elif isinstance(sampling_params, dict):
            sampling_params["stop"] = ["<|im_end|>"]
            sampling_params["stop_token_ids"] = [151645]
        else:
            sampling_params.stop = ["<|im_end|>"]
            sampling_params.stop_token_ids = [151645]

        # =========================
        # DEBUG LOGGING
        # =========================
        log_entry = {
            "timestamp": time.time(),
            "episode_id": states.get("episode_id"),
            "turn": current_turn,

            # input
            "observation": states.get("initial_observation"),
            "action_text": action_text,

            # decisions
            "action_detected": action,
            "expected_action": expected_action,

            # reward
            "reward": float(reward),
            "reward_reason": reason,

            # env
            "done": done,
            "max_turns": self.max_turns,

            # feedback
            "env_feedback": env_feedback,

            # debug
            "hidden_chunks": states.get("hidden_chunks", []),
            "sampling_params": str(sampling_params),
        }

        self._log_to_file(log_entry)

        return {
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "scores": torch.tensor(reward, dtype=torch.float32),
            "environment_feedback": env_feedback,
            "done": done,
            "sampling_params": sampling_params,
            "extra_logs": {
                "turn": current_turn,
                "reward": float(reward)
            }
        }

class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(MockHypothesisEnvInstance)