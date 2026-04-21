import os
import sys
import torch
import yaml
import re
import json
import time
from typing import Dict, Any, Tuple

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)

from marti.utils.agent import AgentExecutorBase, AgentInstanceBase

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def calculate_step_reward(
    parsed_json,
    action: str | None,
    expected_action: str,
    current_turn: int,
    reward_cfg: Dict[str, float]
) -> Tuple[float, str]:
    reward = 0.0
    reason = ""

    if action is None:
        return reward_cfg.get("unknown_action_penalty", -1.0), "No action detected"

    reward += reward_cfg.get("turn_penalty", -0.1)

    if action == "ASK":
        if current_turn == 0 and expected_action == "ASK":
            reward += reward_cfg.get("correct_ask", 1.0)
            reason = "Correct ASK on first turn"
        else:
            reason = "ASK used (continuation/suboptimal)"

    elif action == "GENERATE":
        if current_turn == 0 and expected_action == "GENERATE":
            reward += reward_cfg.get("correct_generate_immediate", 1.0)
            reason = "Correct immediate GENERATE"

        elif current_turn > 0 and expected_action == "ASK":
            reward += reward_cfg.get("correct_generate_after_ask", 0.5)
            reason = "Correct GENERATE after ASK"

        elif current_turn == 0 and expected_action == "ASK":
            reward += reward_cfg.get("hallucination_penalty", -1.0)
            reason = "Hallucination (should ASK first)"
            
        elif current_turn > 0 and expected_action == "GENERATE":
            reward += reward_cfg.get("turn_penalty", -0.1)
            reason = "GENERATE after unnecessary turns"
    else:
        reward += reward_cfg.get("unknown_action_penalty", -1.0)
        reason = f"Unknown action: {action}"

    return reward, reason

class AgentInstance(AgentInstanceBase):
    """
    GRPO Environment - handles individual episode execution and reward calculation
    """
    def __init__(self):
        config_path = os.getenv("MARTI_CONFIG_PATH", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.max_turns = self.config["environment"]["max_turns"]
        self.reward_cfg = self.config["rewards"]

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
        """Reset environment with initial prompt and label"""
        states["current_turn"] = 0
        states["episode_id"] = str(time.time())
        full_prompt = states.get("observation") or states.get("prompt") or states.get("query") or ""

        hidden_match = re.search(r"<HIDDEN_CHUNKS>(.*?)</HIDDEN_CHUNKS>", full_prompt, re.DOTALL)
        if hidden_match:
            try:
                states["hidden_chunks"] = json.loads(hidden_match.group(1))
            except:
                states["hidden_chunks"] = []
            obs = re.sub(r"<HIDDEN_CHUNKS>.*?</HIDDEN_CHUNKS>", "", full_prompt, flags=re.DOTALL).strip()
        else:
            states["hidden_chunks"] = []
            obs = full_prompt

        return {
            "observation": obs,
            "initial_observation": obs,
            "hidden_chunks": states["hidden_chunks"],
            "label": states.get("label", "GENERATE")
        }

    def _detect_action(self, text: str) -> str:
        """Detect agent action from generated text (ASK or GENERATE)"""
        request_match = re.search(r"<REQUEST>(.*?)</REQUEST>", text, re.DOTALL | re.IGNORECASE)
        if request_match:
            content = request_match.group(1).strip().lower()
            if content and "none" not in content and "no additional" not in content:
                return "ASK"
        return "GENERATE"

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """Execute one step of GRPO interaction"""
        action_text = states["action_text"]
        expected_action = states.get("label", "GENERATE")
        current_turn = states.get("current_turn", 0)

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
        hidden_chunks = states.get("hidden_chunks", [])

        if action == "ASK" and not done:
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
        if sampling_params is not None:
            if hasattr(sampling_params, '__dict__'):
                sampling_params.stop = ["<|im_end|>"]
                sampling_params.stop_token_ids = [151645]
            else:
                sampling_params["stop"] = ["<|im_end|>"]
                sampling_params["stop_token_ids"] = [151645]

        log_entry = {
            "timestamp": time.time(),
            "episode_id": states.get("episode_id", "unknown"),
            "turn": current_turn,
            "observation": states.get("initial_observation", ""),
            "action_text": action_text,
            "action_detected": action,
            "expected_action": expected_action,
            "reward": float(reward),
            "reward_reason": reason,
            "done": done,
            "max_turns": self.max_turns,
            "env_feedback": env_feedback,
            "hidden_chunks": hidden_chunks,
        }
        self._log_to_file(log_entry)

        return {
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "scores": torch.tensor(reward, dtype=torch.float32),
            "environment_feedback": env_feedback,
            "done": done,
            "sampling_params": sampling_params,
            "extra_logs": {
                "turn": torch.tensor(current_turn, dtype=torch.float32),
                "reward": torch.tensor(reward, dtype=torch.float32)
            }
        }

class AgentExecutor(AgentExecutorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AgentInstance, *args, **kwargs)