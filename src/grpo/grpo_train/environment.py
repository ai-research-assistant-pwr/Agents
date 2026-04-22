import os
import sys
import torch
import yaml
import re
import json
import time
from typing import Dict, Any, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from agents import AgentPrompts
from rewards import calculate_step_reward

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    import warnings
    warnings.warn(
        "Monkey-patching PreTrainedTokenizerBase.all_special_tokens_extended. "
        "Check if this is still needed after a transformers upgrade.",
        UserWarning
    )
    PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)

from marti.utils.agent import AgentExecutorBase, AgentInstanceBase

class AgentInstance(AgentInstanceBase):
    """
    GRPO Environment - handles individual episode execution and reward calculation.
    """

    async def __init__(self):
        config_path = os.getenv("MARTI_CONFIG_PATH")

        if not config_path or not os.path.exists(config_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.abspath(
                os.path.join(base_dir, "..", "..", "..", "config", "grpo", "config.yaml")
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"CRITICAL AGENT INIT ERROR: Cannot load config from {config_path}. Error: {e}")
            raise e

        self.max_turns = self.config["environment"]["max_turns"]
        self.reward_cfg = self.config["rewards"]

        self.current_turn = 0  # 0, 2… = Retriever | 1, 3… = Generator
        self.episode_id = "unknown"
        self.initial_query = ""
        self.hidden_chunks = []
        self.expected_action = "GENERATE"
        self.history = []

    def _log_to_file(self, data: dict):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "..", "..", "..", "data", "eval_results")
        os.makedirs(log_dir, exist_ok=True)
        run_id = os.getenv("SLURM_JOB_ID", str(int(time.time())))
        log_path = os.path.join(log_dir, f"debug_rollouts_{run_id}.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            print("LOGGING ERROR:", e)

    async def reset(self, states: dict, **kwargs):
        """Initializes the environment for a new episode."""
        self.current_turn = 0
        self.episode_id = str(time.time())
        self.history = []

        self.expected_action = (
            states.get("expected_action")
            or states.get("label")
            or "GENERATE"
        )

        raw_obs = states.get("prompt") or states.get("observation") or states.get("query") or ""

        hidden_match = re.search(r"<HIDDEN_CHUNKS>(.*?)</HIDDEN_CHUNKS>", raw_obs, re.DOTALL)
        if hidden_match:
            try:
                self.hidden_chunks = json.loads(hidden_match.group(1))
            except Exception:
                self.hidden_chunks = []
            obs = re.sub(r"<HIDDEN_CHUNKS>.*?</HIDDEN_CHUNKS>", "", raw_obs, flags=re.DOTALL).strip()
        else:
            self.hidden_chunks = []
            obs = raw_obs

        self.initial_query = obs

        return {
            "observation": obs,
            "label": self.expected_action
        }

    def _detect_agent_action(self, text: str, turn: int) -> str:
        """
        Detects the action taken by the model depending on whose turn it is.
        Even turns: Model = Retriever
        Odd turns:  Model = Generator
        """
        if turn % 2 == 0:
            # Retriever must wrap its summary in <MESSAGE>…</MESSAGE>
            if "<MESSAGE>" in text.upper() and "</MESSAGE>" in text.upper():
                return "RETRIEVER_SUCCESS"
            return "FORMAT_ERROR"
        else:
            # Generator either asks for more data or produces the hypothesis
            request_match = re.search(r"<REQUEST>(.*?)</REQUEST>", text, re.DOTALL | re.IGNORECASE)
            if request_match:
                return "ASK"
            return "GENERATE"

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        action_text = states.get("action_text", "")

        expected_act = (
            states.get("expected_action")
            or states.get("label")
            or self.expected_action
        )

        turn_at_action = self.current_turn

        agent_action = self._detect_agent_action(action_text, turn_at_action)

        if turn_at_action >= self.max_turns or agent_action == "FORMAT_ERROR":
            reward, reason = calculate_step_reward(
                parsed_json=None,
                action=None,
                expected_action=expected_act,
                current_turn=turn_at_action,
                reward_cfg=self.reward_cfg
            )
            self._log_to_file({
                "timestamp": time.time(),
                "episode_id": self.episode_id,
                "turn": turn_at_action,
                "agent_role": "Retriever" if turn_at_action % 2 == 0 else "Generator",
                "action_detected": agent_action,
                "reward": float(reward),
                "reward_reason": reason,
                "done": True,
                "env_feedback_preview": ""
            })
            return {
                "rewards": torch.tensor(reward, dtype=torch.float32),
                "scores": torch.tensor(reward, dtype=torch.float32),
                "environment_feedback": "",
                "done": True,
                "sampling_params": states.get("sampling_params"),
                "extra_logs": {
                    "turn": torch.tensor(turn_at_action, dtype=torch.float32),
                    "reward": torch.tensor(reward, dtype=torch.float32)
                }
            }

        reward_action_map = {
            "RETRIEVER_SUCCESS": "ASK",
            "ASK": "ASK",
            "GENERATE": "GENERATE",
            "FORMAT_ERROR": None
        }

        reward, reason = calculate_step_reward(
            parsed_json=None,
            action=reward_action_map.get(agent_action),
            expected_action=expected_act,
            current_turn=turn_at_action,
            reward_cfg=self.reward_cfg
        )

        done = False
        env_feedback = ""

        if agent_action == "RETRIEVER_SUCCESS":
            msg_match = re.search(r"<MESSAGE>(.*?)</MESSAGE>", action_text, re.DOTALL | re.IGNORECASE)
            retriever_msg = msg_match.group(1).strip() if msg_match else "No content."

            env_feedback = (
                "<|im_end|>\n"
                "<|im_start|>system\n"
                f"{AgentPrompts.get_generator_system_prompt()}\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"Retriever Message: {retriever_msg}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            self.current_turn += 1

        elif agent_action == "ASK":
            req_match = re.search(r"<REQUEST>(.*?)</REQUEST>", action_text, re.DOTALL | re.IGNORECASE)
            gen_req = req_match.group(1).strip() if req_match else "More data needed."

            extra_data = ""
            if self.hidden_chunks:
                extra_data = "\n".join(self.hidden_chunks)
                self.hidden_chunks = []
            else:
                extra_data = "No more detailed chunks found in Weaviate."

            env_feedback = (
                "<|im_end|>\n"
                "<|im_start|>system\n"
                f"{AgentPrompts.get_retriever_system_prompt()}\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"Generator is asking for: {gen_req}\n"
                f"New Detailed Chunks: {extra_data}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            self.current_turn += 1

        elif agent_action == "GENERATE":
            done = True

        sampling_params = states.get("sampling_params")
        if sampling_params is not None:
            stop_tokens = ["<|im_end|>", "<|endoftext|>"]
            if hasattr(sampling_params, "__dict__"):
                sampling_params.stop = stop_tokens
            else:
                sampling_params["stop"] = stop_tokens

        self._log_to_file({
            "timestamp": time.time(),
            "episode_id": self.episode_id,
            "turn": turn_at_action,
            "agent_role": "Retriever" if turn_at_action % 2 == 0 else "Generator",
            "action_detected": agent_action,
            "reward": float(reward),
            "reward_reason": reason,
            "done": done,
            "env_feedback_preview": env_feedback[:100] + "..." if env_feedback else ""
        })

        return {
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "scores": torch.tensor(reward, dtype=torch.float32),
            "environment_feedback": env_feedback,
            "done": done,
            "sampling_params": sampling_params,
            "extra_logs": {
                "turn": torch.tensor(turn_at_action, dtype=torch.float32),
                "reward": torch.tensor(reward, dtype=torch.float32)
            }
        }


class AgentExecutor(AgentExecutorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AgentInstance, *args, **kwargs)