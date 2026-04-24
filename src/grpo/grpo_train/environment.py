import os
import sys
import torch
import yaml
import re
import json
import time
from typing import Dict, Any, Tuple

agents_dir = os.environ.get("AGENTS_DIR", "/home/tymrom7227/disk/Agents")

if agents_dir not in sys.path:
    sys.path.insert(0, agents_dir)

from src.grpo.grpo_train.agents import AgentPrompts
from src.grpo.grpo_train.rewards import calculate_step_reward

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)

from marti.utils.agent import AgentExecutorBase, AgentInstanceBase

class AgentInstance(AgentInstanceBase):
    """
    GRPO Environment - handles individual episode execution and reward calculation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config_path = os.getenv("MARTI_CONFIG_PATH")

        if not config_path or not os.path.exists(config_path):
            config_path = os.path.join(agents_dir, "config", "grpo", "config.yaml")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
        except Exception as e:
            print(f"CRITICAL AGENT INIT ERROR: Cannot load config from {config_path}. Error: {e}")
            raw_config = {
                "environment": {"max_turns": 3},
                "rewards": {
                    "format_correct": 1.0, 
                    "format_error": -1.0,
                    "turn_penalty": -0.1, 
                    "task_success": 1.0,
                    "task_suboptimal": 0.5,
                    "hallucination_penalty": -1.0
                }
            }

        self.max_turns = int(raw_config.get("environment", {}).get("max_turns", 3))
        self.reward_cfg = dict(raw_config.get("rewards", {}))

        self.current_turn = 0  # 0, 2… = Retriever | 1, 3… = Generator
        self.episode_id = "unknown"
        self.initial_query = ""
        self.hidden_chunks = []
        self.expected_action = "GENERATE"
        self.history = []

    def _log_to_file(self, data: dict):
        log_dir = os.path.join(agents_dir, "data", "eval_results")
        os.makedirs(log_dir, exist_ok=True)
        run_id = os.getenv("SLURM_JOB_ID", str(int(time.time())))
        log_path = os.path.join(log_dir, f"debug_rollouts_{run_id}.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            print("LOGGING ERROR:", e)

    def reset(self, states: dict, **kwargs):
        self.current_turn = 0
        self.episode_id = str(time.time())
        self.history = []

        self.expected_action = states.get("expected_action") or states.get("label") or "GENERATE"
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
        return {"observation": obs, "label": self.expected_action}

    def _detect_agent_action(self, text: str, turn: int) -> str:
        """Parses output based on the active role's expected format."""
        if turn % 2 == 0:  # Retriever
            if re.search(r"<MESSAGE>.*?</MESSAGE>", text, re.DOTALL | re.IGNORECASE):
                return "RETRIEVER_SUCCESS"
            return "FORMAT_ERROR"
        else:              # Generator
            if re.search(r"<REQUEST>.*?</REQUEST>", text, re.DOTALL | re.IGNORECASE):
                return "ASK"
            if re.search(r"<THOUGHT>.*?</THOUGHT>", text, re.DOTALL | re.IGNORECASE):
                return "GENERATE"
            return "FORMAT_ERROR"

    def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        action_text = states.get("action_text", "")
        expected_act = states.get("expected_action") or states.get("label") or self.expected_action
        turn_at_action = self.current_turn
        
        agent_action = self._detect_agent_action(action_text, turn_at_action)

        # Force episode end on max turns or format error
        if turn_at_action >= self.max_turns or agent_action == "FORMAT_ERROR":
            reward, reason, metrics = calculate_step_reward(
                action_text=action_text,
                action_type=agent_action,
                expected_action=expected_act,
                current_turn=turn_at_action,
                reward_cfg=self.reward_cfg
            )
            self._log_to_file({"turn": turn_at_action, "action": agent_action, "reward": reward, "done": True})
            return self._build_return(reward, "", True, states, metrics)

        # Map successful parses to logic actions
        reward_action_map = {"RETRIEVER_SUCCESS": "RETRIEVE", "ASK": "ASK", "GENERATE": "GENERATE"}
        
        reward, reason, metrics = calculate_step_reward(
            action_text=action_text,
            action_type=reward_action_map.get(agent_action),
            expected_action=expected_act,
            current_turn=turn_at_action,
            reward_cfg=self.reward_cfg
        )

        done = False
        env_feedback = ""

        # ==========================================
        # CHATML SAFE ROLE SWITCHING
        # ==========================================
        if agent_action == "RETRIEVER_SUCCESS":
            msg_match = re.search(r"<MESSAGE>(.*?)</MESSAGE>", action_text, re.DOTALL | re.IGNORECASE)
            retriever_msg = msg_match.group(1).strip() if msg_match else "No content."

            env_feedback = (
                "\n<|im_start|>user\n"
                "[SYSTEM OVERRIDE: YOU ARE NOW THE GENERATOR AGENT]\n"
                f"{AgentPrompts.get_generator_system_prompt()}\n\n"
                f"Retriever Message:\n{retriever_msg}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            self.current_turn += 1

        elif agent_action == "ASK":
            req_match = re.search(r"<REQUEST>(.*?)</REQUEST>", action_text, re.DOTALL | re.IGNORECASE)
            gen_req = req_match.group(1).strip() if req_match else "More data needed."

            extra_data = "\n".join(self.hidden_chunks) if self.hidden_chunks else "No additional detailed chunks found."
            self.hidden_chunks = [] # Clear chunks after sending

            env_feedback = (
                "\n<|im_start|>user\n"
                "[SYSTEM OVERRIDE: YOU ARE NOW THE RETRIEVER AGENT]\n"
                f"{AgentPrompts.get_retriever_system_prompt()}\n\n"
                f"Generator Request:\n{gen_req}\n\n"
                f"New Detailed Chunks:\n{extra_data}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            self.current_turn += 1

        elif agent_action == "GENERATE":
            done = True

        self._log_to_file({"turn": turn_at_action, "action": agent_action, "reward": reward, "done": done})
        return self._build_return(reward, env_feedback, done, states, metrics)

    def _build_return(self, reward: float, feedback: str, done: bool, states: dict, metrics: dict):
        sampling_params = states.get("sampling_params", {})
        if isinstance(sampling_params, dict):
            sampling_params["stop"] = ["<|im_end|>", "<|endoftext|>"]
        elif hasattr(sampling_params, "stop"):
            sampling_params.stop = ["<|im_end|>", "<|endoftext|>"]

        return {
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "scores": torch.tensor(reward, dtype=torch.float32),
            "environment_feedback": feedback,
            "done": done,
            "sampling_params": sampling_params,
            "extra_logs": {
                "turn": torch.tensor(self.current_turn, dtype=torch.float32),
                "reward": torch.tensor(reward, dtype=torch.float32),
                **{k: torch.tensor(v, dtype=torch.float32) for k, v in metrics.items()}
            }
        }

class AgentExecutor(AgentExecutorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AgentInstance, *args, **kwargs)