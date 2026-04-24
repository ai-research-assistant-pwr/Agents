import os
import sys
import torch
import yaml
import re
import json
import time
from typing import Dict, Any

agents_dir = os.environ.get("AGENTS_DIR", "/home/tymrom7227/disk/Agents")

if agents_dir not in sys.path:
    sys.path.insert(0, agents_dir)

from src.grpo.grpo_train.agents import AgentPrompts
from src.grpo.grpo_train.rewards import calculate_step_reward

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)

from marti.utils.agent import AgentExecutorBase, AgentInstanceBase

def _build_role_transition(
    role_name: str,
    system_prompt: str,
    user_content: str,
) -> str:
    """
    Builds a ChatML-formatted string to switch the model's role between Retriever and Generator.
    """
    return (
        f"\n<|im_start|>system\n"
        f"[ROLE: {role_name}]\n"
        f"{system_prompt}\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_content}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class AgentInstance(AgentInstanceBase):
    """
    GRPO Environment — serves as the multi-agent environment for training Retriever and Generator agents in a cooperative scientific hypothesis generation task.

    Shared model: Retriever and Generator are the same underlying model, but we switch their "role" and expected behavior through system prompts and environment feedback.
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
                "environment": {"max_turns": 4},
                "rewards": {
                    "format_correct": 0.1,
                    "format_error": -1.0,
                    "turn_penalty": -0.1,
                    "task_success": 1.0,
                    "task_suboptimal": 0.5,
                    "hallucination_penalty": -1.0,
                }
            }

        self.max_turns = int(raw_config.get("environment", {}).get("max_turns", 4))
        self.reward_cfg = dict(raw_config.get("rewards", {}))

        # Episode state
        self.current_turn = 0 # even turns: Retriever, odd turns: Generator
        self.episode_id = "unknown"
        self.initial_query = ""
        self.hidden_chunks: list[str] = []
        self.expected_action = "GENERATE"
        self.history: list[dict] = []


    def _log_to_file(self, data: dict):
        """Saves step data to a JSONL file for later analysis."""
        log_dir = os.path.join(agents_dir, "data", "eval_results")
        os.makedirs(log_dir, exist_ok=True)
        run_id = os.getenv("SLURM_JOB_ID", str(int(time.time())))
        log_path = os.path.join(log_dir, f"debug_rollouts_{run_id}.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"LOGGING ERROR: {e}")

    def reset(self, states: dict, **kwargs) -> dict:
        """
        Initializes new episode with the given states. Extracts initial query and hidden chunks, sets up episode state, and returns the initial observation for the model.
        """
        self.current_turn = 0
        self.episode_id = str(time.time())
        self.history = []

        self.expected_action = (
            states.get("expected_action") or states.get("label") or "GENERATE"
        )
        raw_obs = (
            states.get("prompt")
            or states.get("observation")
            or states.get("query")
            or ""
        )

        # Extract hidden_chunks and remove them from the observation before passing to the model
        hidden_match = re.search(
            r"<HIDDEN_CHUNKS>(.*?)</HIDDEN_CHUNKS>", raw_obs, re.DOTALL
        )
        if hidden_match:
            try:
                self.hidden_chunks = json.loads(hidden_match.group(1))
            except Exception:
                self.hidden_chunks = []
            obs = re.sub(
                r"<HIDDEN_CHUNKS>.*?</HIDDEN_CHUNKS>", "", raw_obs, flags=re.DOTALL
            ).strip()
        else:
            self.hidden_chunks = []
            obs = raw_obs

        self.initial_query = obs
        return {"observation": obs, "label": self.expected_action}

    def _detect_agent_action(self, text: str, turn: int) -> str:
        """
        Parses the model's output based on the active role.

        Retriever (even turn): expects <MESSAGE>...</MESSAGE>
        Generator (odd turn): expects <REQUEST> or <THOUGHT>
          - <REQUEST> has priority: if the model generated both tags,
            we treat it as ASK (the model is requesting data, but also has thoughts).
          - <THOUGHT> without <REQUEST> → GENERATE.
          - Lack of both → FORMAT_ERROR.
        """
        if turn % 2 == 0:  # Retriever
            if re.search(r"<MESSAGE>.*?</MESSAGE>", text, re.DOTALL | re.IGNORECASE):
                return "RETRIEVER_SUCCESS"
            return "FORMAT_ERROR"

        else:  # Generator
            if re.search(r"<REQUEST>.*?</REQUEST>", text, re.DOTALL | re.IGNORECASE):
                return "ASK"
            if re.search(r"<THOUGHT>.*?</THOUGHT>", text, re.DOTALL | re.IGNORECASE):
                return "GENERATE"
            return "FORMAT_ERROR"

    def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        """
        Processes the model's action, calculates the reward, and builds feedback for the next turn.

        Pipeline:
          1. Parse the model's action (_detect_agent_action)
          2. If FORMAT_ERROR or max_turns exceeded -> penalty, done=True
          3. Calculate the reward (calculate_step_reward)
          4. Build environment_feedback for the next turn (role switching)
          5. Return the result
        """
        action_text = states.get("action_text", "")
        expected_act = (
            states.get("expected_action")
            or states.get("label")
            or self.expected_action
        )
        turn_at_action = self.current_turn

        agent_action = self._detect_agent_action(action_text, turn_at_action)

        # early check for episode termination conditions: max turns or format error
        if turn_at_action >= self.max_turns or agent_action == "FORMAT_ERROR":
            reward, reason, metrics = calculate_step_reward(
                action_text=action_text,
                action_type=agent_action,
                expected_action=expected_act,
                current_turn=turn_at_action,
                reward_cfg=self.reward_cfg,
            )
            self._log_to_file({
                "episode_id": self.episode_id,
                "turn": turn_at_action,
                "action": agent_action,
                "reason": reason,
                "reward": reward,
                "done": True,
                "forced": True,
            })
            return self._build_return(reward, "", True, states, metrics)

        # mapping from detected action to reward logic
        reward_action_map = {
            "RETRIEVER_SUCCESS": "RETRIEVE",
            "ASK": "ASK",
            "GENERATE": "GENERATE",
        }

        reward, reason, metrics = calculate_step_reward(
            action_text=action_text,
            action_type=reward_action_map.get(agent_action),
            expected_action=expected_act,
            current_turn=turn_at_action,
            reward_cfg=self.reward_cfg,
        )

        done = False
        env_feedback = ""

        # role switching

        if agent_action == "RETRIEVER_SUCCESS":
            # Retriever finished compression -> pass <MESSAGE> to generator
            msg_match = re.search(
                r"<MESSAGE>(.*?)</MESSAGE>", action_text, re.DOTALL | re.IGNORECASE
            )
            retriever_msg = msg_match.group(1).strip() if msg_match else "No content."

            env_feedback = _build_role_transition(
                role_name="GENERATOR",
                system_prompt=AgentPrompts.get_generator_system_prompt(),
                user_content=f"Retriever Message:\n{retriever_msg}",
            )
            self.current_turn += 1

        elif agent_action == "ASK":
            # Generator asks for more data -> provide hidden_chunks to retriever
            req_match = re.search(
                r"<REQUEST>(.*?)</REQUEST>", action_text, re.DOTALL | re.IGNORECASE
            )
            gen_req = req_match.group(1).strip() if req_match else "More data needed."

            # Provide hidden_chunks and clear them (one-time use)
            extra_data = (
                "\n".join(self.hidden_chunks)
                if self.hidden_chunks
                else "No additional detailed chunks found."
            )
            self.hidden_chunks = []

            env_feedback = _build_role_transition(
                role_name="RETRIEVER",
                system_prompt=AgentPrompts.get_retriever_system_prompt(),
                user_content=(
                    f"Generator Request:\n{gen_req}\n\n"
                    f"New Detailed Chunks:\n{extra_data}"
                ),
            )
            self.current_turn += 1

        elif agent_action == "GENERATE":
            # Generator generated a hypothesis -> end of episode
            done = True

        # Save step to history
        self.history.append({
            "turn": turn_at_action,
            "action": agent_action,
            "reward": reward,
            "reason": reason,
        })

        self._log_to_file({
            "episode_id": self.episode_id,
            "turn": turn_at_action,
            "action": agent_action,
            "reason": reason,
            "reward": reward,
            "done": done,
        })

        return self._build_return(reward, env_feedback, done, states, metrics)

    def _build_return(
        self,
        reward: float,
        feedback: str,
        done: bool,
        states: dict,
        metrics: dict,
    ) -> Dict[str, Any]:
        """
        Builds a return dictionary for the step function, including rewards, environment feedback, done flag, and extra logs.
        """
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
                **{
                    k: torch.tensor(float(v), dtype=torch.float32)
                    for k, v in metrics.items()
                },
            },
        }

class AgentExecutor(AgentExecutorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AgentInstance, *args, **kwargs)