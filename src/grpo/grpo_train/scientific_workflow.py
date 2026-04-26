import os
import re
import json
import time
import asyncio
from typing import Dict, List, Any, Optional

from marti.utils.logging_utils import init_logger
from src.grpo.grpo_train.agents import AgentPrompts
from src.grpo.grpo_train.rewards import calculate_step_reward

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "INFO"))


def _build_initial_input(role_name: str, system_prompt: str, user_content: str) -> str:
    """Buduje pierwszy prompt inicjujący konwersację."""
    return (
        f"<|im_start|>system\n"
        f"[ROLE: {role_name}]\n"
        f"{system_prompt}\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_content}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _build_role_transition(role_name: str, system_prompt: str, user_content: str) -> str:
    """Buduje przejście roli wewnątrz już trwającej konwersacji."""
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


def _detect_agent_action(text: str, turn: int) -> str:
    """Wykrywa akcję modelu na podstawie aktualnej tury."""
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


async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager=None,
    task: str = "scientific_discovery",
    metadata: Optional[Dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Scientific Multi-Agent Workflow dla GRPO.
    Shared model plays both Retriever i Generator role, alternating every turn.
    """
    start_time = time.time()
    
    # agent and environment setup
    agent = agents[0]
    llm = agent.get("llm")
    tokenizer = agent.get("tokenizer")
    sampling_params = agent.get("sampling_params")
    
    if hasattr(sampling_params, "stop"):
        sampling_params.stop = ["<|im_end|>", "<|endoftext|>"]
    elif isinstance(sampling_params, dict):
        sampling_params["stop"] = ["<|im_end|>", "<|endoftext|>"]

    workflow_args = kwargs.get("workflow_args", {})
    max_turns = workflow_args.get("max_turns", 4)
    
    # Default reward configuration, can be overridden by workflow_args
    reward_cfg = workflow_args.get("reward_cfg", {
        "format_correct": 0.1,
        "format_error": -1.0,
        "turn_penalty": -0.1,
        "task_success": 1.0,
        "task_suboptimal": 0.5,
        "hallucination_penalty": -1.0,
    })

    # Parsing hidden chunks from prompt (if any) and cleaning prompt for initial input
    hidden_chunks = []
    hidden_match = re.search(r"<HIDDEN_CHUNKS>(.*?)</HIDDEN_CHUNKS>", prompt, re.DOTALL)
    if hidden_match:
        try:
            hidden_chunks = json.loads(hidden_match.group(1))
        except Exception:
            hidden_chunks = []
        clean_prompt = re.sub(r"<HIDDEN_CHUNKS>.*?</HIDDEN_CHUNKS>", "", prompt, flags=re.DOTALL).strip()
    else:
        clean_prompt = prompt.strip()

    # Initial state
    current_turn = 0
    done = False
    trajectory = []
    total_episode_reward = 0.0
    
    expected_action = label if label else "GENERATE"
    
    # Prompt initialization with system instructions and user query
    current_context = _build_initial_input(
        role_name="RETRIEVER",
        system_prompt=AgentPrompts.get_retriever_system_prompt(),
        user_content=clean_prompt
    )

    # Main loop workflow alternating between Retriever and Generator roles, max 4 rounds (Retriever -> Generator -> Retriever -> Generator)
    while not done and current_turn < max_turns:
        
        # Tokenizing current context
        input_token_ids = tokenizer(
            current_context, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].tolist()

        # Async call to the model to generate response based on current context and sampling parameters
        response = await llm.generate_async.remote(
            prompt_ids=input_token_ids,
            sampling_params=sampling_params
        )
        
        action_text = response.outputs[0].text
        output_token_ids = response.outputs[0].token_ids
        sequence_ids = input_token_ids + output_token_ids

        # Extracting logprobs for the generated tokens if available (for potential use in reward shaping or analysis)
        rollout_log_probs = None
        if getattr(sampling_params, "logprobs", None) is not None:
            rollout_log_probs = [0.0] * len(input_token_ids)
            if hasattr(response.outputs[0], "logprobs") and response.outputs[0].logprobs is not None:
                for i, logprob_dict in enumerate(response.outputs[0].logprobs):
                    if i < len(output_token_ids) and output_token_ids[i] in logprob_dict:
                        rollout_log_probs.append(logprob_dict[output_token_ids[i]].logprob)
                    else:
                        rollout_log_probs.append(0.0)
            else:
                rollout_log_probs.extend([0.0] * len(output_token_ids))

        # Action analysis and reward calculation
        agent_action = _detect_agent_action(action_text, current_turn)
        
        reward_action_map = {
            "RETRIEVER_SUCCESS": "RETRIEVE",
            "ASK": "ASK",
            "GENERATE": "GENERATE",
        }
        logical_action = reward_action_map.get(agent_action, "FORMAT_ERROR")

        step_reward, reason, metrics = calculate_step_reward(
            action_text=action_text,
            action_type=logical_action,
            expected_action=expected_action,
            current_turn=current_turn,
            reward_cfg=reward_cfg,
        )
        total_episode_reward += step_reward

        # Saving step trajectory
        agent_role = "RETRIEVER" if current_turn % 2 == 0 else "GENERATOR"
        trajectory.append({
            "turn_id": current_turn,
            "agent_index": 0,
            "agent_id": agent.get("agent_id", "shared_agent"),
            "agent_name": agent.get("agent_name", "shared_agent"),
            "agent_role": agent_role,
            "agent_input": current_context,
            "agent_output": action_text,
            "output_ids": output_token_ids,
            "sequence_ids": sequence_ids,
            "rollout_log_prob": rollout_log_probs,
            "reward": step_reward,
            "metadata": {
                "action": agent_action,
                "reason": reason,
                **metrics
            },
        })

        # Format error handling - if model's output format is incorrect, end episode immediately with penalty
        if agent_action == "FORMAT_ERROR":
            logger.info(f"Episode terminated early due to FORMAT_ERROR at turn {current_turn}.")
            done = True
            break
            
        # Task success check - if model performs expected action (e.g., generates final answer when expected), end episode with success reward
        if agent_action == "GENERATE":
            done = True
            break

        # Preparing context for next turn based on current action and role transition logic
        if agent_action == "RETRIEVER_SUCCESS":
            msg_match = re.search(r"<MESSAGE>(.*?)</MESSAGE>", action_text, re.DOTALL | re.IGNORECASE)
            retriever_msg = msg_match.group(1).strip() if msg_match else "No content."

            next_turn_addition = _build_role_transition(
                role_name="GENERATOR",
                system_prompt=AgentPrompts.get_generator_system_prompt(),
                user_content=f"Retriever Message:\n{retriever_msg}"
            )
            # Updating current context by appending the model's output and the new system prompt for the next role
            current_context = current_context + action_text + "<|im_end|>" + next_turn_addition
            current_turn += 1

        elif agent_action == "ASK":
            req_match = re.search(r"<REQUEST>(.*?)</REQUEST>", action_text, re.DOTALL | re.IGNORECASE)
            gen_req = req_match.group(1).strip() if req_match else "More data needed."

            extra_data = "\n".join(hidden_chunks) if hidden_chunks else "No additional detailed chunks found."
            hidden_chunks = []

            next_turn_addition = _build_role_transition(
                role_name="RETRIEVER",
                system_prompt=AgentPrompts.get_retriever_system_prompt(),
                user_content=f"Generator Request:\n{gen_req}\n\nNew Detailed Chunks:\n{extra_data}"
            )
            current_context = current_context + action_text + "<|im_end|>" + next_turn_addition
            current_turn += 1

    end_time = time.time()
    logger.info(f"Workflow completed. Turns: {current_turn}, Reward: {total_episode_reward:.2f}, Time: {end_time - start_time:.2f}s")

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": total_episode_reward,
    }