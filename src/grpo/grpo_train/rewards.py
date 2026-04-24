import re
from typing import Dict, Tuple, Any

# Future reward components placeholders

def compute_groundedness(generation: str, source_data: str) -> float:
    return 0.0

def compute_clarity(generation: str) -> float:
    return 0.0

def compute_relevancy(request: str, missing_context: str) -> float:
    return 0.0

def compute_diversity(hypothesis: str, previous_hypotheses: list[str]) -> float:
    return 0.0

def compute_communication_quality(
    retriever_message: str,
    generator_request: str | None
) -> float:
    return 0.0


# MAIN REWARD FUNCTION

def calculate_step_reward(
    action_text: str,
    action_type: str | None,
    expected_action: str,
    current_turn: int,
    reward_cfg: Dict[str, float],
    source_data: str = "",
    previous_hypotheses: list[str] | None = None,
    generator_request: str | None = None,
) -> Tuple[float, str, Dict[str, float]]:
    """
    Calculates reward based on the agent's action, expected action, and current turn in the GRPO environment.

    Round logic (max_turns=4):
      Tura 0: Retriever    → expected RETRIEVE (kompresja do <MESSAGE>)
      Tura 1: Generator    → expected ASK lub GENERATE (first response)
      Tura 2: Retriever    → expected RETRIEVE (answer to <REQUEST>)
      Tura 3: Generator    → expected GENERATE (after receiving additional info)

    Returns: (total_reward, reason_string, metrics_dict)
    """
    total_reward = 0.0
    reason = ""
    metrics = {
        "rule_reward": 0.0,
        "format_reward": 0.0,
        "model_reward": 0.0,
        "turn_penalty": 0.0,
        "groundedness": 0.0,
        "clarity": 0.0,
        "relevancy": 0.0,
        "diversity": 0.0,
        "communication_quality": 0.0,
    }

    # Format error check
    if action_type == "FORMAT_ERROR" or action_type is None:
        penalty = reward_cfg.get("format_error", -1.0)
        metrics["format_reward"] = penalty
        total_reward = penalty
        return round(total_reward, 4), "Format Error: Missing required XML tags.", metrics

    metrics["format_reward"] = reward_cfg.get("format_correct", 0.1)

    # -------------------------------------------------------------------------
    # Turn penalty
    # -------------------------------------------------------------------------
    turn_pen = current_turn * reward_cfg.get("turn_penalty", -0.1)
    metrics["turn_penalty"] = turn_pen
    total_reward += turn_pen

    
    # Role specific reward logic
    is_retriever = (current_turn % 2 == 0)

    if is_retriever:
        # --- RETRIEVER ---
        # For now, reward only the correct compression format.
        if action_type == "RETRIEVE":
            metrics["rule_reward"] = reward_cfg.get("task_success", 0.5)
            reason = "Retriever processed chunks correctly."

    else:
        # --- GENERATOR ---
        # Logic for 4 cases based on expected vs actual action and turn number.
        if action_type == "GENERATE":
            if current_turn == 1 and expected_action == "GENERATE":
                # Case A: optimal immediate generation
                metrics["rule_reward"] = reward_cfg.get("task_success", 1.0)
                reason = "Correctly GENERATEd immediately (sufficient data)."

            elif current_turn == 1 and expected_action == "ASK":
                # Case B: hallucination - model generates without data
                metrics["rule_reward"] = reward_cfg.get("hallucination_penalty", -1.0)
                reason = "Premature GENERATE — should have ASKed for missing data."

            elif current_turn > 1 and expected_action == "ASK":
                # Case C: optimal generation after gathering data
                metrics["rule_reward"] = reward_cfg.get("task_success", 1.0)
                reason = "Correctly GENERATEd after gathering additional info."

            else:
                # Case D: generated correctly, but asked unnecessarily earlier
                # (current_turn > 1 AND expected_action == "GENERATE")
                metrics["rule_reward"] = reward_cfg.get("task_suboptimal", 0.5)
                reason = "GENERATEd correctly, but wasted a turn asking unnecessarily."

        elif action_type == "ASK":
            if current_turn == 1 and expected_action == "ASK":
                # Case E: optimal request for missing data
                metrics["rule_reward"] = reward_cfg.get("task_success", 1.0)
                reason = "Correctly ASKed for missing data."

            else:
                # Case F: too many rounds or unnecessary ASK
                metrics["rule_reward"] = reward_cfg.get("task_suboptimal", 0.2)
                reason = "Suboptimal ASK (unnecessary or repeated request)."

    # Aggregation of future reward components
    total_reward += (
        metrics["format_reward"]
        + metrics["rule_reward"]
        + metrics["model_reward"]
        + metrics["groundedness"]
        + metrics["clarity"]
        + metrics["relevancy"]
        + metrics["diversity"]
        + metrics["communication_quality"]
    )

    return round(total_reward, 4), reason, metrics