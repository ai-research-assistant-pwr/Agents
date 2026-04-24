import re
from typing import Dict, Tuple, Any

# placeholders for future reward components
def compute_groundedness(generation: str, source_data: str) -> float:
    return 0.0

def compute_clarity(generation: str) -> float:
    return 0.0

def compute_relevancy(request: str, missing_context: str) -> float:
    return 0.0


# main reward function for GRPO environment
def calculate_step_reward(
    action_text: str,
    action_type: str | None,
    expected_action: str,
    current_turn: int,
    reward_cfg: Dict[str, float]
) -> Tuple[float, str, Dict[str, float]]:
    """
    Calculates aggregated reward for GRPO.
    Returns: (total_reward, reason_string, metrics_dict)
    """
    total_reward = 0.0
    reason = ""
    metrics = {
        "rule_reward": 0.0,
        "format_reward": 0.0,
        "model_reward": 0.0,
        "turn_penalty": 0.0
    }

    # Format and syntax checks
    if action_type == "FORMAT_ERROR" or action_type is None:
        penalty = reward_cfg.get("format_error", -1.0)
        metrics["format_reward"] += penalty
        return (penalty, "Format Error: Missing required XML tags.", metrics)
    else:
        metrics["format_reward"] += reward_cfg.get("format_correct", 0.1)

    # Turn Penalty to Encourage Conciseness
    turn_pen = current_turn * reward_cfg.get("turn_penalty", -0.1)
    metrics["turn_penalty"] += turn_pen
    total_reward += turn_pen

    # Role-Specific Logic Rewards
    is_retriever = (current_turn % 2 == 0)
    
    if is_retriever:
        if action_type == "RETRIEVE":
            metrics["rule_reward"] += reward_cfg.get("task_success", 0.5)
            reason = "Retriever processed chunks correctly."
    else:
        # Generator Turn
        if action_type == "ASK":
            if current_turn == 1 and expected_action == "ASK":
                metrics["rule_reward"] += reward_cfg.get("task_success", 1.0)
                reason = "Correctly ASKed for missing data."
            else:
                metrics["rule_reward"] += reward_cfg.get("task_suboptimal", 0.2)
                reason = "Suboptimal ASK (continuation or unnecessary)."
                
        elif action_type == "GENERATE":
            if current_turn == 1 and expected_action == "GENERATE":
                metrics["rule_reward"] += reward_cfg.get("task_success", 1.0)
                reason = "Correctly GENERATEd immediately."
            elif current_turn > 1 and expected_action == "ASK":
                metrics["rule_reward"] += reward_cfg.get("task_success", 1.0)
                reason = "Correctly GENERATEd after gathering info."
            elif current_turn == 1 and expected_action == "ASK":
                metrics["rule_reward"] += reward_cfg.get("hallucination_penalty", -1.0)
                reason = "Hallucination/Premature Generation (should have ASKed)."
            else:
                metrics["rule_reward"] += reward_cfg.get("task_suboptimal", 0.5)
                reason = "GENERATEd correctly, but took unnecessary turns."

    total_reward += metrics["format_reward"] + metrics["rule_reward"] + metrics["model_reward"]
    return round(total_reward, 4), reason, metrics