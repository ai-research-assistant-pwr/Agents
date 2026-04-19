from typing import Dict, Any, Tuple

def calculate_step_reward(
    parsed_json: Dict[str, Any] | None, 
    action: str | None, 
    expected_action: str, 
    current_turn: int,
    reward_cfg: Dict[str, float]
) -> Tuple[float, str]:
    """
    Calculates the reward for a given step in the GRPO environment based on the config.
    """
    reward = 0.0
    reason = ""

    if not parsed_json or not action or "content" not in parsed_json:
        return reward_cfg["invalid_json_penalty"], "Invalid JSON format"

    reward += reward_cfg["valid_json"]
    reward += reward_cfg["turn_penalty"]

    if action == "ASK":
        if current_turn == 0 and expected_action == "ASK":
            reward += reward_cfg["correct_ask"]
            reason = "Correctly identified missing context and asked for more info."
        else:
            reason = "Asked for info, continuing episode."
            
    elif action == "GENERATE":
        if current_turn == 0 and expected_action == "GENERATE":
            reward += reward_cfg["correct_generate_immediate"]
            reason = "Generated immediately with sufficient context."
        elif current_turn > 0 and expected_action == "ASK":
            reward += reward_cfg["correct_generate_after_ask"]
            reason = "Generated successfully after asking for missing context."
        elif current_turn == 0 and expected_action == "ASK":
            reward += reward_cfg["hallucination_penalty"]
            reason = "Hallucination: generated without asking when context was insufficient."
    else:
        reward += reward_cfg["unknown_action_penalty"]
        reason = f"Unknown action: {action}"

    return reward, reason