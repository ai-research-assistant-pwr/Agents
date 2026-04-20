from typing import Dict, Tuple

def calculate_step_reward(
    parsed_json,
    action: str | None,
    expected_action: str,
    current_turn: int,
    reward_cfg: Dict[str, float]
) -> Tuple[float, str]:
    """
    Calculates reward for GRPO step based on action correctness.
    """
    reward = 0.0
    reason = ""

    if action is None:
        return reward_cfg["unknown_action_penalty"], "No action detected"

    reward += reward_cfg["turn_penalty"]

    if action == "ASK":
        if current_turn == 0 and expected_action == "ASK":
            reward += reward_cfg["correct_ask"]
            reason = "Correct ASK on first turn"
        else:
            reason = "ASK used (continuation)"

    elif action == "GENERATE":
        if current_turn == 0 and expected_action == "GENERATE":
            reward += reward_cfg["correct_generate_immediate"]
            reason = "Correct immediate GENERATE"

        elif current_turn > 0 and expected_action == "ASK":
            reward += reward_cfg["correct_generate_after_ask"]
            reason = "Correct GENERATE after ASK"

        elif current_turn == 0 and expected_action == "ASK":
            reward += reward_cfg["hallucination_penalty"]
            reason = "Hallucination (should ASK first)"

    else:
        reward += reward_cfg["unknown_action_penalty"]
        reason = f"Unknown action: {action}"

    return reward, reason