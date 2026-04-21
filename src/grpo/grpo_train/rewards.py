from typing import Dict, Tuple

def calculate_step_reward(
    parsed_json,
    action: str | None,
    expected_action: str,
    current_turn: int,
    reward_cfg: Dict[str, float]
) -> Tuple[float, str]:
    """
    Calculates reward for GRPO step based on action correctness and strategy.
    
    The reward system encourages:
    - Correct actions (ASK when needed, GENERATE when ready)
    - Multi-turn information gathering (when appropriate)
    - Early completion (when sufficient information is available)
    
    Args:
        parsed_json: Parsed JSON from action (not used in current version)
        action: Detected action ("ASK", "GENERATE", or None)
        expected_action: Ground truth action from dataset
        current_turn: Current turn number (0-indexed)
        reward_cfg: Dictionary with reward configuration values
        
    Returns:
        Tuple of (reward: float, reason: str) explaining the reward
    """
    reward = 0.0
    reason = ""

    # Unknown action penalty - severe penalty for unparseable actions
    if action is None:
        return reward_cfg.get("unknown_action_penalty", -1.0), "No action detected"

    # Apply turn penalty - encourage efficient solution finding
    reward += reward_cfg.get("turn_penalty", -0.1)

    if action == "ASK":
        # Asking for information
        if current_turn == 0 and expected_action == "ASK":
            # Correct: should ask first according to ground truth
            reward += reward_cfg.get("correct_ask", 1.0)
            reason = "Correct ASK on first turn"
        else:
            # Still valid to ask, but not the optimal first move
            reason = "ASK used (continuation/suboptimal)"

    elif action == "GENERATE":
        # Generating answer
        if current_turn == 0 and expected_action == "GENERATE":
            # Correct: can generate immediately and should according to ground truth
            reward += reward_cfg.get("correct_generate_immediate", 1.0)
            reason = "Correct immediate GENERATE"

        elif current_turn > 0 and expected_action == "ASK":
            # Good: generated after gathering info (expected after ASK)
            reward += reward_cfg.get("correct_generate_after_ask", 0.5)
            reason = "Correct GENERATE after ASK"

        elif current_turn == 0 and expected_action == "ASK":
            # Bad: hallucinating - should have asked first but generated directly
            reward += reward_cfg.get("hallucination_penalty", -1.0)
            reason = "Hallucination (should ASK first)"
            
        elif current_turn > 0 and expected_action == "GENERATE":
            # Minor penalty: took too many turns to generate
            reward += reward_cfg.get("turn_penalty", -0.1)
            reason = "GENERATE after unnecessary turns"

    else:
        # Unknown action type
        reward += reward_cfg.get("unknown_action_penalty", -1.0)
        reason = f"Unknown action: {action}"

    return reward, reason