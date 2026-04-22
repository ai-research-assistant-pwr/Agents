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
    Aware of multi-agent turns (even turns = Retriever, odd turns = Generator).
    """
    reward = 0.0
    reason = ""

    # --------------------------------------------------------
    # RETRIEVER TURN (ROUNDS: 0, 2, 4...)
    # --------------------------------------------------------
    if current_turn % 2 == 0:
        # Retriever should produce a well-formatted <MESSAGE> with relevant info
        if action == "ASK": 
            reward += reward_cfg.get("correct_format", 1.0)
            reason = "Retriever correctly formatted <MESSAGE>"
        else:
            reward += reward_cfg.get("unknown_action_penalty", -1.0)
            reason = "Retriever failed to use <MESSAGE> tags"
        
        return reward, reason


    # --------------------------------------------------------
    # GENERATOR TURN (ROUNDS: 1, 3, 5...)
    # --------------------------------------------------------
    
    # Penalty for each turn to encourage efficiency
    reward += reward_cfg.get("turn_penalty", -0.1)

    if action == "ASK":
        if current_turn == 1 and expected_action == "ASK":
            reward += reward_cfg.get("correct_ask", 1.0)
            reason = "Generator correctly ASKed for more data on first try"
        else:
            reason = "Generator ASK used (continuation/suboptimal)"

    elif action == "GENERATE":
        if current_turn == 1 and expected_action == "GENERATE":
            reward += reward_cfg.get("correct_generate_immediate", 1.0)
            reason = "Generator correctly GENERATEd immediately"

        elif current_turn > 1 and expected_action == "ASK":
            reward += reward_cfg.get("correct_generate_after_ask", 0.5)
            reason = "Generator correctly GENERATEd after gathering info"

        elif current_turn == 1 and expected_action == "ASK":
            reward += reward_cfg.get("hallucination_penalty", -1.0)
            reason = "Generator hallucination (should have ASKed first)"
            
        elif current_turn > 1 and expected_action == "GENERATE":
            reward += reward_cfg.get("turn_penalty", -0.1)
            reason = "GENERATE after unnecessary turns"

    else:
        reward += reward_cfg.get("unknown_action_penalty", -1.0)
        reason = f"Unknown action from Generator: {action}"

    return reward, reason