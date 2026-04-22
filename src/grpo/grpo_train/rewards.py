from typing import Dict, Tuple


def calculate_step_reward(
    parsed_json,
    action: str | None,
    expected_action: str,
    current_turn: int,
    reward_cfg: Dict[str, float]
) -> Tuple[float, str]:
    """
    Calculates reward for a GRPO step based on action correctness and strategy.
    Aware of multi-agent turns (even turns = Retriever, odd turns = Generator).
    """
    reward = 0.0
    reason = ""

    # ------------------------------------------------------------------ #
    # RETRIEVER TURN (turns 0, 2, 4…)                                     #
    # The Retriever's only job is to produce a well-formatted <MESSAGE>.  #
    # ------------------------------------------------------------------ #
    if current_turn % 2 == 0:
        if action == "ASK":
            reward += reward_cfg.get("correct_format", 1.0)
            reason = "Retriever correctly formatted <MESSAGE>"
        else:
            reward += reward_cfg.get("unknown_action_penalty", -1.0)
            reason = "Retriever failed to use <MESSAGE> tags"

        return reward, reason

    # ------------------------------------------------------------------ #
    # GENERATOR TURN (turns 1, 3, 5…)                                     #
    # ------------------------------------------------------------------ #

    # Small penalty for each additional turn to encourage efficiency.
    reward += reward_cfg.get("turn_penalty", -0.1)

    if action == "ASK":
        if current_turn == 1 and expected_action == "ASK":
            reward += reward_cfg.get("correct_ask", 1.0)
            reason = "Generator correctly ASKed for more data on first try"
        else:
            # ASK on a later turn or when not expected — neutral continuation
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
            reward += reward_cfg.get("correct_generate_immediate", 1.0)
            # Additional penalty per wasted turn (turn_penalty already added once above)
            extra_turns = current_turn - 1
            reward += reward_cfg.get("turn_penalty", -0.1) * extra_turns
            reason = f"Generator GENERATEd correctly but after {extra_turns} unnecessary turn(s)"

    else:
        reward += reward_cfg.get("unknown_action_penalty", -1.0)
        reason = f"Unknown action from Generator: {action}"

    return reward, reason