"""
Reward functions for the scientific hypothesis generation pipeline.

Retriever reward     – gaussian centred at TARGET_WORDS (100) on raw output word count
Generator turn 1     – binary: does the output look like a question?
Generator turn 3     – numbered-list format score + count score (peak at 3 hypotheses)
"""

import re
import math
from typing import Tuple

TARGET_WORDS: int = 100
WORD_COUNT_SIGMA: float = 30.0


def retriever_word_count_reward(output: str) -> float:
    """
    Gaussian reward centred at TARGET_WORDS words evaluated on the raw output.
    Returns a value in (0, 1].
    """
    word_count = len(output.split())
    score = math.exp(-0.5 * ((word_count - TARGET_WORDS) / WORD_COUNT_SIGMA) ** 2)
    return round(score, 4)


def generator_request_format_reward(output: str) -> float:
    """
    Turn 1 – reward the generator for asking a question.
    Returns 1.0 if output contains a '?', else 0.0.
    """
    return 1.0 if "?" in output else 0.0


def generator_hypothesis_reward(output: str) -> float:
    """
    Turn 3 – reward the generator for the hypothesis list.

    format_score – fraction of detected items that start with a number+dot/paren
    count_score  – triangular, peak=3: 1.0 for 3, 0.67 for 2/4, 0.33 for 1/5, 0.0 for 0/6+

    Returns 0.5 * format_score + 0.5 * count_score in [0, 1].
    """
    # Numbered items: lines starting with "1. " / "1) " etc.
    numbered = re.findall(r"(?m)^\s*\d+[.)]\s+\S", output)
    n = len(numbered)

    format_score = min(n / 3.0, 1.0)  # up to 3 properly formatted items = full score

    if n == 0:
        count_score = 0.0
    elif n <= 6:
        count_score = max(0.0, 1.0 - abs(n - 3) / 3.0)
    else:
        count_score = 0.0

    return round(0.5 * format_score + 0.5 * count_score, 4)


# ── legacy shim ───────────────────────────────────────────────────────────────


def calculate_step_reward(
    action_text: str,
    action_type: str,
    turn_id: int,
    **_kwargs,
) -> Tuple[float, str, dict]:
    if turn_id % 2 == 0:
        r = retriever_word_count_reward(action_text)
        reason = f"retriever word-count reward (turn {turn_id})"
    elif turn_id == 1:
        r = generator_request_format_reward(action_text)
        reason = "generator request format reward"
    else:
        r = generator_hypothesis_reward(action_text)
        reason = "generator hypothesis reward"
    return r, reason, {"reward": r}
