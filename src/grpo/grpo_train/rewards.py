"""
Reward functions for the scientific hypothesis generation pipeline.

Retriever reward     – gaussian centred at TARGET_WORDS (100) on raw output word count
Generator turn 1     – binary: does the output look like a question?
Generator turn 3     – numbered-list format score + count score (peak at 3 hypotheses)
"""

import re
import math
from typing import List, Tuple

import aiohttp

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


# ── embedding similarity reward ───────────────────────────────────────────────


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(x**2 for x in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-8)


def _parse_hypotheses(output: str) -> List[str]:
    """Extract individual numbered hypotheses from the generator output.

    Returns an empty list if no numbered items are found (wrong format).
    """
    # Match lines starting with a number followed by . or )
    return re.findall(r"(?m)^\s*\d+[.)]\s+(.+)", output)


async def _get_embeddings(
    texts: List[str],
    server_host: str,
    server_port: int,
    model: str = "Qwen/Qwen3-Embedding-4B",
) -> List[List[float]]:
    """Call the vLLM OpenAI-compatible /v1/embeddings endpoint."""
    url = f"http://{server_host}:{server_port}/v1/embeddings"
    payload = {"model": model, "input": texts}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
    # Response follows OpenAI format: data[i]["embedding"]
    ordered = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in ordered]


async def embedding_similarity_reward(
    hypotheses: List[str],
    label: str,
    server_host: str,
    server_port: int,
) -> float:
    """Compute mean cosine similarity between each generated hypothesis and
    the target label using the vLLM embedding server.

    Parameters
    ----------
    hypotheses   : List of hypothesis strings extracted from the generator output.
    label        : Ground-truth hypothesis string.
    server_host  : Hostname/IP of the vLLM embedding server.
    server_port  : Port of the vLLM embedding server.

    Returns
    -------
    Mean cosine similarity in [-1, 1], typically in [0, 1] for semantic text.
    """
    texts = hypotheses + [label]
    embeddings = await _get_embeddings(texts, server_host, server_port)
    label_emb = embeddings[-1]
    hyp_embs = embeddings[:-1]
    similarities = [_cosine_similarity(e, label_emb) for e in hyp_embs]
    return round(sum(similarities) / len(similarities), 4)


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
