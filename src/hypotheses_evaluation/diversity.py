"""Vendi Score-based diversity calculation using embeddings or an LLM judge.

Usage:
    from hypotheses_evaluation.diversity import (
        calculate_diversity,
        calculate_judge_diversity,
        calculate_judge_diversity_gemini,
    )

    score = calculate_diversity(hypotheses=["hyp1", "hyp2", ...])
    score = calculate_judge_diversity(hypotheses=["hyp1", "hyp2", ...])
    score = calculate_judge_diversity_gemini(hypotheses=["hyp1", "hyp2", ...])
"""

from __future__ import annotations

import itertools
import json
from typing import Literal, Optional

from pydantic import BaseModel

import numpy as np
import scipy.linalg
import torch
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from app.api_client.base import Message
from app.api_client.google_client import GoogleAPIClient

# ---------------------------------------------------------------------------
# Pydantic schemas for judge-based diversity
# ---------------------------------------------------------------------------


class _SimilarityJudgment(BaseModel):
    """Structured output schema for pairwise similarity judgment via Gemini."""

    similarity: Literal["0", "1"]


# ---------------------------------------------------------------------------
# Embedding-based diversity (SPECTER2)
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
_embedding_model: Optional[SentenceTransformer] = None


def _load_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print("Loading SPECTER2 embedding model (all-mpnet-base-v2)...")
        _embedding_model = SentenceTransformer(
            "all-mpnet-base-v2",
            device=device,
        )
        print("SPECTER2 model loaded.")
    return _embedding_model


def _vendi_score(similarity_matrix: np.ndarray) -> float:
    n = similarity_matrix.shape[0]
    K = similarity_matrix / n
    eigenvalues = scipy.linalg.eigvalsh(K)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))

    return float(np.exp(entropy))


def calculate_diversity(
    hypotheses: list[str],
    batch_size: int = 64,
) -> float:
    n = len(hypotheses)
    if n < 2:
        return 1.0

    model = _load_embedding_model()
    embeddings: np.ndarray = model.encode(
        hypotheses,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    similarity_matrix = embeddings @ embeddings.T
    np.fill_diagonal(similarity_matrix, 1.0)

    return _vendi_score(similarity_matrix) / n


# ---------------------------------------------------------------------------
# Judge-based diversity (local LLM pairwise comparison)
# ---------------------------------------------------------------------------

_judge_llm: Optional[LLM] = None


def _load_judge_model() -> LLM:
    global _judge_llm
    if _judge_llm is None:
        MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
        # MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
        print(f"Loading local judge model ({MODEL_NAME})...")
        _judge_llm = LLM(
            model=MODEL_NAME,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            tensor_parallel_size=4,
        )
        print("Judge model loaded.")
    return _judge_llm


def _call_local_judge(idea_a: str, idea_b: str, model: LLM) -> float:
    prompt = f"""<|im_start|>system
You are a rigorous scientific reviewer. Evaluate whether two hypotheses propose THE SAME fundamental research mechanism (architecture, method, algorithm). Ignore the fact that they share the same general topic. Return only JSON.
<|im_end|>
<|im_start|>user
Idea A: {idea_a}
Idea B: {idea_b}

Return the result:
{{"similarity": 1}} - if they propose the same idea/method.
{{"similarity": 0}} - if they propose fundamentally different research paths.
<|im_end|>
<|im_start|>assistant
"""
    schema = '{"type": "object", "properties": {"similarity": {"type": "integer", "enum": [0, 1]}}, "required": ["similarity"]}'

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
        structured_outputs=(StructuredOutputsParams(json=schema)),
    )

    outputs = model.generate(prompt, sampling_params, use_tqdm=False)
    response_text = outputs[0].outputs[0].text.strip()

    try:
        result = json.loads(response_text)
        return float(result.get("similarity", 0.0))
    except json.JSONDecodeError:
        print(f"  Parse error: {response_text}")
        return 0.0


def calculate_judge_diversity(hypotheses: list[str]) -> float:
    n = len(hypotheses)
    if n < 2:
        return 1.0

    model = _load_judge_model()
    similarity_matrix = np.zeros((n, n))
    np.fill_diagonal(similarity_matrix, 1.0)

    for i, j in itertools.combinations(range(n), 2):
        sim_score = _call_local_judge(hypotheses[i], hypotheses[j], model)
        similarity_matrix[i, j] = sim_score
        similarity_matrix[j, i] = sim_score

    vendi = _vendi_score(similarity_matrix)
    return vendi / n


def calculate_judge_diversity_gemini(
    hypotheses: list[str],
    model: str = "gemini-3.5-flash",
) -> float:
    n = len(hypotheses)
    if n < 2:
        return 1.0

    client = GoogleAPIClient(model=model)
    similarity_matrix = np.zeros((n, n))
    np.fill_diagonal(similarity_matrix, 1.0)

    for i, j in itertools.combinations(range(n), 2):
        messages = [
            Message(
                role="system",
                content="You are a rigorous scientific reviewer. Evaluate whether two hypotheses propose THE SAME fundamental research mechanism (architecture, method, algorithm). Ignore the fact that they share the same general topic. Return only JSON.",
            ),
            Message(
                role="user",
                content=f"Idea A: {hypotheses[i]}\nIdea B: {hypotheses[j]}\n\nReturn the result:\n{{\"similarity\": 1}} - if they propose the same idea/method.\n{{\"similarity\": 0}} - if they propose fundamentally different research paths.",
            ),
        ]
        result = client.call(messages, response_schema=_SimilarityJudgment)
        sim_score = float(result.content.similarity)
        similarity_matrix[i, j] = sim_score
        similarity_matrix[j, i] = sim_score

    vendi = _vendi_score(similarity_matrix)
    return vendi / n
