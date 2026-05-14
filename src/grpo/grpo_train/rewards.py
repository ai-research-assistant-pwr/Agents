"""
Reward functions for the scientific hypothesis generation pipeline.

Retriever reward     – gaussian centred at TARGET_WORDS (100) on raw output word count
Generator turn 1     – binary: does the output look like a question?
Generator turn 3     – numbered-list format score + count score (peak at 3 hypotheses)
"""

import re
import math
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

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


async def get_hypothesis_and_label_embeddings(
    hypotheses: List[str],
    label: str,
    server_host: str,
    server_port: int,
) -> Tuple[List[List[float]], List[float]]:
    """Fetch embeddings for all hypotheses and the label in a single request.

    Returns
    -------
    (hyp_embs, label_emb) – hypothesis embeddings and label embedding.
    """
    texts = hypotheses + [label]
    embeddings = await _get_embeddings(texts, server_host, server_port)
    return embeddings[:-1], embeddings[-1]


def embedding_similarity_reward_from_embs(
    hyp_embs: List[List[float]],
    label_emb: List[float],
) -> float:
    """Mean cosine similarity between pre-computed hypothesis and label embeddings."""
    similarities = [_cosine_similarity(e, label_emb) for e in hyp_embs]
    return round(sum(similarities) / len(similarities), 4)


def hypothesis_diversity_reward(hyp_embs: List[List[float]]) -> float:
    """Diversity reward based on pairwise cosine similarity among hypotheses.

    Computes mean pairwise cosine similarity between all hypothesis embeddings,
    then returns ``1 - mean_similarity`` so that more diverse hypotheses are
    rewarded higher.

    Returns 0.0 if fewer than 3 hypotheses are provided.
    """
    if len(hyp_embs) < 3:
        return 0.0
    pairs = [(i, j) for i in range(len(hyp_embs)) for j in range(i + 1, len(hyp_embs))]
    mean_sim = sum(
        _cosine_similarity(hyp_embs[i], hyp_embs[j]) for i, j in pairs
    ) / len(pairs)
    return round(1.0 - mean_sim, 4)


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
    hyp_embs, label_emb = await get_hypothesis_and_label_embeddings(
        hypotheses, label, server_host, server_port
    )
    return embedding_similarity_reward_from_embs(hyp_embs, label_emb)


# ── reranker reward helpers ───────────────────────────────────────────────────

# Qwen3-Reranker prompt templates (required for correct model behaviour)
_RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_RERANKER_QUERY_TEMPLATE = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
_RERANKER_DOC_TEMPLATE = "<Document>: {doc}" + _RERANKER_SUFFIX


async def _get_rerank_scores(
    query: str,
    instruction: str,
    documents: List[str],
    server_host: str,
    server_port: int,
    model: str = "Qwen/Qwen3-Reranker-0.6B",
) -> List[float]:
    """Call the vLLM /v1/rerank endpoint with Qwen3-Reranker prompt formatting.

    Parameters
    ----------
    query       : The query or hypothesis string (already plain text).
    instruction : Task-specific instruction for the reranker.
    documents   : List of document strings to score against the query.
    server_host : Hostname/IP of the vLLM reranker server.
    server_port : Port of the vLLM reranker server.
    model       : HuggingFace model name served by vLLM.

    Returns
    -------
    List of relevance scores (floats) in the same order as ``documents``.
    Returns a list of 0.0 values if the server call fails.
    """

    formatted_query = _RERANKER_QUERY_TEMPLATE.format(
        prefix=_RERANKER_PREFIX,
        instruction=instruction,
        query=query,
    )
    formatted_docs = [_RERANKER_DOC_TEMPLATE.format(doc=doc) for doc in documents]

    url = f"http://{server_host}:{server_port}/v1/rerank"

    payload = {
        "model": model,
        "query": formatted_query,
        "documents": formatted_docs,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        # Response: {"results": [{"index": int, "relevance_score": float}, ...]}
        ordered = sorted(data["results"], key=lambda x: x["index"])
        return [item["relevance_score"] for item in ordered]
    except Exception:
        return [0.0] * len(documents) 


# ── groundedness reward ───────────────────────────────────────────────────────

_GROUNDEDNESS_INSTRUCTION = (
    "Given a scientific hypothesis, retrieve scientific papers that provide "
    "evidence supporting or grounding the hypothesis"
)


async def groundedness_reward(
    hypotheses: List[str],
    papers: List[Dict[str, Any]],
    rerank_host: str,
    rerank_port: int,
    model: str = "Qwen/Qwen3-Reranker-0.6B",
) -> float:
    """Compute groundedness of hypotheses against retrieved papers.

    For each hypothesis the reranker scores all papers; the mean score across
    papers gives the groundedness for that hypothesis.  The final reward is the
    mean groundedness across all hypotheses.

    Returns 0.0 when there are no hypotheses or no papers.
    """
    if not hypotheses or not papers:
        return 0.0

    paper_texts = [
        p.get("summary", p.get("abstract", p.get("title", ""))) for p in papers
    ]
    paper_texts = [t for t in paper_texts if t]
    if not paper_texts:
        return 0.0

    scores_per_hyp: List[float] = []
    for hyp in hypotheses:
        scores = await _get_rerank_scores(
            query=hyp,
            instruction=_GROUNDEDNESS_INSTRUCTION,
            documents=paper_texts,
            server_host=rerank_host,
            server_port=rerank_port,
            model=model,
        )
        scores_per_hyp.append(sum(scores) / len(scores))

    return round(sum(scores_per_hyp) / len(scores_per_hyp), 4)


# ── relevancy reward ──────────────────────────────────────────────────────────

_RELEVANCY_INSTRUCTION = (
    "Given a research query, retrieve hypotheses that are relevant to and "
    "directly address the query"
)


async def relevancy_reward(
    hypotheses: List[str],
    query: str,
    rerank_host: str,
    rerank_port: int,
    model: str = "Qwen/Qwen3-Reranker-0.6B",
) -> float:
    """Compute relevancy of hypotheses to the original research query.

    Each hypothesis is scored against the query by the reranker; the reward is
    the mean score across all hypotheses.

    Returns 0.0 when there are no hypotheses.
    """
    if not hypotheses:
        return 0.0

    scores = await _get_rerank_scores(
        query=query,
        instruction=_RELEVANCY_INSTRUCTION,
        documents=hypotheses,
        server_host=rerank_host,
        server_port=rerank_port,
        model=model,
    )
    return round(sum(scores) / len(scores), 4)
