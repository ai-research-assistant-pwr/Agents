"""
State-transition function for the scientific hypothesis generation pipeline.
============================================================================

``apply_step`` is the single pure transition function: given the current
``TrajectoryState``, the completed step type, and the extracted payload, it
returns the next state with turn_id incremented and the appropriate accumulator
list updated.

``compute_final_reward`` computes the weighted similarity + diversity +
groundedness + relevancy reward using the embedding and reranker servers.

Step transitions
----------------
  retriever_search   → appends search_result to state.search_results
  retriever_message  → appends message to state.retriever_messages
  generator_ask      → appends question to state.generator_questions
  generator_generate → appends hypotheses to state.hypotheses; trajectory ends
                       when turn_id reaches len(step_sequence)
"""

from typing import Any, Dict, List, Optional, Tuple

from src.grpo.grpo_train.rewards import (
    embedding_similarity_reward_from_embs,
    get_hypothesis_and_label_embeddings,
    groundedness_reward,
    hypothesis_diversity_reward,
    relevancy_reward,
)
from src.grpo.grpo_train.state import TrajectoryState


# ── reward computation ────────────────────────────────────────────────────────


async def compute_final_reward(
    hypotheses: List[str],
    label: str,
    embed_host: str,
    embed_port: int,
    similarity_weight: float,
    diversity_weight: float,
    rerank_host: Optional[str] = None,
    rerank_port: Optional[int] = None,
    groundedness_weight: float = 0.0,
    relevancy_weight: float = 0.0,
    papers: Optional[List[Dict[str, Any]]] = None,
    query: Optional[str] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Compute the weighted reward for a list of hypotheses.

    Returns ``(reward, similarity_score, diversity_score, groundedness_score, relevancy_score)``.
    All values are 0.0 when the hypothesis list is empty.

    Groundedness and relevancy are only computed when ``rerank_host`` and
    ``rerank_port`` are provided and the respective weights are non-zero.
    """
    if not hypotheses:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    hyp_embs, label_emb = await get_hypothesis_and_label_embeddings(
        hypotheses, label, embed_host, embed_port
    )
    sim = embedding_similarity_reward_from_embs(hyp_embs, label_emb)
    div = hypothesis_diversity_reward(hyp_embs)

    ground = 0.0
    relev = 0.0
    use_reranker = rerank_host is not None and rerank_port is not None

    if use_reranker and groundedness_weight > 0.0:
        ground = await groundedness_reward(
            hypotheses=hypotheses,
            papers=papers or [],
            rerank_host=rerank_host,
            rerank_port=rerank_port,
        )

    if use_reranker and relevancy_weight > 0.0 and query:
        relev = await relevancy_reward(
            hypotheses=hypotheses,
            query=query,
            rerank_host=rerank_host,
            rerank_port=rerank_port,
        )

    reward = round(
        similarity_weight * sim
        + diversity_weight * div
        + groundedness_weight * ground
        + relevancy_weight * relev,
        4,
    )
    return reward, sim, div, ground, relev


# ── state transition ──────────────────────────────────────────────────────────


def apply_step(
    state: TrajectoryState,
    step: str,
    payload: Dict[str, Any],
    turn_record: Dict[str, Any],
    debug_entry: Dict[str, Any],
) -> TrajectoryState:
    """
    Pure transition: advance turn_id and update the relevant accumulator.

    Parameters
    ----------
    state       : Current (immutable) trajectory state.
    step        : Step type string (e.g. ``"retriever_search"``).
    payload     : Extracted content for this step:
                    retriever_search   → {"query": str, "search_result": str}
                    retriever_message  → {"message": str}
                    generator_ask      → {"question": str}
                    generator_generate → {"hypotheses": List[str]}
    turn_record : MARTI-compatible record dict to append.
    debug_entry : Human-readable debug dict to append.
    """
    new_records = state.trajectory_records + [turn_record]
    new_debug = state.debug_entries + [debug_entry]

    if step == "retriever_search":
        return TrajectoryState(
            turn_id=state.turn_id + 1,
            step_sequence=state.step_sequence,
            search_results=state.search_results + [payload["search_result"]],
            retriever_messages=state.retriever_messages,
            generator_questions=state.generator_questions,
            paper_block=state.paper_block,
            query=state.query,
            papers=state.papers,
            hypotheses=state.hypotheses,
            trajectory_records=new_records,
            debug_entries=new_debug,
        )

    if step == "retriever_message":
        return TrajectoryState(
            turn_id=state.turn_id + 1,
            step_sequence=state.step_sequence,
            search_results=state.search_results,
            retriever_messages=state.retriever_messages + [payload["message"]],
            generator_questions=state.generator_questions,
            paper_block=state.paper_block,
            query=state.query,
            papers=state.papers,
            hypotheses=state.hypotheses,
            trajectory_records=new_records,
            debug_entries=new_debug,
        )

    if step == "generator_ask":
        return TrajectoryState(
            turn_id=state.turn_id + 1,
            step_sequence=state.step_sequence,
            search_results=state.search_results,
            retriever_messages=state.retriever_messages,
            generator_questions=state.generator_questions + [payload["question"]],
            paper_block=state.paper_block,
            query=state.query,
            papers=state.papers,
            hypotheses=state.hypotheses,
            trajectory_records=new_records,
            debug_entries=new_debug,
        )

    if step == "generator_generate":
        return TrajectoryState(
            turn_id=state.turn_id + 1,
            step_sequence=state.step_sequence,
            search_results=state.search_results,
            retriever_messages=state.retriever_messages,
            generator_questions=state.generator_questions,
            paper_block=state.paper_block,
            query=state.query,
            papers=state.papers,
            hypotheses=payload["hypotheses"],
            trajectory_records=new_records,
            debug_entries=new_debug,
        )

    raise ValueError(f"Unknown step type in apply_step: {step!r}")
