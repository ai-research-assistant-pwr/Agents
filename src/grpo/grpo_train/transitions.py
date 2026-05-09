"""
State-transition functions for the scientific hypothesis generation pipeline.
=============================================================================

``apply_tool_call`` is the core pure transition function: given the current
``TrajectoryState`` and a validated tool invocation it returns the next state.

``compute_final_reward`` is an async helper that calls the embedding server
and computes the weighted similarity + diversity reward.

``_terminal_bad_call`` is an internal helper that stamps a trajectory as
failed due to a malformed or disallowed tool call.
"""

from typing import Any, Dict, List, Tuple

from src.grpo.grpo_train.rewards import (
    embedding_similarity_reward_from_embs,
    get_hypothesis_and_label_embeddings,
    hypothesis_diversity_reward,
)
from src.grpo.grpo_train.state import HistoryEntry, TrajectoryState


# ── reward computation ────────────────────────────────────────────────────────


async def compute_final_reward(
    hypotheses: List[str],
    label: str,
    embed_host: str,
    embed_port: int,
    similarity_weight: float,
    diversity_weight: float,
) -> Tuple[float, float, float]:
    """
    Compute the weighted reward for a list of hypotheses.

    Returns ``(reward, similarity_score, diversity_score)``.
    All three values are 0.0 when the hypothesis list is empty.
    """
    if not hypotheses:
        return 0.0, 0.0, 0.0
    hyp_embs, label_emb = await get_hypothesis_and_label_embeddings(
        hypotheses, label, embed_host, embed_port
    )
    sim = embedding_similarity_reward_from_embs(hyp_embs, label_emb)
    div = hypothesis_diversity_reward(hyp_embs)
    reward = round(similarity_weight * sim + diversity_weight * div, 4)
    return reward, sim, div


# ── terminal-state factory ────────────────────────────────────────────────────


def _terminal_bad_call(
    state: TrajectoryState,
    turn_record: Dict[str, Any],
    debug_entry: Dict[str, Any],
) -> TrajectoryState:
    """Return a terminal state representing a malformed or disallowed tool call."""
    return TrajectoryState(
        turn_id=state.turn_id + 1,
        current_agent=state.current_agent,
        history=state.history,
        tool_usage=state.tool_usage,
        paper_block=state.paper_block,
        query=state.query,
        is_terminal=True,
        terminal_reason="bad_tool_call",
        hypotheses=None,
        trajectory_records=state.trajectory_records + [turn_record],
        debug_entries=state.debug_entries + [debug_entry],
    )


# ── state transitions ─────────────────────────────────────────────────────────


def apply_tool_call(
    state: TrajectoryState,
    tool_name: str,
    tool_args: Dict[str, Any],
    turn_record: Dict[str, Any],
    debug_entry: Dict[str, Any],
) -> TrajectoryState:
    """
    Pure transition function.

    Given the current state and a *validated* tool invocation, return the next
    ``TrajectoryState``.  The three possible outcomes are:

    ``send_to_generator``
        Control passes to the generator; the retriever message is appended to
        ``history``.

    ``ask_retriever``
        Control passes back to the retriever; the generator's question is
        appended to ``history`` and ``ask_retriever`` usage is incremented.

    ``generate_hypotheses``
        Terminal state; ``hypotheses`` is populated.

    Any other tool name (unreachable after validation in ``execute_turn``) falls
    through to ``_terminal_bad_call``.
    """
    new_records = state.trajectory_records + [turn_record]
    new_debug = state.debug_entries + [debug_entry]

    if tool_name == "send_to_generator":
        message = tool_args.get("message", "")
        new_entry = HistoryEntry(
            agent="retriever", tool="send_to_generator", message=message
        )
        return TrajectoryState(
            turn_id=state.turn_id + 1,
            current_agent="generator",
            history=state.history + [new_entry],
            tool_usage=state.tool_usage,
            paper_block=state.paper_block,
            query=state.query,
            is_terminal=False,
            terminal_reason=None,
            hypotheses=None,
            trajectory_records=new_records,
            debug_entries=new_debug,
        )

    if tool_name == "ask_retriever":
        question = tool_args.get("question", "")
        new_entry = HistoryEntry(
            agent="generator", tool="ask_retriever", message=question
        )
        new_usage = {
            **state.tool_usage,
            "ask_retriever": state.tool_usage.get("ask_retriever", 0) + 1,
        }
        return TrajectoryState(
            turn_id=state.turn_id + 1,
            current_agent="retriever",
            history=state.history + [new_entry],
            tool_usage=new_usage,
            paper_block=state.paper_block,
            query=state.query,
            is_terminal=False,
            terminal_reason=None,
            hypotheses=None,
            trajectory_records=new_records,
            debug_entries=new_debug,
        )

    if tool_name == "generate_hypotheses":
        hypotheses = tool_args.get("hypotheses", [])
        return TrajectoryState(
            turn_id=state.turn_id + 1,
            current_agent="generator",
            history=state.history,
            tool_usage=state.tool_usage,
            paper_block=state.paper_block,
            query=state.query,
            is_terminal=True,
            terminal_reason="hypotheses_generated",
            hypotheses=hypotheses,
            trajectory_records=new_records,
            debug_entries=new_debug,
        )

    # unknown tool — should be unreachable after validation in execute_turn
    return _terminal_bad_call(state, turn_record, debug_entry)
