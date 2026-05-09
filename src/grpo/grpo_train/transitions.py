"""
State-transition functions for the scientific hypothesis generation pipeline.
=============================================================================

``apply_tool_call`` is the core pure transition function: given the current
``TrajectoryState`` and a validated tool invocation it returns the next state.

``compute_final_reward`` is an async helper that calls the embedding server
and computes the weighted similarity + diversity reward.

``_terminal_bad_call`` is an internal helper that stamps a trajectory as
failed due to a malformed or disallowed tool call.

Tool transitions summary
------------------------
  search_papers      → stays on retriever; appends (raw_output, result) to
                       retriever_search_exchanges; increments search_papers count.
  send_to_generator  → switches to generator; resets retriever_search_exchanges
                       and search_papers count (for next potential retriever turn).
  ask_retriever      → switches to retriever; resets retriever_search_exchanges
                       and search_papers count for the new retriever turn.
  generate_hypotheses→ terminal; populates hypotheses.
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
        ask_retriever_limit=state.ask_retriever_limit,
        retriever_search_limit=state.retriever_search_limit,
        retriever_search_exchanges=state.retriever_search_exchanges,
        paper_block=state.paper_block,
        query=state.query,
        is_terminal=True,
        terminal_reason="bad_tool_call",
        hypotheses=None,
        trajectory_records=state.trajectory_records + [turn_record],
        debug_entries=state.debug_entries + [debug_entry],
    )


def _reset_retriever_turn_state(tool_usage: Dict[str, int]) -> Dict[str, int]:
    """Return a copy of tool_usage with the per-retriever-turn search counter reset."""
    return {**tool_usage, "search_papers": 0}


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
    ``TrajectoryState``.  The ``tool_args`` dict may carry a private key
    ``"_search_result"`` (injected by ``execute_turn``) when ``tool_name`` is
    ``"search_papers"``; this value is stored in ``retriever_search_exchanges``.

    Outcomes by tool
    ----------------
    search_papers
        Control stays with the retriever.  The pair
        ``(raw_llm_output, search_result)`` is appended to
        ``retriever_search_exchanges`` so the next prompt includes it.
        ``search_papers`` usage is incremented.

    send_to_generator
        Control passes to the generator.  ``retriever_search_exchanges`` and
        the ``search_papers`` counter are reset (clean slate for the next
        potential retriever turn).

    ask_retriever
        Control passes back to the retriever for a new turn.
        ``retriever_search_exchanges`` and the ``search_papers`` counter are
        reset so the retriever starts fresh.

    generate_hypotheses
        Terminal state; ``hypotheses`` is populated.

    Unknown tool name
        Falls through to ``_terminal_bad_call`` (should be unreachable after
        validation in ``execute_turn``).
    """
    new_records = state.trajectory_records + [turn_record]
    new_debug = state.debug_entries + [debug_entry]

    # ── search_papers ─────────────────────────────────────────────────────────
    if tool_name == "search_papers":
        raw_output = tool_args.get("_raw_llm_output", "")
        search_result = tool_args.get("_search_result", "")
        new_exchanges = state.retriever_search_exchanges + [(raw_output, search_result)]
        new_usage = {
            **state.tool_usage,
            "search_papers": state.tool_usage.get("search_papers", 0) + 1,
        }
        return TrajectoryState(
            turn_id=state.turn_id + 1,
            current_agent="retriever",
            history=state.history,
            tool_usage=new_usage,
            ask_retriever_limit=state.ask_retriever_limit,
            retriever_search_limit=state.retriever_search_limit,
            retriever_search_exchanges=new_exchanges,
            paper_block=state.paper_block,
            query=state.query,
            is_terminal=False,
            terminal_reason=None,
            hypotheses=None,
            trajectory_records=new_records,
            debug_entries=new_debug,
        )

    # ── send_to_generator ─────────────────────────────────────────────────────
    if tool_name == "send_to_generator":
        message = tool_args.get("message", "")
        new_entry = HistoryEntry(
            agent="retriever", tool="send_to_generator", message=message
        )
        return TrajectoryState(
            turn_id=state.turn_id + 1,
            current_agent="generator",
            history=state.history + [new_entry],
            tool_usage=_reset_retriever_turn_state(state.tool_usage),
            ask_retriever_limit=state.ask_retriever_limit,
            retriever_search_limit=state.retriever_search_limit,
            retriever_search_exchanges=[],  # reset for next retriever turn
            paper_block=state.paper_block,
            query=state.query,
            is_terminal=False,
            terminal_reason=None,
            hypotheses=None,
            trajectory_records=new_records,
            debug_entries=new_debug,
        )

    # ── ask_retriever ─────────────────────────────────────────────────────────
    if tool_name == "ask_retriever":
        question = tool_args.get("question", "")
        new_entry = HistoryEntry(
            agent="generator", tool="ask_retriever", message=question
        )
        new_usage = {
            **_reset_retriever_turn_state(state.tool_usage),
            "ask_retriever": state.tool_usage.get("ask_retriever", 0) + 1,
        }
        return TrajectoryState(
            turn_id=state.turn_id + 1,
            current_agent="retriever",
            history=state.history + [new_entry],
            tool_usage=new_usage,
            ask_retriever_limit=state.ask_retriever_limit,
            retriever_search_limit=state.retriever_search_limit,
            retriever_search_exchanges=[],  # fresh start for new retriever turn
            paper_block=state.paper_block,
            query=state.query,
            is_terminal=False,
            terminal_reason=None,
            hypotheses=None,
            trajectory_records=new_records,
            debug_entries=new_debug,
        )

    # ── generate_hypotheses ───────────────────────────────────────────────────
    if tool_name == "generate_hypotheses":
        hypotheses = tool_args.get("hypotheses", [])
        return TrajectoryState(
            turn_id=state.turn_id + 1,
            current_agent="generator",
            history=state.history,
            tool_usage=state.tool_usage,
            ask_retriever_limit=state.ask_retriever_limit,
            retriever_search_limit=state.retriever_search_limit,
            retriever_search_exchanges=state.retriever_search_exchanges,
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
