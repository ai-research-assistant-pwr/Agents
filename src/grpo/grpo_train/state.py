"""
Trajectory state for the scientific hypothesis generation pipeline.
===================================================================

``TrajectoryState`` is a plain dataclass treated as immutable: every state
transition creates a new instance.  ``HistoryEntry`` records a single completed
tool call and is stored in ``TrajectoryState.history``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HistoryEntry:
    """Record of one completed tool call, stored in ``TrajectoryState.history``."""

    agent: str  # "retriever" | "generator"
    tool: str  # "send_to_generator" | "ask_retriever" | "generate_hypotheses"
    message: str  # content passed to the tool (synthesis / question / hypotheses)


@dataclass
class TrajectoryState:
    """
    Snapshot of the trajectory at a given point in the pipeline.

    Treated as immutable: every state transition returns a *new* instance.
    Do not mutate fields in place.

    Fields
    ------
    turn_id                    : Zero-based counter incremented after each LLM call.
    current_agent              : Which agent acts next — "retriever" or "generator".
    history                    : Ordered list of completed ``HistoryEntry`` records
                                 (inter-agent messages only; search exchanges are not
                                 included here).
    tool_usage                 : Map of tool_name → cumulative uses this trajectory.
    ask_retriever_limit        : Max number of ask_retriever calls allowed per trajectory
                                 (configurable via workflow_args).
    retriever_search_limit     : Max number of search_papers calls allowed per retriever
                                 turn (resets to 0 at the start of each new retriever
                                 turn, configurable via workflow_args).
    retriever_search_exchanges : List of (raw_llm_output, formatted_search_result) pairs
                                 accumulated during the *current* retriever turn.  Each
                                 pair is injected as an assistant+tool message block in
                                 the next retriever prompt so the model sees its full
                                 prior reasoning.  Reset to [] when a new retriever turn
                                 begins (send_to_generator or ask_retriever transition).
    paper_block                : Formatted paper context string (constant for trajectory).
    query                      : Original research query (constant for trajectory).
    is_terminal                : True once the trajectory has ended for any reason.
    terminal_reason            : One of "hypotheses_generated", "bad_tool_call",
                                 "max_turns", or None while still running.
    hypotheses                 : Parsed list of hypotheses; populated only when
                                 terminal_reason == "hypotheses_generated".
    trajectory_records         : Accumulated MARTI-compatible turn records.
    debug_entries              : Accumulated human-readable debug records.
    """

    turn_id: int
    current_agent: str
    history: List[HistoryEntry]
    tool_usage: Dict[str, int]
    ask_retriever_limit: int
    retriever_search_limit: int
    retriever_search_exchanges: List[Tuple[str, str]]  # (raw_llm_output, search_result)
    paper_block: str
    query: str
    is_terminal: bool
    terminal_reason: Optional[str]
    hypotheses: Optional[List[str]]
    trajectory_records: List[Dict[str, Any]]
    debug_entries: List[Dict[str, Any]]


def initial_state(
    query: str,
    paper_block: str,
    ask_retriever_limit: int = 1,
    retriever_search_limit: int = 3,
) -> TrajectoryState:
    """Return the starting state for a new trajectory (retriever goes first)."""
    return TrajectoryState(
        turn_id=0,
        current_agent="retriever",
        history=[],
        tool_usage={"ask_retriever": 0, "search_papers": 0},
        ask_retriever_limit=ask_retriever_limit,
        retriever_search_limit=retriever_search_limit,
        retriever_search_exchanges=[],
        paper_block=paper_block,
        query=query,
        is_terminal=False,
        terminal_reason=None,
        hypotheses=None,
        trajectory_records=[],
        debug_entries=[],
    )
