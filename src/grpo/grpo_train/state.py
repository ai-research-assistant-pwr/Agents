"""
Trajectory state for the scientific hypothesis generation pipeline.
===================================================================

``TrajectoryState`` is a plain dataclass treated as immutable: every state
transition creates a new instance.  ``HistoryEntry`` records a single completed
tool call and is stored in ``TrajectoryState.history``.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
    turn_id           : Zero-based index of the *next* turn to execute.
    current_agent     : Which agent acts next — "retriever" or "generator".
    history           : Ordered list of completed ``HistoryEntry`` records.
    tool_usage        : Map of tool_name → number of times used this trajectory.
    paper_block       : Formatted paper context string (constant for the trajectory).
    query             : Original research query (constant for the trajectory).
    is_terminal       : True once the trajectory has ended (any reason).
    terminal_reason   : One of "hypotheses_generated", "bad_tool_call", "max_turns",
                        or None while still running.
    hypotheses        : Parsed list of hypotheses; set only when
                        terminal_reason == "hypotheses_generated".
    trajectory_records: Accumulated MARTI-compatible turn records.
    debug_entries     : Accumulated human-readable debug records.
    """

    turn_id: int
    current_agent: str
    history: List[HistoryEntry]
    tool_usage: Dict[str, int]
    paper_block: str
    query: str
    is_terminal: bool
    terminal_reason: Optional[str]
    hypotheses: Optional[List[str]]
    trajectory_records: List[Dict[str, Any]]
    debug_entries: List[Dict[str, Any]]


def initial_state(query: str, paper_block: str) -> TrajectoryState:
    """Return the starting state for a new trajectory (retriever goes first)."""
    return TrajectoryState(
        turn_id=0,
        current_agent="retriever",
        history=[],
        tool_usage={"ask_retriever": 0},
        paper_block=paper_block,
        query=query,
        is_terminal=False,
        terminal_reason=None,
        hypotheses=None,
        trajectory_records=[],
        debug_entries=[],
    )
