"""
Trajectory state for the scientific hypothesis generation pipeline.
===================================================================

The trajectory is now a **fixed-length sequence** of steps pre-computed from
``ask_retriever_limit`` (K) and ``retriever_search_limit`` (S).

Step sequence for K=1, S=1
---------------------------
  0  retriever_search    – retriever outputs a search query
  1  retriever_message   – retriever outputs synthesis to generator
  2  generator_ask       – generator outputs a follow-up question
  3  retriever_search    – retriever outputs a search query (for the question)
  4  retriever_message   – retriever outputs refined synthesis
  5  generator_generate  – generator outputs numbered hypothesis list

General formula: total_turns = (K + 1) * (S + 1) + (K + 1) = (K + 1) * (S + 2)

``TrajectoryState`` is treated as immutable; every transition returns a new instance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def build_step_sequence(
    ask_retriever_limit: int, retriever_search_limit: int
) -> List[str]:
    """
    Pre-compute the fixed ordered list of step types.

    For each retriever episode (initial + one per generator ask):
      - S x ``retriever_search`` turns
      - 1 x ``retriever_message`` turn
    Between retriever episodes (K times):
      - 1 x ``generator_ask`` turn
    At the end:
      - 1 x ``generator_generate`` turn
    """
    steps: List[str] = []
    n_retriever_episodes = ask_retriever_limit + 1
    for episode in range(n_retriever_episodes):
        for _ in range(retriever_search_limit):
            steps.append("retriever_search")
        steps.append("retriever_message")
        if episode < ask_retriever_limit:
            steps.append("generator_ask")
    steps.append("generator_generate")
    return steps


@dataclass
class TrajectoryState:
    """
    Snapshot of the trajectory at a given point in the pipeline.

    Treated as immutable: every state transition returns a *new* instance.

    Fields
    ------
    turn_id             : Zero-based index into ``step_sequence``.
    step_sequence       : Ordered list of step type strings (constant per trajectory).
    search_results      : Weaviate search results accumulated across all retriever_search
                          turns. Each entry is a formatted result string.
    retriever_messages  : Synthesis messages from each retriever_message turn.
    generator_questions : Questions from each generator_ask turn.
    paper_block         : Formatted paper context string (constant for trajectory).
    query               : Original research query (constant for trajectory).
    hypotheses          : Parsed hypothesis list; populated only after generator_generate.
    trajectory_records  : Accumulated MARTI-compatible turn records.
    debug_entries       : Accumulated human-readable debug records.
    """

    turn_id: int
    step_sequence: List[str]
    search_results: List[str]
    retriever_messages: List[str]
    generator_questions: List[str]
    paper_block: str
    query: str
    hypotheses: Optional[List[str]]
    trajectory_records: List[Dict[str, Any]]
    debug_entries: List[Dict[str, Any]]

    @property
    def current_step(self) -> Optional[str]:
        """The step type for the current turn, or None if the trajectory is complete."""
        if self.turn_id < len(self.step_sequence):
            return self.step_sequence[self.turn_id]
        return None

    @property
    def is_terminal(self) -> bool:
        return self.turn_id >= len(self.step_sequence)

    @property
    def current_agent(self) -> str:
        step = self.current_step
        if step in ("retriever_search", "retriever_message"):
            return "retriever"
        return "generator"


def initial_state(
    query: str,
    paper_block: str,
    ask_retriever_limit: int = 1,
    retriever_search_limit: int = 1,
) -> TrajectoryState:
    """Return the starting state for a new trajectory."""
    return TrajectoryState(
        turn_id=0,
        step_sequence=build_step_sequence(ask_retriever_limit, retriever_search_limit),
        search_results=[],
        retriever_messages=[],
        generator_questions=[],
        paper_block=paper_block,
        query=query,
        hypotheses=None,
        trajectory_records=[],
        debug_entries=[],
    )
