"""
Prompt-building utilities for the scientific hypothesis generation pipeline.
============================================================================

``build_agent_prompt`` is the main entry point: it inspects the current
``TrajectoryState`` and constructs the full ChatML-formatted prompt (system +
user + assistant prefix) for whichever agent is about to act.

Internal helpers
----------------
  _build_retriever_prompt  – initial synthesis or focused refinement
  _build_generator_prompt  – ask-retriever variant or hypothesize variant
"""

import re as _re
from typing import Optional

from src.grpo.grpo_train.agents import AgentPrompts
from src.grpo.grpo_train.state import TrajectoryState
from src.grpo.grpo_train.tools import (
    ASK_RETRIEVER_LIMIT,
    available_tools,
    build_tool_section,
)


# ── ChatML formatting ─────────────────────────────────────────────────────────


def _fmt(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}\n<|im_end|>\n"


def build_prompt(system: str, user: str) -> str:
    """Full ChatML prompt ready for generation (ends at <|im_start|>assistant)."""
    return _fmt("system", system) + _fmt("user", user) + "<|im_start|>assistant\n"


# ── thinking-token stripping ──────────────────────────────────────────────────

_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks from model output."""
    return _THINK_RE.sub("", text).strip()


# ── agent-specific prompt builders ───────────────────────────────────────────


def _build_retriever_prompt(state: TrajectoryState) -> str:
    tools = available_tools(state)
    tool_section = build_tool_section(tools)

    # check whether we are responding to a generator follow-up question
    generator_question: Optional[str] = None
    for entry in reversed(state.history):
        if entry.tool == "ask_retriever":
            generator_question = entry.message
            break

    if generator_question is None:
        system = AgentPrompts.retriever_system() + "\n\n" + tool_section
        user = (
            f"Research query:\n{state.query}\n\nAvailable papers:\n{state.paper_block}"
        )
    else:
        system = AgentPrompts.retriever_refine_system() + "\n\n" + tool_section
        user = (
            f"The generator is asking:\n{generator_question}\n\n"
            f"Available papers (same pool):\n{state.paper_block}"
        )

    return build_prompt(system, user)


def _build_generator_prompt(state: TrajectoryState) -> str:
    tools = available_tools(state)
    tool_section = build_tool_section(tools)

    retriever_outputs = [e for e in state.history if e.tool == "send_to_generator"]
    ask_retriever_exhausted = (
        state.tool_usage.get("ask_retriever", 0) >= ASK_RETRIEVER_LIMIT
    )

    # use the hypothesize-only system prompt once ask_retriever is no longer available
    if ask_retriever_exhausted or len(retriever_outputs) >= 2:
        system = AgentPrompts.generator_hypothesize_system() + "\n\n" + tool_section
    else:
        system = AgentPrompts.generator_system() + "\n\n" + tool_section

    # build user message from accumulated retriever outputs
    if len(retriever_outputs) == 0:
        user = f"Research query:\n{state.query}"
    elif len(retriever_outputs) == 1:
        user = (
            f"Research query:\n{state.query}\n\n"
            f"Retriever synthesis:\n{retriever_outputs[0].message}"
        )
    else:
        user = (
            f"Research query:\n{state.query}\n\n"
            f"Retriever synthesis:\n{retriever_outputs[0].message}\n\n"
            f"Additional retriever context:\n{retriever_outputs[1].message}"
        )

    return build_prompt(system, user)


def build_agent_prompt(state: TrajectoryState) -> str:
    """Build the full ChatML prompt for the agent whose turn it is."""
    if state.current_agent == "retriever":
        return _build_retriever_prompt(state)
    return _build_generator_prompt(state)
