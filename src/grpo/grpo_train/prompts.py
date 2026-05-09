"""
Prompt-building utilities for the scientific hypothesis generation pipeline.
============================================================================

``build_agent_prompt`` is the main entry point: it inspects the current
``TrajectoryState`` and constructs the full ChatML-formatted prompt (system +
user + prior search exchanges + assistant prefix) for whichever agent is about
to act.

Retriever prompt structure
--------------------------
When the retriever has already called ``search_papers`` one or more times in
the current turn, those exchanges are injected *after* the initial user message
so the model sees its full prior reasoning:

  <|im_start|>system\\n…\\n<|im_end|>
  <|im_start|>user\\n…\\n<|im_end|>
  <|im_start|>assistant\\n{raw output incl. <think> + <tool_call>}\\n<|im_end|>
  <|im_start|>tool\\n{formatted search result}\\n<|im_end|>
  … repeated for each prior search_papers call …
  <|im_start|>assistant\\n          ← model generates here

This ensures the input_ids fed to the LLM include all prior tokens and the
output_ids for a turn contain *only* the newly generated tokens.
"""

from typing import Optional

from src.grpo.grpo_train.agents import AgentPrompts
from src.grpo.grpo_train.state import TrajectoryState
from src.grpo.grpo_train.tools import available_tools, build_tool_section


# ── ChatML helpers ────────────────────────────────────────────────────────────


def _fmt(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}\n<|im_end|>\n"


def build_prompt(system: str, user: str) -> str:
    """Full ChatML prompt ready for generation (ends at <|im_start|>assistant)."""
    return _fmt("system", system) + _fmt("user", user) + "<|im_start|>assistant\n"


# ── agent-specific prompt builders ───────────────────────────────────────────


def _build_retriever_prompt(state: TrajectoryState) -> str:
    tools = available_tools(state)
    tool_section = build_tool_section(tools, state)

    # detect whether we're responding to a generator follow-up question
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

    # base: system + user
    prompt = _fmt("system", system) + _fmt("user", user)

    # inject prior search_papers exchanges from the current retriever turn so the
    # model sees its full reasoning history (thinking tokens + tool calls included)
    for raw_llm_output, search_result in state.retriever_search_exchanges:
        prompt += _fmt("assistant", raw_llm_output)
        prompt += _fmt("tool", search_result)

    # assistant prefix — model generates from here
    prompt += "<|im_start|>assistant\n"
    return prompt


def _build_generator_prompt(state: TrajectoryState) -> str:
    tools = available_tools(state)
    tool_section = build_tool_section(tools, state)

    retriever_outputs = [e for e in state.history if e.tool == "send_to_generator"]
    ask_retriever_exhausted = (
        state.tool_usage.get("ask_retriever", 0) >= state.ask_retriever_limit
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
