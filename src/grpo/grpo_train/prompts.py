"""
Prompt-building utilities for the scientific hypothesis generation pipeline.
============================================================================

``build_agent_prompt`` dispatches on the current step type from
``TrajectoryState.current_step`` and builds the appropriate ChatML prompt.

Each step type has a predictable context:

  retriever_search    – shows the research query, paper block, and (if this is
                        a follow-up search) the generator's question. If prior
                        searches exist in this retriever episode, shows them too.
  retriever_message   – shows the research query (or generator question), paper
                        block, and all search results from the current episode.
  generator_ask       – shows the research query and all retriever syntheses so far.
  generator_generate  – shows the research query and all retriever syntheses.

The retriever episodes are separated by generator_ask turns.  The number of
search results belonging to the current retriever episode is determined by
counting how many retriever_search steps have occurred since the last
generator_ask (or the start).
"""

from src.grpo.grpo_train.agents import AgentPrompts
from src.grpo.grpo_train.state import TrajectoryState


# ── ChatML helpers ────────────────────────────────────────────────────────────


def _fmt(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}\n<|im_end|>\n"


def _build_prompt(system: str, user: str) -> str:
    """Full ChatML prompt ready for generation (ends at <|im_start|>assistant)."""
    return _fmt("system", system) + _fmt("user", user) + "<|im_start|>assistant\n"


# ── episode helpers ───────────────────────────────────────────────────────────


def _current_episode_index(state: TrajectoryState) -> int:
    """Return which retriever episode we are in (0-based)."""
    return len(state.generator_questions)


def _search_results_for_episode(state: TrajectoryState, episode: int) -> list:
    """
    Return the search results that belong to the given retriever episode.

    The number of searches per episode equals the number of consecutive
    ``retriever_search`` steps before the first ``retriever_message`` in the
    step sequence (this equals ``retriever_search_limit``).
    """
    searches_per_episode = 0
    for s in state.step_sequence:
        if s == "retriever_search":
            searches_per_episode += 1
        elif s == "retriever_message":
            break

    start = episode * searches_per_episode
    end = start + searches_per_episode
    return state.search_results[start:end]


# ── step-specific prompt builders ─────────────────────────────────────────────


def _build_retriever_search_prompt(state: TrajectoryState) -> str:
    episode = _current_episode_index(state)
    episode_results = _search_results_for_episode(state, episode)

    if episode == 0:
        # Initial retriever turn — search to answer the main query
        system = AgentPrompts.retriever_search_system()
        user = (
            f"Research query:\n{state.query}\n\nAvailable papers:\n{state.paper_block}"
        )
    else:
        # Follow-up retriever turn — search to answer the generator's question
        question = state.generator_questions[episode - 1]
        system = AgentPrompts.retriever_search_followup_system()
        user = (
            f"Generator's question:\n{question}\n\n"
            f"Available papers:\n{state.paper_block}"
        )

    # If we already ran searches in this episode, show them so the model can
    # pick a different/refined query
    user_parts = [user]
    for i, result in enumerate(episode_results, 1):
        user_parts.append(f"\n[Previous search {i} result]\n{result}")

    return _build_prompt(system, "\n".join(user_parts))


def _build_retriever_message_prompt(state: TrajectoryState) -> str:
    episode = _current_episode_index(state)
    episode_results = _search_results_for_episode(state, episode)

    if episode == 0:
        system = AgentPrompts.retriever_message_system()
        user = (
            f"Research query:\n{state.query}\n\nAvailable papers:\n{state.paper_block}"
        )
    else:
        question = state.generator_questions[episode - 1]
        system = AgentPrompts.retriever_message_followup_system()
        user = (
            f"Generator's question:\n{question}\n\n"
            f"Available papers:\n{state.paper_block}"
        )

    # Append the search results gathered in this episode
    result_block = (
        "\n\n".join(
            f"[Search result {i}]\n{r}" for i, r in enumerate(episode_results, 1)
        )
        if episode_results
        else "(No search results available.)"
    )

    user = user + f"\n\nSearch results:\n{result_block}"
    return _build_prompt(system, user)


def _build_generator_ask_prompt(state: TrajectoryState) -> str:
    system = AgentPrompts.generator_ask_system()

    user_parts = [f"Research query:\n{state.query}"]
    for i, msg in enumerate(state.retriever_messages, 1):
        user_parts.append(f"\n[Retriever synthesis {i}]\n{msg}")

    return _build_prompt(system, "\n".join(user_parts))


def _build_generator_generate_prompt(state: TrajectoryState) -> str:
    system = AgentPrompts.generator_generate_system()

    user_parts = [f"Research query:\n{state.query}"]
    for i, msg in enumerate(state.retriever_messages, 1):
        label = (
            "Initial retriever synthesis"
            if i == 1
            else f"Additional retriever context {i - 1}"
        )
        user_parts.append(f"\n[{label}]\n{msg}")

    return _build_prompt(system, "\n".join(user_parts))


# ── main entry point ──────────────────────────────────────────────────────────


def build_agent_prompt(state: TrajectoryState) -> str:
    """Build the full ChatML prompt for the current step."""
    step = state.current_step
    if step == "retriever_search":
        return _build_retriever_search_prompt(state)
    if step == "retriever_message":
        return _build_retriever_message_prompt(state)
    if step == "generator_ask":
        return _build_generator_ask_prompt(state)
    if step == "generator_generate":
        return _build_generator_generate_prompt(state)
    raise ValueError(f"Unknown step type: {step!r}")
