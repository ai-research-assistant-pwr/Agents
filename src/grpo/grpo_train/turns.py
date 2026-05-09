"""
Turn functions for the scientific hypothesis generation pipeline.
=================================================================

Agents communicate exclusively via structured tool calls.  Every LLM response
is expected to contain optional <think>…</think> reasoning followed by exactly
one <tool_call>{"name": "…", "arguments": {…}}</tool_call> block.

Tool availability
-----------------
  Retriever  : send_to_generator(message)
  Generator  : generate_hypotheses(hypotheses)
               ask_retriever(question)   ← limited to 1 use per trajectory;
                                           removed from the prompt once exhausted

State
-----
``TrajectoryState`` is a plain dataclass treated as immutable: every transition
creates a new instance.  Helper ``apply_tool_call`` returns the next state given
the current state and a validated tool invocation; ``execute_turn`` orchestrates
one full LLM call and delegates to it.
"""

import json
import re as _re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.grpo.grpo_train.agents import AgentPrompts
from src.grpo.grpo_train.rewards import (
    _parse_hypotheses,
    get_hypothesis_and_label_embeddings,
    embedding_similarity_reward_from_embs,
    hypothesis_diversity_reward,
)


# ── low-level helpers ─────────────────────────────────────────────────────────


def _fmt(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}\n<|im_end|>\n"


def build_prompt(system: str, user: str) -> str:
    """Full ChatML prompt ready for generation (ends at <|im_start|>assistant)."""
    return _fmt("system", system) + _fmt("user", user) + "<|im_start|>assistant\n"


def tokenize(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][
        0
    ].tolist()


def extract_rollout_log_probs(
    response_output,
    input_ids: List[int],
    output_ids: List[int],
    sampling_params,
) -> Optional[List[float]]:
    if getattr(sampling_params, "logprobs", None) is None:
        return None
    log_probs: List[float] = [0.0] * len(input_ids)
    if hasattr(response_output, "logprobs") and response_output.logprobs is not None:
        for i, logprob_dict in enumerate(response_output.logprobs):
            if i < len(output_ids) and output_ids[i] in logprob_dict:
                log_probs.append(logprob_dict[output_ids[i]].logprob)
            else:
                log_probs.append(0.0)
    else:
        log_probs.extend([0.0] * len(output_ids))
    return log_probs


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks from model output."""
    return _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()


# ── tool-call parsing ─────────────────────────────────────────────────────────

_TOOL_CALL_RE = _re.compile(r"<tool_call>(.*?)</tool_call>", _re.DOTALL)


def parse_tool_call(output: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Extract ``(tool_name, arguments)`` from the model's raw output.

    Returns ``None`` if no valid ``<tool_call>`` block is found or the JSON
    inside is malformed.
    """
    match = _TOOL_CALL_RE.search(output)
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
        name = data["name"]
        args = data.get("arguments", {})
        return name, args
    except Exception:
        return None


# ── history entry ─────────────────────────────────────────────────────────────


@dataclass
class HistoryEntry:
    """Record of one completed tool call, stored in ``TrajectoryState.history``."""

    agent: str  # "retriever" | "generator"
    tool: str  # "send_to_generator" | "ask_retriever" | "generate_hypotheses"
    message: str  # content passed to the tool (synthesis text / question / etc.)


# ── trajectory state ──────────────────────────────────────────────────────────


@dataclass
class TrajectoryState:
    """
    Immutable snapshot of the trajectory at a given point.

    Treated as immutable: every state transition returns a *new* instance.
    Do not mutate fields in place.
    """

    turn_id: int
    current_agent: str  # "retriever" | "generator"
    history: List[HistoryEntry]  # all completed turns so far
    tool_usage: Dict[str, int]  # tool_name → number of times used
    paper_block: str  # static paper context (whole trajectory)
    query: str  # original research query
    is_terminal: bool
    terminal_reason: Optional[
        str
    ]  # "hypotheses_generated" | "bad_tool_call" | "max_turns"
    hypotheses: Optional[
        List[str]
    ]  # set when terminal_reason == "hypotheses_generated"
    trajectory_records: List[Dict[str, Any]]  # accumulated MARTI records
    debug_entries: List[Dict[str, Any]]  # accumulated debug records


def initial_state(query: str, paper_block: str) -> TrajectoryState:
    """Create the starting state for a new trajectory."""
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


# ── tool availability ─────────────────────────────────────────────────────────

_ASK_RETRIEVER_LIMIT: int = 1

_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "send_to_generator": {
        "description": (
            "Send your synthesis or focused answer to the generator agent. "
            "Use this when you have extracted the relevant information from the papers."
        ),
        "parameters": {
            "message": "string — your complete synthesis or answer text",
        },
    },
    "generate_hypotheses": {
        "description": (
            "Output the final list of scientific hypotheses. "
            "Use this when you have enough information to generate well-grounded hypotheses."
        ),
        "parameters": {
            "hypotheses": (
                "array of strings — each element is one complete, self-contained hypothesis"
            ),
        },
    },
    "ask_retriever": {
        "description": (
            "Ask the retriever a single focused follow-up question to fill a critical "
            f"evidence gap. Allowed at most {_ASK_RETRIEVER_LIMIT} time(s) per trajectory."
        ),
        "parameters": {
            "question": "string — your precise, specific question for the retriever",
        },
    },
}


def available_tools(state: TrajectoryState) -> List[str]:
    """Return the list of tool names the current agent may call."""
    if state.current_agent == "retriever":
        return ["send_to_generator"]

    # generator
    tools = ["generate_hypotheses"]
    if state.tool_usage.get("ask_retriever", 0) < _ASK_RETRIEVER_LIMIT:
        tools.append("ask_retriever")
    return tools


# ── prompt building ───────────────────────────────────────────────────────────


def _build_tool_section(tools: List[str]) -> str:
    """Render the tool-call instructions block for the given tool names."""
    lines = [
        "## Tool Use\n",
        "After your thinking, you MUST output exactly one tool call using this format:\n",
        '<tool_call>{"name": "<tool_name>", "arguments": {<arguments as JSON>}}</tool_call>\n',
        "Do not output anything after the closing </tool_call> tag.\n",
        "\n### Available tools:\n",
    ]
    for name in tools:
        schema = _TOOL_SCHEMAS[name]
        param_str = json.dumps(schema["parameters"], indent=4)
        lines.append(f"**{name}**")
        lines.append(f"  {schema['description']}")
        lines.append(f"  Parameters:\n{param_str}\n")
    return "\n".join(lines)


def _build_retriever_prompt(state: TrajectoryState) -> str:
    tools = available_tools(state)
    tool_section = _build_tool_section(tools)

    # detect whether we're responding to a generator question
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
    tool_section = _build_tool_section(tools)

    retriever_outputs = [e for e in state.history if e.tool == "send_to_generator"]
    ask_retriever_exhausted = (
        state.tool_usage.get("ask_retriever", 0) >= _ASK_RETRIEVER_LIMIT
    )

    # choose system prompt variant
    if ask_retriever_exhausted or len(retriever_outputs) >= 2:
        # must generate hypotheses now
        system = AgentPrompts.generator_hypothesize_system() + "\n\n" + tool_section
    else:
        # can still ask a question or hypothesize
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
    """Build the full ChatML prompt for the current agent turn."""
    if state.current_agent == "retriever":
        return _build_retriever_prompt(state)
    return _build_generator_prompt(state)


# ── record construction ───────────────────────────────────────────────────────


def make_turn_record(
    turn_id: int,
    agent_name: str,
    role: str,
    prompt: str,
    output: str,
    input_ids: List[int],
    output_ids: List[int],
    sampling_params,
    resp_output,
    reward: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "turn_id": turn_id,
        "agent_index": 0,
        "agent_id": agent_name,
        "agent_name": agent_name,
        "agent_role": role,
        "agent_input": prompt,
        "agent_output": output,
        "output_ids": output_ids,
        "sequence_ids": input_ids + output_ids,
        "rollout_log_prob": extract_rollout_log_probs(
            resp_output, input_ids, output_ids, sampling_params
        ),
        "reward": reward,
        "metadata": metadata or {},
    }


def make_debug_entry(
    turn_id: int,
    role: str,
    prompt: str,
    output: str,
    reward: float,
    input_ids: List[int],
    output_ids: List[int],
    tool_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "turn_id": turn_id,
        "role": role,
        "input": prompt,
        "output": output,
        "reward": reward,
        "tool_called": tool_name,
        "n_input_tokens": len(input_ids),
        "n_output_tokens": len(output_ids),
    }
    if extra:
        entry.update(extra)
    return entry


# ── state transitions ─────────────────────────────────────────────────────────


def _terminal_bad_call(
    state: TrajectoryState,
    turn_record: Dict[str, Any],
    debug_entry: Dict[str, Any],
) -> TrajectoryState:
    """Return a terminal state representing a malformed / disallowed tool call."""
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


def apply_tool_call(
    state: TrajectoryState,
    tool_name: str,
    tool_args: Dict[str, Any],
    turn_record: Dict[str, Any],
    debug_entry: Dict[str, Any],
) -> TrajectoryState:
    """
    Pure transition function.  Given the current state and a *validated* tool
    invocation, return the next ``TrajectoryState``.
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

    # unknown tool name (should not be reached after validation in execute_turn)
    return _terminal_bad_call(state, turn_record, debug_entry)


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


# ── execute turn ──────────────────────────────────────────────────────────────


async def execute_turn(
    state: TrajectoryState,
    llm,
    tokenizer,
    sampling_params,
    agent_name: str,
    label: str,
    embed_host: str,
    embed_port: int,
    similarity_weight: float = 0.7,
    diversity_weight: float = 0.3,
) -> TrajectoryState:
    """
    Execute one trajectory turn.

    Builds the prompt for the current agent, calls the LLM, parses the tool
    call, validates it, and delegates to ``apply_tool_call`` to return the next
    state.  Terminates with ``terminal_reason="bad_tool_call"`` and reward 0
    if the response is missing or has a disallowed tool call.
    """
    role = state.current_agent
    tools = available_tools(state)
    prompt = build_agent_prompt(state)

    input_ids = tokenize(tokenizer, prompt)
    resp = await llm.generate_async.remote(
        prompt_ids=input_ids, sampling_params=sampling_params
    )
    output: str = resp.outputs[0].text
    output_ids: List[int] = list(resp.outputs[0].token_ids)

    # ── parse tool call ───────────────────────────────────────────────────────
    parsed = parse_tool_call(output)

    if parsed is None:
        turn_record = make_turn_record(
            state.turn_id,
            agent_name,
            role,
            prompt,
            output,
            input_ids,
            output_ids,
            sampling_params,
            resp.outputs[0],
            reward=0.0,
        )
        debug_entry = make_debug_entry(
            state.turn_id,
            role,
            prompt,
            output,
            reward=0.0,
            input_ids=input_ids,
            output_ids=output_ids,
            tool_name=None,
            extra={"error": "missing_tool_call"},
        )
        return _terminal_bad_call(state, turn_record, debug_entry)

    tool_name, tool_args = parsed

    # ── validate tool is allowed this turn ────────────────────────────────────
    if tool_name not in tools:
        turn_record = make_turn_record(
            state.turn_id,
            agent_name,
            role,
            prompt,
            output,
            input_ids,
            output_ids,
            sampling_params,
            resp.outputs[0],
            reward=0.0,
        )
        debug_entry = make_debug_entry(
            state.turn_id,
            role,
            prompt,
            output,
            reward=0.0,
            input_ids=input_ids,
            output_ids=output_ids,
            tool_name=tool_name,
            extra={"error": f"tool_not_allowed: {tool_name} not in {tools}"},
        )
        return _terminal_bad_call(state, turn_record, debug_entry)

    # ── compute reward (only for generate_hypotheses) ─────────────────────────
    reward = 0.0
    extra_debug: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    if tool_name == "generate_hypotheses":
        hypotheses = tool_args.get("hypotheses", [])
        reward, sim_score, div_score = await compute_final_reward(
            hypotheses,
            label,
            embed_host,
            embed_port,
            similarity_weight,
            diversity_weight,
        )
        extra_debug = {
            "similarity_score": sim_score,
            "diversity_score": div_score,
            "similarity_weight": similarity_weight,
            "diversity_weight": diversity_weight,
        }
        metadata = {"label": label}

    turn_record = make_turn_record(
        state.turn_id,
        agent_name,
        role,
        prompt,
        output,
        input_ids,
        output_ids,
        sampling_params,
        resp.outputs[0],
        reward=reward,
        metadata=metadata,
    )
    debug_entry = make_debug_entry(
        state.turn_id,
        role,
        prompt,
        output,
        reward=reward,
        input_ids=input_ids,
        output_ids=output_ids,
        tool_name=tool_name,
        extra=extra_debug if extra_debug else None,
    )

    return apply_tool_call(state, tool_name, tool_args, turn_record, debug_entry)
