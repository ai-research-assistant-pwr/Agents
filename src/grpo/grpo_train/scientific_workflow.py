"""
Scientific Hypothesis Generation Workflow
==========================================
Variable-length pipeline driven by tool calls.

Architecture
------------
  Both agents (retriever and generator) communicate exclusively via structured
  tool calls.  Every LLM response must contain one ``<tool_call>`` block after
  optional ``<think>`` reasoning:

      <think>…</think>
      <tool_call>{"name": "<tool>", "arguments": {…}}</tool_call>

  Retriever tools
  ~~~~~~~~~~~~~~~
    send_to_generator(message)  – forward synthesis/answer to the generator

  Generator tools
  ~~~~~~~~~~~~~~~
    generate_hypotheses(hypotheses)  – terminal; produces the final list
    ask_retriever(question)          – request a follow-up from the retriever
                                       (limited to 1 use per trajectory; removed
                                        from the prompt once the limit is reached)

Trajectory flow
---------------
  The pipeline starts with the retriever (turn 0) and alternates agents
  according to tool calls.  The loop terminates when:

    - the generator calls ``generate_hypotheses``  → reward computed
    - any agent produces a missing / malformed tool call               → reward 0
    - an agent calls a disallowed tool                                 → reward 0
    - ``max_turns`` is reached without a terminal tool call            → reward 0

  Every turn appends one record to the trajectory list; the final reward is
  propagated to all records at the end.

Reward
------
  Weighted combination of:
    similarity_weight × mean cosine similarity (hypotheses ↔ ground-truth label)
    diversity_weight  × pairwise diversity among hypotheses
  Requires ≥ 1 hypothesis; returns 0.0 on empty list or bad format.

Data contract
-------------
  prompt   – raw user query string
  label    – ground-truth hypothesis (used in terminal reward)
  metadata – dict with key "papers": list of {"id", "title", "summary"} dicts

kwargs (via workflow_args JSON or top-level)
--------------------------------------------
  embed_host        – hostname of the vLLM embedding server  (default: "localhost")
  embed_port        – port of the vLLM embedding server       (default: 8000)
  max_turns         – hard cap on trajectory length            (default: 10)
  debug_dir         – directory for per-trajectory JSON logs  (default: "./workflow_debug_logs")
  similarity_weight – weight for the embedding similarity reward (default: 0.7)
  diversity_weight  – weight for the hypothesis diversity reward  (default: 0.3)

Debug logging
-------------
  Set DEBUG = True to write one JSON file per trajectory into ``debug_dir``.
  Files are written atomically (temp file + rename).
"""

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from marti.utils.logging_utils import init_logger
from src.grpo.grpo_train.tools import search_weaviate
from src.grpo.grpo_train.turns import (
    TrajectoryState,
    execute_turn,
    initial_state,
)

logger = init_logger(__name__)
logger.setLevel("WARN")

DEBUG: bool = True

# ── feature flags ─────────────────────────────────────────────────────────────

# When True, paper context is retrieved live from Weaviate by embedding the
# user query, instead of being loaded from the dataset row's metadata field.
USE_WEAVIATE_CONTEXT: bool = False

# Number of papers to fetch from Weaviate when USE_WEAVIATE_CONTEXT is True.
WEAVIATE_TOP_N: int = 6

# ── paper-context helper ──────────────────────────────────────────────────────


def _format_papers(papers: List[Dict[str, str]], max_papers: int = 8) -> str:
    selected = papers[:max_papers]
    lines = []
    for i, p in enumerate(selected, 1):
        title = p.get("title", "Unknown title")
        summary = p.get("summary", p.get("abstract", "No summary available."))
        lines.append(f"[Paper {i}] {title}\n{summary.strip()}")
    return "\n\n".join(lines)


# ── debug helper ──────────────────────────────────────────────────────────────


def _write_debug_log(
    debug_dir: str,
    prompt_id: int,
    prompt: str,
    label: str,
    total_reward: float,
    elapsed_s: float,
    terminal_reason: Optional[str],
    turns: List[Dict[str, Any]],
) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    ts_ms = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    path = os.path.join(debug_dir, f"traj_{ts_ms}_{uid}.json")
    tmp_path = path + ".tmp"
    payload = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "label": label,
        "total_reward": total_reward,
        "terminal_reason": terminal_reason,
        "elapsed_s": elapsed_s,
        "turns": turns,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    logger.warning(f"[DEBUG] trajectory log written → {path}")


# ── terminal-state helpers ────────────────────────────────────────────────────


def _finalize_max_turns(state: TrajectoryState) -> TrajectoryState:
    """Return a copy of *state* marked as terminal due to max-turns exhaustion."""
    return TrajectoryState(
        turn_id=state.turn_id,
        current_agent=state.current_agent,
        history=state.history,
        tool_usage=state.tool_usage,
        paper_block=state.paper_block,
        query=state.query,
        is_terminal=True,
        terminal_reason="max_turns",
        hypotheses=None,
        trajectory_records=state.trajectory_records,
        debug_entries=state.debug_entries,
    )


def _extract_final_reward(state: TrajectoryState) -> float:
    """Return the reward from the last trajectory record, or 0.0 if none."""
    if state.trajectory_records:
        return state.trajectory_records[-1].get("reward", 0.0)
    return 0.0


# ── main workflow ─────────────────────────────────────────────────────────────


async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager=None,
    task: str = "scientific_hypothesis",
    metadata: Optional[Dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Variable-length scientific hypothesis generation pipeline.

    Parameters
    ----------
    prompt   : The user research query.
    label    : Ground-truth hypothesis (used in terminal reward).
    agents   : List with at least one agent dict containing 'llm', 'tokenizer',
               'sampling_params'.  A single model plays both roles.
    metadata : Must include key 'papers': list of dicts with 'id', 'title',
               'summary' fields.
    """
    t_start = time.time()

    # ── agent setup ──────────────────────────────────────────────────────────
    agent = agents[0]
    llm = agent["llm"]
    tokenizer = agent["tokenizer"]
    sp = agent["sampling_params"]
    agent_name: str = agent.get("agent_id", "shared_agent")

    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    if hasattr(sp, "stop"):
        sp.stop = stop_tokens
    elif isinstance(sp, dict):
        sp["stop"] = stop_tokens

    # ── kwargs unpacking ──────────────────────────────────────────────────────
    # MARTI passes --workflow_args JSON as a single kwarg named "workflow_args",
    # not spread into **kwargs. Read from there first, fall back to top-level.
    _wargs: Dict[str, Any] = kwargs.get("workflow_args") or {}
    embed_host: str = _wargs.get("embed_host", kwargs.get("embed_host", "localhost"))
    embed_port: int = int(_wargs.get("embed_port", kwargs.get("embed_port", 8000)))
    max_turns: int = int(_wargs.get("max_turns", kwargs.get("max_turns", 10)))
    debug_dir: str = _wargs.get(
        "debug_dir", kwargs.get("debug_dir", "./workflow_debug_logs")
    )
    similarity_weight: float = float(
        _wargs.get("similarity_weight", kwargs.get("similarity_weight", 0.7))
    )
    diversity_weight: float = float(
        _wargs.get("diversity_weight", kwargs.get("diversity_weight", 0.3))
    )
    prompt_id: int = kwargs.get("prompt_id", 0)

    # ── paper context ─────────────────────────────────────────────────────────
    if USE_WEAVIATE_CONTEXT:
        weaviate_url: str = _wargs.get(
            "weaviate_url", kwargs.get("weaviate_url", "http://localhost:8080")
        )
        papers = search_weaviate(prompt, WEAVIATE_TOP_N, weaviate_url)
        paper_block = _format_papers(papers, max_papers=WEAVIATE_TOP_N)
    else:
        metadata = json.loads(json.loads(metadata))
        papers = (metadata or {}).get("papers", [])
        paper_block = _format_papers(papers, max_papers=WEAVIATE_TOP_N)

    # ── trajectory loop ───────────────────────────────────────────────────────
    state: TrajectoryState = initial_state(prompt, paper_block)

    while not state.is_terminal and state.turn_id < max_turns:
        state = await execute_turn(
            state=state,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sp,
            agent_name=agent_name,
            label=label,
            embed_host=embed_host,
            embed_port=embed_port,
            similarity_weight=similarity_weight,
            diversity_weight=diversity_weight,
        )

    if not state.is_terminal:
        state = _finalize_max_turns(state)

    # ── assemble results ──────────────────────────────────────────────────────
    total_reward = _extract_final_reward(state)
    trajectory = list(state.trajectory_records)

    # propagate final reward to all trajectory records
    for record in trajectory:
        record["reward"] = total_reward

    reward_matrix = [r.get("reward", 0.0) for r in trajectory]

    elapsed = time.time() - t_start
    logger.warning(
        f"workflow done | turns={state.turn_id} | reason={state.terminal_reason} "
        f"| reward={total_reward:.3f} | time={elapsed:.1f}s"
    )

    if DEBUG:
        _write_debug_log(
            debug_dir=debug_dir,
            prompt_id=prompt_id,
            prompt=prompt,
            label=label,
            total_reward=total_reward,
            elapsed_s=round(elapsed, 2),
            terminal_reason=state.terminal_reason,
            turns=list(state.debug_entries),
        )

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "reward_matrix": reward_matrix,
        "final_reward": total_reward,
    }
