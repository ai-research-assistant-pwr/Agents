"""
Scientific Hypothesis Generation Workflow
==========================================
Fixed 4-turn pipeline – each turn is an independent trajectory record:

  Turn 0  [RETRIEVER]  query + up-to-5 paper summaries  →  initial synthesis
  Turn 1  [GENERATOR]  query + retriever synthesis       →  follow-up question
  Turn 2  [RETRIEVER]  generator question + same papers  →  focused clarification
  Turn 3  [GENERATOR]  query + both retriever outputs    →  numbered hypothesis list

Each turn is independent: it receives its own fresh prompt and its trajectory
record carries only that turn's ``sequence_ids`` (input_ids + output_ids) and
``output_ids``.  Prior turns' outputs are injected as plain text (thinking
tokens stripped) inside the next turn's user message.

Reward
------
  Turns 0–2 : 0.0  (no per-turn signal)
  Turn 3    : mean cosine similarity between each generated hypothesis and the
              ground-truth label, computed via the vLLM embedding server.
              Returns 0.0 if the output is not in the expected numbered-list format.

Data contract
-------------
  prompt   – raw user query string
  label    – ground-truth hypothesis (used in Turn 3 reward)
  metadata – dict with key "papers": list of {"id", "title", "summary"} dicts

kwargs (via workflow_args JSON or top-level)
--------------------------------------------
  embed_host        – hostname of the vLLM embedding server  (default: "localhost")
  embed_port        – port of the vLLM embedding server       (default: 8000)
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
from src.grpo.grpo_train.turns import (
    turn_0_retriever_synthesize,
    turn_1_generator_ask,
    turn_2_retriever_refine,
    turn_3_generator_hypothesize,
)

logger = init_logger(__name__)
logger.setLevel("WARN")

DEBUG: bool = True

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
        "elapsed_s": elapsed_s,
        "turns": turns,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    logger.warning(f"[DEBUG] trajectory log written → {path}")


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
    Fixed 4-turn scientific hypothesis generation pipeline.

    Parameters
    ----------
    prompt   : The user research query.
    label    : Ground-truth hypothesis (used in Turn 3 reward).
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

    metadata = json.loads(json.loads(metadata))
    papers: List[Dict[str, str]] = (metadata or {}).get("papers", [])
    paper_block = _format_papers(papers, max_papers=5)

    # ── kwargs unpacking ──────────────────────────────────────────────────────
    # MARTI passes --workflow_args JSON as a single kwarg named "workflow_args",
    # not spread into **kwargs. Read from there first, fall back to top-level.
    _wargs: Dict[str, Any] = kwargs.get("workflow_args") or {}
    embed_host: str = _wargs.get("embed_host", kwargs.get("embed_host", "localhost"))
    embed_port: int = int(_wargs.get("embed_port", kwargs.get("embed_port", 8000)))
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

    # ── execute turns ─────────────────────────────────────────────────────────
    t0 = await turn_0_retriever_synthesize(
        llm, tokenizer, sp, agent_name, prompt, paper_block
    )
    t1 = await turn_1_generator_ask(
        llm, tokenizer, sp, agent_name, prompt, t0.output_content
    )
    t2 = await turn_2_retriever_refine(
        llm, tokenizer, sp, agent_name, t1.output_content, paper_block
    )
    t3 = await turn_3_generator_hypothesize(
        llm,
        tokenizer,
        sp,
        agent_name,
        prompt,
        t0.output_content,
        t2.output_content,
        label,
        embed_host,
        embed_port,
        similarity_weight=similarity_weight,
        diversity_weight=diversity_weight,
    )

    # ── assemble results ──────────────────────────────────────────────────────
    total_reward = t3.reward
    trajectory = [
        t0.trajectory_record,
        t1.trajectory_record,
        t2.trajectory_record,
        t3.trajectory_record,
    ]
    reward_matrix = [t0.reward, t1.reward, t2.reward, t3.reward]

    # propagate final reward to all trajectory records
    for record in trajectory:
        record["reward"] = total_reward

    elapsed = time.time() - t_start
    logger.warning(
        f"workflow done | turns=4 | reward={total_reward:.3f} | time={elapsed:.1f}s"
    )

    if DEBUG:
        _write_debug_log(
            debug_dir=debug_dir,
            prompt_id=prompt_id,
            prompt=prompt,
            label=label,
            total_reward=total_reward,
            elapsed_s=round(elapsed, 2),
            turns=[t0.debug_entry, t1.debug_entry, t2.debug_entry, t3.debug_entry],
        )

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "reward_matrix": reward_matrix,
        "final_reward": total_reward,
    }
