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
import re
import random

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


# ── emergent communication helper──────────────────────────────────────────────

MASK_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b")

def _inject_noise(
    text: str,
    noise_prob: float,
    mask_token: str = "[MASK]",
    span_probability: float = 0.15,
    max_span_length: int = 3,
) -> str:
    if noise_prob <= 0.0:
        return text

    matches = list(MASK_PATTERN.finditer(text))
    if not matches:
        return text

    output = []
    last_end = 0
    i = 0

    while i < len(matches):
        match = matches[i]

        output.append(text[last_end:match.start()])

        if random.random() < noise_prob:
            span_len = 1

            if random.random() < span_probability:
                span_len = random.randint(1, max_span_length)

            output.append(mask_token)

            end_idx = min(i + span_len - 1, len(matches) - 1)
            last_end = matches[end_idx].end()
            i = end_idx + 1
        else:
            output.append(match.group(0))
            last_end = match.end()
            i += 1

    output.append(text[last_end:])
    return "".join(output)


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
    t_start = time.time()

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
    paper_block = _format_papers(papers, max_papers=6)

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

    apply_length_penalty: bool = str(
        _wargs.get("apply_length_penalty", kwargs.get("apply_length_penalty", "False"))
    ).lower() == "true"
    length_penalty_lambda: float = float(
        _wargs.get("length_penalty_lambda", kwargs.get("length_penalty_lambda", 0.001))
    )

    apply_channel_noise: bool = str(
        _wargs.get("apply_channel_noise", kwargs.get("apply_channel_noise", "False"))
    ).lower() == "true"
    noise_probability: float = float(
        _wargs.get("noise_probability", kwargs.get("noise_probability", 0.1))
    )

    t0 = await turn_0_retriever_synthesize(
        llm, tokenizer, sp, agent_name, prompt, paper_block
    )

    t0_message = (
        _inject_noise(t0.output_content, noise_probability)
        if apply_channel_noise
        else t0.output_content
    )

    t1 = await turn_1_generator_ask(
        llm, tokenizer, sp, agent_name, prompt, t0_message
    )

    t2 = await turn_2_retriever_refine(
        llm, tokenizer, sp, agent_name, t1.output_content, paper_block
    )

    t2_message = (
        _inject_noise(t2.output_content, noise_probability)
        if apply_channel_noise
        else t2.output_content
    )

    t3 = await turn_3_generator_hypothesize(
        llm,
        tokenizer,
        sp,
        agent_name,
        prompt,
        t0_message,
        t2_message,
        label,
        embed_host,
        embed_port,
        similarity_weight=similarity_weight,
        diversity_weight=diversity_weight,
    )

    total_reward = t3.reward
    trajectory = [
        t0.trajectory_record,
        t1.trajectory_record,
        t2.trajectory_record,
        t3.trajectory_record,
    ]
    reward_matrix = [t0.reward, t1.reward, t2.reward, t3.reward]

    debug_entries = [
        t0.debug_entry,
        t1.debug_entry,
        t2.debug_entry,
        t3.debug_entry,
    ]

    if apply_channel_noise:
        debug_entries[0]["noisy_output_content"] = t0_message
        debug_entries[2]["noisy_output_content"] = t2_message

    for i, record in enumerate(trajectory):
        step_reward = total_reward

        if apply_length_penalty and record["agent_role"] == "retriever":
            clean_message = debug_entries[i]["output_content"]
            token_count = len(clean_message.split())
            penalty = length_penalty_lambda * token_count
            step_reward -= penalty

            debug_entries[i]["length_penalty"] = round(penalty, 4)
            debug_entries[i]["n_channel_tokens"] = token_count

        record["reward"] = step_reward
        debug_entries[i]["reward"] = step_reward

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
            turns=debug_entries,
        )

    retriever_avg_tokens = (
        debug_entries[0].get("n_output_tokens", 0)
        + debug_entries[2].get("n_output_tokens", 0)
    ) / 2.0

    retriever_avg_penalty = (
        debug_entries[0].get("length_penalty", 0)
        + debug_entries[2].get("length_penalty", 0)
    ) / 2.0

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "reward_matrix": reward_matrix,
        "final_reward": total_reward,
        "metrics": {
            "retriever_avg_tokens": retriever_avg_tokens,
            "retriever_avg_length_penalty": retriever_avg_penalty,
        },
    }