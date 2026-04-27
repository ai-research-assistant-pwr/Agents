"""
Scientific Hypothesis Generation Workflow
==========================================
Fixed 4-turn pipeline (no dynamic branching, no XML tags):

  Turn 0  [RETRIEVER]  query + up-to-8 paper summaries  →  ~100-word synthesis
  Turn 1  [GENERATOR]  query + retriever synthesis       →  question asking for more context
  Turn 2  [RETRIEVER]  generator question + same papers  →  focused follow-up (~100 words)
  Turn 3  [GENERATOR]  query + both retriever outputs    →  numbered hypothesis list

Rewards
-------
  Retriever (turns 0 & 2) : gaussian centred at 100 words on raw output
  Generator turn 1        : 1.0 if output contains '?', else 0.0
  Generator turn 3        : format score (numbered items) + count score (peak at 3)

Data contract
-------------
  prompt   – raw user query string
  label    – ground-truth hypothesis (for logging; not used in reward)
  metadata – dict with key "papers": list of {"id", "title", "summary"} dicts
"""

import time
from typing import Any, Dict, List, Optional

from marti.utils.logging_utils import init_logger
from src.grpo.grpo_train.agents import AgentPrompts
from src.grpo.grpo_train.rewards import (
    retriever_word_count_reward,
    generator_request_format_reward,
    generator_hypothesis_reward,
)


logger = init_logger(__name__)
logger.setLevel("WARN")

# ──────────────────────────────────────────────────────────────────────────────
# Token helpers
# ──────────────────────────────────────────────────────────────────────────────


def _fmt(role: str, content: str) -> str:
    """Format a single ChatML turn."""
    return f"<|im_start|>{role}\n{content}\n<|im_end|>\n"


def _build_prompt(system: str, user: str) -> str:
    """Full ChatML prompt ready for generation (ends at <|im_start|>assistant)."""
    return _fmt("system", system) + _fmt("user", user) + "<|im_start|>assistant\n"


def _extend_prompt(base: str, assistant_reply: str, system: str, user: str) -> str:
    """Close the previous assistant turn and open a new system/user/assistant block."""
    return (
        base
        + assistant_reply
        + "<|im_end|>\n"
        + _fmt("system", system)
        + _fmt("user", user)
        + "<|im_start|>assistant\n"
    )


def _tokenize(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][
        0
    ].tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Paper context helpers
# ──────────────────────────────────────────────────────────────────────────────


def _format_papers(papers: List[Dict[str, str]], max_papers: int = 8) -> str:
    """Render a list of paper dicts into a readable block."""
    selected = papers[:max_papers]
    lines = []
    for i, p in enumerate(selected, 1):
        title = p.get("title", "Unknown title")
        summary = p.get("summary", p.get("abstract", "No summary available."))
        lines.append(f"[Paper {i}] {title}\n{summary.strip()}")
    return "\n\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────────


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
    label    : Ground-truth hypothesis (for logging; not used in reward).
    agents   : List with at least one agent dict containing 'llm', 'tokenizer',
               'sampling_params'.  A single model plays both roles.
    metadata : Must include key 'papers': list of dicts with 'id', 'title',
               'summary' fields, pre-loaded from 11_neo4j_papers.csv.
    """
    t_start = time.time()

    # ── agent setup ──────────────────────────────────────────────────────────
    agent = agents[0]
    llm = agent["llm"]
    tokenizer = agent["tokenizer"]
    sp = agent["sampling_params"]

    # Ensure model stops cleanly at the end of each assistant turn
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    if hasattr(sp, "stop"):
        sp.stop = stop_tokens
    elif isinstance(sp, dict):
        sp["stop"] = stop_tokens

    max_length: int = kwargs.get("max_length", 2048)
    papers: List[Dict[str, str]] = (metadata or {}).get("papers", [])
    paper_block = _format_papers(papers, max_papers=8)

    # ── trajectory bookkeeping ────────────────────────────────────────────────
    all_output_ids: List[int] = []
    action_mask: List[int] = []

    # Seed sequence with the first prompt (no loss on the initial prompt tokens)
    turn0_prompt = _build_prompt(
        system=AgentPrompts.retriever_system(),
        user=f"Research query:\n{prompt}\n\nAvailable papers:\n{paper_block}",
    )
    sequence_ids = _tokenize(tokenizer, turn0_prompt)
    action_mask.extend([0] * len(sequence_ids))
    all_output_ids.extend(
        [0] * len(sequence_ids)
    )  # placeholder, will not be used for loss

    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Turn 0 – RETRIEVER: synthesise paper summaries into a concise message
    # ──────────────────────────────────────────────────────────────────────────
    resp0 = await llm.generate_async.remote(prompt_ids=sequence_ids, sampling_params=sp)
    out0: str = resp0.outputs[0].text
    ids0: List[int] = list(resp0.outputs[0].token_ids)

    r0 = retriever_word_count_reward(out0)
    total_reward += r0
    trajectory.append(
        {
            "turn_id": 0,
            "role": "retriever",
            "prompt": turn0_prompt,
            "output": out0,
            "reward": r0,
        }
    )

    sequence_ids.extend(ids0)
    all_output_ids.extend(ids0)
    action_mask.extend([1] * len(ids0))

    retriever_msg_1 = out0.strip()

    # ──────────────────────────────────────────────────────────────────────────
    # Turn 1 – GENERATOR: ask for additional context
    # ──────────────────────────────────────────────────────────────────────────
    turn1_env = (
        "<|im_end|>\n"
        + _fmt("system", AgentPrompts.generator_ask_system())
        + _fmt(
            "user",
            f"Research query:\n{prompt}\n\nRetriever synthesis:\n{retriever_msg_1}",
        )
        + "<|im_start|>assistant\n"
    )
    turn1_env_ids = _tokenize(tokenizer, turn1_env)
    sequence_ids.extend(turn1_env_ids)
    all_output_ids.extend(turn1_env_ids)
    action_mask.extend([0] * len(turn1_env_ids))

    resp1 = await llm.generate_async.remote(prompt_ids=sequence_ids, sampling_params=sp)
    out1: str = resp1.outputs[0].text
    ids1: List[int] = list(resp1.outputs[0].token_ids)

    r1 = generator_request_format_reward(out1)
    total_reward += r1
    trajectory.append(
        {
            "turn_id": 1,
            "role": "generator",
            "output": out1,
            "reward": r1,
        }
    )

    sequence_ids.extend(ids1)
    all_output_ids.extend(ids1)
    action_mask.extend([1] * len(ids1))

    gen_request = out1.strip()

    # ──────────────────────────────────────────────────────────────────────────
    # Turn 2 – RETRIEVER: respond to generator's request with focused context
    # ──────────────────────────────────────────────────────────────────────────
    turn2_env = (
        "<|im_end|>\n"
        + _fmt("system", AgentPrompts.retriever_system())
        + _fmt(
            "user",
            f"The generator is asking:\n{gen_request}\n\n"
            f"Available papers (same pool):\n{paper_block}",
        )
        + "<|im_start|>assistant\n"
    )
    turn2_env_ids = _tokenize(tokenizer, turn2_env)
    sequence_ids.extend(turn2_env_ids)
    all_output_ids.extend(turn2_env_ids)
    action_mask.extend([0] * len(turn2_env_ids))

    resp2 = await llm.generate_async.remote(prompt_ids=sequence_ids, sampling_params=sp)
    out2: str = resp2.outputs[0].text
    ids2: List[int] = list(resp2.outputs[0].token_ids)

    r2 = retriever_word_count_reward(out2)
    total_reward += r2
    trajectory.append(
        {
            "turn_id": 2,
            "role": "retriever",
            "output": out2,
            "reward": r2,
        }
    )

    sequence_ids.extend(ids2)
    all_output_ids.extend(ids2)
    action_mask.extend([1] * len(ids2))

    retriever_msg_2 = out2.strip()

    # ──────────────────────────────────────────────────────────────────────────
    # Turn 3 – GENERATOR: produce the final hypothesis list
    # ──────────────────────────────────────────────────────────────────────────
    turn3_env = (
        "<|im_end|>\n"
        + _fmt("system", AgentPrompts.generator_hypothesize_system())
        + _fmt(
            "user",
            f"Research query:\n{prompt}\n\n"
            f"Retriever synthesis:\n{retriever_msg_1}\n\n"
            f"Additional retriever context:\n{retriever_msg_2}",
        )
        + "<|im_start|>assistant\n"
    )
    turn3_env_ids = _tokenize(tokenizer, turn3_env)
    sequence_ids.extend(turn3_env_ids)
    all_output_ids.extend(turn3_env_ids)
    action_mask.extend([0] * len(turn3_env_ids))

    resp3 = await llm.generate_async.remote(prompt_ids=sequence_ids, sampling_params=sp)
    out3: str = resp3.outputs[0].text
    ids3: List[int] = list(resp3.outputs[0].token_ids)

    r3 = generator_hypothesis_reward(out3)
    total_reward += r3
    trajectory.append(
        {
            "turn_id": 3,
            "role": "generator",
            "output": out3,
            "reward": r3,
        }
    )

    sequence_ids.extend(ids3)
    all_output_ids.extend(ids3)
    action_mask.extend([1] * len(ids3))

    logger.warning(
        f"workflow done | turns=4 | reward={total_reward:.3f} | "
        f"time={time.time() - t_start:.1f}s"
    )

    # ── pack into the single trajectory record expected by GRPO trainer ──────
    # action_mask aligns with sequence_ids token-by-token.
    # The initial prompt tokens have mask=0 (no loss), model-generated tokens
    # have mask=1.  Environment injection tokens also have mask=0.
    trajectory_record = {
        "node_id": 0,
        "agent_id": agent.get("agent_id", "shared_agent"),
        "agent_role": "generator",  # final role for bookkeeping
        "agent_input": turn0_prompt,
        "agent_output": out3,  # final generation
        "output_ids": all_output_ids,
        "sequence_ids": sequence_ids,
        "action_mask": action_mask,
        "reward": total_reward,
        "rollout_log_prob": None,
        "metadata": {
            "turn_rewards": [t["reward"] for t in trajectory],
            "label": label,
        },
    }

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": [trajectory_record],
        "final_reward": total_reward,
    }
