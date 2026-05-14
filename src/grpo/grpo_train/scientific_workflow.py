"""
Scientific Hypothesis Generation Workflow
==========================================
Fixed-length pipeline: every trajectory runs exactly
``(ASK_RETRIEVER_LIMIT + 1) * (RETRIEVER_SEARCH_LIMIT + 2)`` steps.

Step sequence (K=ASK_RETRIEVER_LIMIT, S=RETRIEVER_SEARCH_LIMIT)
----------------------------------------------------------------
For each of the (K+1) retriever episodes:
  S x retriever_search   – retriever outputs a search query; Weaviate is called
  1 x retriever_message  – retriever outputs synthesis text for the generator
  (between episodes, K times)
  1 x generator_ask      – generator outputs a follow-up question
Finally:
  1 x generator_generate – generator outputs a numbered hypothesis list

No tool-call formatting is required from the LLMs.  Each model is prompted
to output only the relevant content for its step.  The final reward
(embedding similarity + diversity + groundedness + relevancy) is computed on
the parsed hypotheses and propagated back to all trajectory records.

kwargs (via --workflow_args JSON)
---------------------------------
  embed_host             – hostname of the vLLM embedding server  (default: "localhost")
  embed_port             – port of the vLLM embedding server       (default: 8000)
  rerank_host            – hostname of the vLLM reranker server   (default: same as embed_host)
  rerank_port            – port of the vLLM reranker server       (default: 8001)
  debug_dir              – directory for per-trajectory JSON logs  (default: "./workflow_debug_logs")
  debug                  – write per-trajectory JSON logs          (default: true)
  similarity_weight      – weight for embedding similarity reward  (default: 0.7)
  diversity_weight       – weight for hypothesis diversity reward   (default: 0.3)
  groundedness_weight    – weight for groundedness reward (reranker) (default: 0.0)
  relevancy_weight       – weight for relevancy reward (reranker)    (default: 0.0)
  use_weaviate_context   – fetch papers live from Weaviate         (default: false)
  weaviate_top_n         – number of Weaviate results to fetch     (default: 6)
  weaviate_url           – Weaviate base URL
  ask_retriever_limit    – K: generator→retriever rounds           (default: 1)
  retriever_search_limit – S: Weaviate searches per episode        (default: 1)
  apply_length_penalty   – enable Information Bottleneck penalty   (default: false)
  length_penalty_lambda  – λ for length penalty                    (default: 0.001)
  apply_channel_noise    – enable channel noise injection          (default: false)
  noise_probability      – probability of masking each word token  (default: 0.1)
"""

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
if agents_dir not in sys.path:
    sys.path.insert(0, agents_dir)

from marti.utils.logging_utils import init_logger
from src.grpo.grpo_train.tools import search_weaviate
from src.grpo.grpo_train.state import TrajectoryState, initial_state
from src.grpo.grpo_train.turns import execute_turn

logger = init_logger(__name__)
logger.setLevel("WARN")


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
    _wargs: Dict[str, Any] = kwargs.get("workflow_args") or {}

    def _get(key: str, default: Any) -> Any:
        return _wargs.get(key, kwargs.get(key, default))

    embed_host: str = _get("embed_host", "localhost")
    embed_port: int = int(_get("embed_port", 8000))
    rerank_host: str = _get("rerank_host", embed_host)
    rerank_port: int = int(_get("rerank_port", 8001))
    debug_dir: str = _get("debug_dir", "./workflow_debug_logs")
    debug: bool = str(_get("debug", "true")).lower() not in ("false", "0", "no")
    is_eval: bool = kwargs.get("is_eval", False)
    if debug:
        split_name = "eval" if is_eval else "train"
        debug_dir = os.path.join(debug_dir, split_name)
    similarity_weight: float = float(_get("similarity_weight", 0.7))
    diversity_weight: float = float(_get("diversity_weight", 0.3))
    groundedness_weight: float = float(_get("groundedness_weight", 0.0))
    relevancy_weight: float = float(_get("relevancy_weight", 0.0))
    use_weaviate_context: bool = str(
        _get("use_weaviate_context", "false")
    ).lower() not in ("false", "0", "no")
    weaviate_top_n: int = int(_get("weaviate_top_n", 6))
    ask_retriever_limit: int = int(_get("ask_retriever_limit", 1))
    retriever_search_limit: int = int(_get("retriever_search_limit", 1))
    weaviate_url: str = _get("weaviate_url", "http://localhost:8080")
    prompt_id: int = kwargs.get("prompt_id", 0)

    # ── emergent communication parameters ─────────────────────────────────────
    apply_length_penalty: bool = str(
        _get("apply_length_penalty", "false")
    ).lower() not in ("false", "0", "no")
    length_penalty_lambda: float = float(_get("length_penalty_lambda", 0.001))

    apply_channel_noise: bool = str(
        _get("apply_channel_noise", "false")
    ).lower() not in ("false", "0", "no")
    noise_probability: float = float(_get("noise_probability", 0.1))

    # ── paper context ─────────────────────────────────────────────────────────
    if use_weaviate_context:
        papers = search_weaviate(
            prompt, weaviate_top_n, weaviate_url, embed_host, embed_port
        )
        paper_block = _format_papers(papers, max_papers=weaviate_top_n)
    else:
        metadata = json.loads(json.loads(metadata))
        papers = (metadata or {}).get("papers", [])
        paper_block = _format_papers(papers, max_papers=weaviate_top_n)

    # ── trajectory loop ───────────────────────────────────────────────────────
    state: TrajectoryState = initial_state(
        prompt,
        paper_block,
        papers=papers,
        ask_retriever_limit=ask_retriever_limit,
        retriever_search_limit=retriever_search_limit,
    )

    while not state.is_terminal:
        state = await execute_turn(
            state=state,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sp,
            agent_name=agent_name,
            label=label,
            embed_host=embed_host,
            embed_port=embed_port,
            weaviate_url=weaviate_url,
            similarity_weight=similarity_weight,
            diversity_weight=diversity_weight,
            rerank_host=rerank_host,
            rerank_port=rerank_port,
            groundedness_weight=groundedness_weight,
            relevancy_weight=relevancy_weight,
            apply_channel_noise=apply_channel_noise,
            noise_probability=noise_probability,
        )

    # ── assemble results & apply length penalty (Technique 1) ─────────────────
    trajectory = list(state.trajectory_records)
    debug_entries = list(state.debug_entries)

    final_reward = trajectory[-1].get("reward", 0.0) if trajectory else 0.0
    total_length_penalty = 0.0

    for i, (record, entry) in enumerate(zip(trajectory, debug_entries)):
        extra_dict = entry.get("extra", {})
        
        agent_role = entry.get("agent_role") or extra_dict.get("agent_role")
        turn_type = entry.get("turn_type") or extra_dict.get("turn_type")
        
        if agent_role == "retriever" and turn_type == "retriever_message":
            if apply_length_penalty:
                clean_content = entry.get("output_content") or extra_dict.get("output_content", "")
                
                tokens = tokenizer.encode(clean_content, add_special_tokens=False)
                token_count = len(tokens)
                
                penalty = length_penalty_lambda * token_count
                total_length_penalty += penalty

                if "extra" in entry:
                    entry["extra"]["length_penalty"] = round(penalty, 4)
                    entry["extra"]["n_channel_tokens"] = token_count
                else:
                    entry["length_penalty"] = round(penalty, 4)
                    entry["n_channel_tokens"] = token_count

    # Final reward after applying the length penalty
    penalised_reward = final_reward - total_length_penalty

    # Propagate the final reward (after length penalty) back to all trajectory records
    for i, (record, entry) in enumerate(zip(trajectory, debug_entries)):
        record["reward"] = penalised_reward
        entry["reward"] = penalised_reward

    if apply_length_penalty and total_length_penalty > 0.0:
        logger.warning(
            f"[EC] length_penalty={total_length_penalty:.4f} | "
            f"final_reward={final_reward:.3f} | "
            f"penalised_reward={penalised_reward:.3f}"
        )

    reward_matrix = [r.get("reward", 0.0) for r in trajectory]

    elapsed = time.time() - t_start
    logger.warning(
        f"workflow done | turns={state.turn_id} | reward={penalised_reward:.3f} | time={elapsed:.1f}s"
    )

    if debug:
        _write_debug_log(
            debug_dir=debug_dir,
            prompt_id=prompt_id,
            prompt=prompt,
            label=label,
            total_reward=penalised_reward,
            elapsed_s=round(elapsed, 2),
            turns=debug_entries,
        )

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "reward_matrix": reward_matrix,
        "final_reward": penalised_reward,
    }