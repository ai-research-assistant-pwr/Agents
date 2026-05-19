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

Agent routing
-------------
The workflow expects agents to be passed with a ``role`` field set to either
``"retriever"`` or ``"generator"``.  Routing is done via ``state.current_agent``
(not by string-matching step names) so it stays correct even if step names
change in the future.

Two-agent mode (independent circuits, EAP-compatible):
  AGENTS_CONFIG contains two entries with distinct agent_id values.
  Each agent accumulates its own gradient history and, after training,
  can be used independently for Attribution Patching / EAP analysis.

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
  length_penalty_lambda  – λ for length penalty                    (default: 0.005)
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

# ── valid agent roles ─────────────────────────────────────────────────────────
# Centralised so that routing logic and error messages stay in sync.
_RETRIEVER_ROLE = "retriever"
_GENERATOR_ROLE = "generator"
_VALID_ROLES    = {_RETRIEVER_ROLE, _GENERATOR_ROLE}

# Steps that belong to the retriever agent.  Everything else goes to generator.
# Using an explicit set rather than substring matching so renames never silently
# misroute a step.
_RETRIEVER_STEPS = {"retriever_search", "retriever_message"}


# ── paper-context helper ──────────────────────────────────────────────────────


def _format_papers(papers: List[Dict[str, str]], max_papers: int = 8) -> str:
    selected = papers[:max_papers]
    lines = []
    for i, p in enumerate(selected, 1):
        title   = p.get("title",   "Unknown title")
        summary = p.get("summary", p.get("abstract", "No summary available."))
        lines.append(f"[Paper {i}] {title}\n{summary.strip()}")
    return "\n\n".join(lines)


# ── agent-dict builder ────────────────────────────────────────────────────────


def _build_agent_dict(
    agents: List[Dict[str, Any]],
    stop_tokens: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Build a role → agent mapping and validate that both required roles are
    present.

    Each agent entry must have a ``role`` field set to ``"retriever"`` or
    ``"generator"``.  If an agent's ``sampling_params`` supports a ``stop``
    attribute it is set here so we do not repeat the logic in the loop.

    Raises
    ------
    ValueError
        If a required role is missing or an unknown role is encountered.
    """
    agent_dict: Dict[str, Dict[str, Any]] = {}

    for a in agents:
        role = a.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(
                f"Agent '{a.get('agent_id', '<unknown>')}' has role={role!r}. "
                f"Expected one of {sorted(_VALID_ROLES)}."
            )
        if role in agent_dict:
            raise ValueError(
                f"Duplicate agent role {role!r}. "
                "Each role must be assigned to exactly one agent."
            )

        sp = a["sampling_params"]
        if hasattr(sp, "stop"):
            sp.stop = stop_tokens
        elif isinstance(sp, dict):
            sp["stop"] = stop_tokens

        agent_dict[role] = a

    missing = _VALID_ROLES - set(agent_dict)
    if missing:
        raise ValueError(
            f"Missing agent(s) for role(s): {sorted(missing)}. "
            "Both 'retriever' and 'generator' agents must be provided."
        )

    return agent_dict


def _route_agent(
    state: TrajectoryState,
    agent_dict: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Return the agent that should execute the current trajectory step.

    Routing is based on ``state.current_step`` matched against the explicit
    ``_RETRIEVER_STEPS`` set — not substring matching — so future step-name
    changes do not silently misroute.

    Falls back to ``state.current_agent`` if it is set and the step is not in
    the retriever set (defensive: handles any step added later that does not
    follow the retriever_*/generator_* naming convention).
    """
    step = state.current_step

    if step in _RETRIEVER_STEPS:
        role = _RETRIEVER_ROLE
    else:
        # Prefer explicit field over heuristic when available
        role = getattr(state, "current_agent", None) or _GENERATOR_ROLE

    agent = agent_dict.get(role)
    if agent is None:
        raise RuntimeError(
            f"No agent registered for role {role!r} at step {step!r}. "
            f"Available roles: {list(agent_dict.keys())}"
        )
    return agent


# ── debug helper ──────────────────────────────────────────────────────────────


def _write_debug_log(
    debug_dir: str,
    prompt_id: int,
    prompt: str,
    label: str,
    total_reward: float,
    elapsed_s: float,
    turns: List[Dict[str, Any]],
    ec_stats: Dict[str, Any],
) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    ts_ms   = int(time.time() * 1000)
    uid     = uuid.uuid4().hex[:8]
    path    = os.path.join(debug_dir, f"traj_{ts_ms}_{uid}.json")
    tmp_path = path + ".tmp"
    payload = {
        "prompt_id":    prompt_id,
        "prompt":       prompt,
        "label":        label,
        "total_reward": total_reward,
        "elapsed_s":    elapsed_s,
        "ec_stats":     ec_stats,
        "turns":        turns,
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
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    # _build_agent_dict validates roles and sets stop tokens; raises early with
    # a clear message if something is misconfigured so we never get a KeyError
    # halfway through a trajectory.
    agent_dict = _build_agent_dict(agents, stop_tokens)

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
    length_penalty_lambda: float = float(_get("length_penalty_lambda", 0.005))

    apply_channel_noise: bool = str(
        _get("apply_channel_noise", "false")
    ).lower() not in ("false", "0", "no")
    noise_probability: float = float(_get("noise_probability", 0.1))

    # ── paper context ─────────────────────────────────────────────────────────
    if use_weaviate_context:
        papers = search_weaviate(prompt, weaviate_top_n, weaviate_url, embed_host, embed_port)
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
        # Route to the correct agent for this step
        active_agent = _route_agent(state, agent_dict)
        state = await execute_turn(
            state = state,
            llm = active_agent["llm"],
            tokenizer = active_agent["tokenizer"],
            sampling_params = active_agent["sampling_params"],
            agent_name = active_agent.get("agent_id"),
            label = label,
            embed_host = embed_host,
            embed_port = embed_port,
            weaviate_url = weaviate_url,
            similarity_weight = similarity_weight,
            diversity_weight = diversity_weight,
            rerank_host = rerank_host,
            rerank_port = rerank_port,
            groundedness_weight = groundedness_weight,
            relevancy_weight = relevancy_weight,
            apply_channel_noise = apply_channel_noise,
            noise_probability = noise_probability,
            is_eval = is_eval,
        )

    # ── assemble results & apply length penalty (Technique 1) ─────────────────
    trajectory = list(state.trajectory_records)
    debug_entries = list(state.debug_entries)

    final_reward = trajectory[-1].get("reward", 0.0) if trajectory else 0.0
    total_length_penalty = 0.0
    channel_token_counts: List[int] = []

    for record, entry in zip(trajectory, debug_entries):
        extra_dict = entry.get("extra", {})
        agent_role = entry.get("agent_role") or extra_dict.get("agent_role")
        turn_type  = entry.get("turn_type")  or extra_dict.get("turn_type")

        if agent_role == _RETRIEVER_ROLE and turn_type == "retriever_message":
            n_tokens = entry.get("n_channel_tokens", 0) or extra_dict.get("n_channel_tokens", 0)
            channel_token_counts.append(n_tokens)

            if apply_length_penalty and not is_eval:
                penalty = length_penalty_lambda * n_tokens
                total_length_penalty += penalty
                extra_dict["length_penalty"] = round(penalty, 4)
                extra_dict["n_channel_tokens"] = n_tokens

    penalised_reward = final_reward - total_length_penalty

    for record, entry in zip(trajectory, debug_entries):
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

    ec_stats: Dict[str, Any] = {
        "n_retriever_messages": len(channel_token_counts),
        "channel_token_counts": channel_token_counts,
        "mean_channel_tokens": (
            round(sum(channel_token_counts) / len(channel_token_counts), 2)
            if channel_token_counts else 0.0
        ),
        "total_length_penalty": round(total_length_penalty, 4),
        "task_reward_before_penalty": round(final_reward, 4),
        "apply_length_penalty": apply_length_penalty,
        "apply_channel_noise": apply_channel_noise,
        "is_eval": is_eval,
    }

    if debug:
        _write_debug_log(
            debug_dir = debug_dir,
            prompt_id = prompt_id,
            prompt = prompt,
            label = label,
            total_reward = penalised_reward,
            elapsed_s = round(elapsed, 2),
            turns = debug_entries,
            ec_stats = ec_stats,
        )

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "reward_matrix": reward_matrix,
        "final_reward": penalised_reward,
        "ec_stats": ec_stats,
    }