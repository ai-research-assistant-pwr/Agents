"""
Record construction and tokenisation helpers for the scientific workflow.
=========================================================================

Functions here are responsible for:
  - tokenising a prompt string into token-id lists
  - extracting rollout log-probabilities from a vLLM response output
  - building MARTI-compatible trajectory records (``make_turn_record``)
  - building human-readable debug entries (``make_debug_entry``)
"""

import json
import re as _re
from typing import Any, Dict, List, Optional

_TOOL_CALL_RE = _re.compile(r"<tool_call>(.*?)</tool_call>", _re.DOTALL)
_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks and return the remaining text stripped."""
    return _THINK_RE.sub("", text).strip()


# ── tokenisation ──────────────────────────────────────────────────────────────


def tokenize(tokenizer, text: str) -> List[int]:
    """Tokenise *text* without special tokens; return a plain Python int list."""
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][
        0
    ].tolist()


# ── log-probability extraction ────────────────────────────────────────────────


def extract_rollout_log_probs(
    response_output,
    input_ids: List[int],
    output_ids: List[int],
    sampling_params,
) -> Optional[List[float]]:
    """
    Extract per-token log-probabilities from a vLLM response output.

    Returns ``None`` when the sampling params did not request log-probs.
    The returned list has length ``len(input_ids) + len(output_ids)``:
    input positions are filled with 0.0 (no log-prob available).
    """
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


# ── tool-call parsing ─────────────────────────────────────────────────────────


def parse_tool_call(output: str):
    """
    Extract ``(tool_name, arguments)`` from the model's raw output.

    Looks for the first ``<tool_call>…</tool_call>`` block and parses the JSON
    inside it.  Returns ``None`` if the block is missing or the JSON is invalid.
    """
    match = _TOOL_CALL_RE.search(output)
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
        return data["name"], data.get("arguments", {})
    except Exception:
        return None


# ── record builders ───────────────────────────────────────────────────────────


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
    """Build a MARTI-compatible trajectory record for one turn."""
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
    """Build a human-readable debug entry for one turn."""
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
