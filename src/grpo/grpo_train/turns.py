"""
Turn functions for the scientific hypothesis generation pipeline.
=================================================================

Each function executes one turn of the 4-turn pipeline, handles generation,
thinking-token stripping, reward calculation, and returns a ``TurnResult``
containing everything the workflow needs: the trajectory record (for MARTI),
the debug entry (for JSON logging), the stripped output text (fed into the
next turn's prompt), and the scalar reward.

Turn layout
-----------
  turn_0_retriever_synthesize  – query + papers  →  initial synthesis
  turn_1_generator_ask         – query + synthesis  →  follow-up question
  turn_2_retriever_refine      – question + papers  →  focused clarification
  turn_3_generator_hypothesize – query + both retrievals  →  hypothesis list
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.grpo.grpo_train.agents import AgentPrompts
from src.grpo.grpo_train.rewards import embedding_similarity_reward, _parse_hypotheses


# ── shared helpers (imported by workflow too) ─────────────────────────────────


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


import re as _re


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks from model output."""
    return _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()


# ── result container ──────────────────────────────────────────────────────────


@dataclass
class TurnResult:
    """Everything produced by a single pipeline turn."""

    trajectory_record: Dict[str, Any]  # appended to trajectory list
    debug_entry: Dict[str, Any]  # appended to debug log
    output_content: str  # thinking-stripped output for next turn
    reward: float  # scalar reward for this turn


# ── turn functions ────────────────────────────────────────────────────────────


async def turn_0_retriever_synthesize(
    llm,
    tokenizer,
    sampling_params,
    agent_name: str,
    query: str,
    paper_block: str,
) -> TurnResult:
    """Turn 0 – Retriever: synthesise paper summaries into initial evidence block."""
    prompt = build_prompt(
        system=AgentPrompts.retriever_system(),
        user=f"Research query:\n{query}\n\nAvailable papers:\n{paper_block}",
    )
    input_ids = tokenize(tokenizer, prompt)
    resp = await llm.generate_async.remote(
        prompt_ids=input_ids, sampling_params=sampling_params
    )
    output: str = resp.outputs[0].text
    output_ids: List[int] = list(resp.outputs[0].token_ids)
    sequence_ids = input_ids + output_ids
    output_content = strip_thinking(output)
    reward = 0.0

    trajectory_record = {
        "turn_id": 0,
        "agent_index": 0,
        "agent_id": agent_name,
        "agent_name": agent_name,
        "agent_role": "retriever",
        "agent_input": prompt,
        "agent_output": output,
        "output_ids": output_ids,
        "sequence_ids": sequence_ids,
        "rollout_log_prob": extract_rollout_log_probs(
            resp.outputs[0], input_ids, output_ids, sampling_params
        ),
        "reward": reward,
        "metadata": {},
    }
    debug_entry = {
        "turn_id": 0,
        "role": "retriever",
        "input": prompt,
        "output": output,
        "output_content": output_content,
        "reward": reward,
        "n_input_tokens": len(input_ids),
        "n_output_tokens": len(output_ids),
    }
    return TurnResult(
        trajectory_record=trajectory_record,
        debug_entry=debug_entry,
        output_content=output_content,
        reward=reward,
    )


async def turn_1_generator_ask(
    llm,
    tokenizer,
    sampling_params,
    agent_name: str,
    query: str,
    retriever_synthesis: str,
) -> TurnResult:
    """Turn 1 – Generator: formulate a focused follow-up question for the Retriever."""
    prompt = build_prompt(
        system=AgentPrompts.generator_ask_system(),
        user=f"Research query:\n{query}\n\nRetriever synthesis:\n{retriever_synthesis}",
    )
    input_ids = tokenize(tokenizer, prompt)
    resp = await llm.generate_async.remote(
        prompt_ids=input_ids, sampling_params=sampling_params
    )
    output: str = resp.outputs[0].text
    output_ids: List[int] = list(resp.outputs[0].token_ids)
    sequence_ids = input_ids + output_ids
    output_content = strip_thinking(output)
    reward = 0.0

    trajectory_record = {
        "turn_id": 1,
        "agent_index": 0,
        "agent_id": agent_name,
        "agent_name": agent_name,
        "agent_role": "generator",
        "agent_input": prompt,
        "agent_output": output,
        "output_ids": output_ids,
        "sequence_ids": sequence_ids,
        "rollout_log_prob": extract_rollout_log_probs(
            resp.outputs[0], input_ids, output_ids, sampling_params
        ),
        "reward": reward,
        "metadata": {},
    }
    debug_entry = {
        "turn_id": 1,
        "role": "generator",
        "input": prompt,
        "output": output,
        "output_content": output_content,
        "reward": reward,
        "n_input_tokens": len(input_ids),
        "n_output_tokens": len(output_ids),
    }
    return TurnResult(
        trajectory_record=trajectory_record,
        debug_entry=debug_entry,
        output_content=output_content,
        reward=reward,
    )


async def turn_2_retriever_refine(
    llm,
    tokenizer,
    sampling_params,
    agent_name: str,
    generator_question: str,
    paper_block: str,
) -> TurnResult:
    """Turn 2 – Retriever: answer the generator's follow-up question from the papers."""
    prompt = build_prompt(
        system=AgentPrompts.retriever_refine_system(),
        user=(
            f"The generator is asking:\n{generator_question}\n\n"
            f"Available papers (same pool):\n{paper_block}"
        ),
    )
    input_ids = tokenize(tokenizer, prompt)
    resp = await llm.generate_async.remote(
        prompt_ids=input_ids, sampling_params=sampling_params
    )
    output: str = resp.outputs[0].text
    output_ids: List[int] = list(resp.outputs[0].token_ids)
    sequence_ids = input_ids + output_ids
    output_content = strip_thinking(output)
    reward = 0.0

    trajectory_record = {
        "turn_id": 2,
        "agent_index": 0,
        "agent_id": agent_name,
        "agent_name": agent_name,
        "agent_role": "retriever",
        "agent_input": prompt,
        "agent_output": output,
        "output_ids": output_ids,
        "sequence_ids": sequence_ids,
        "rollout_log_prob": extract_rollout_log_probs(
            resp.outputs[0], input_ids, output_ids, sampling_params
        ),
        "reward": reward,
        "metadata": {},
    }
    debug_entry = {
        "turn_id": 2,
        "role": "retriever",
        "input": prompt,
        "output": output,
        "output_content": output_content,
        "reward": reward,
        "n_input_tokens": len(input_ids),
        "n_output_tokens": len(output_ids),
    }
    return TurnResult(
        trajectory_record=trajectory_record,
        debug_entry=debug_entry,
        output_content=output_content,
        reward=reward,
    )


async def turn_3_generator_hypothesize(
    llm,
    tokenizer,
    sampling_params,
    agent_name: str,
    query: str,
    retriever_synthesis: str,
    retriever_refinement: str,
    label: str,
    embed_host: str,
    embed_port: int,
) -> TurnResult:
    """Turn 3 – Generator: produce the final numbered hypothesis list.

    Reward is the mean cosine similarity between each generated hypothesis and
    the ground-truth label embedding, computed via the vLLM embedding server.
    Returns reward=0.0 if the output is not in the expected numbered-list format.
    """
    prompt = build_prompt(
        system=AgentPrompts.generator_hypothesize_system(),
        user=(
            f"Research query:\n{query}\n\n"
            f"Retriever synthesis:\n{retriever_synthesis}\n\n"
            f"Additional retriever context:\n{retriever_refinement}"
        ),
    )
    input_ids = tokenize(tokenizer, prompt)
    resp = await llm.generate_async.remote(
        prompt_ids=input_ids, sampling_params=sampling_params
    )
    output: str = resp.outputs[0].text
    output_ids: List[int] = list(resp.outputs[0].token_ids)
    sequence_ids = input_ids + output_ids
    output_content = strip_thinking(output)

    hypotheses = _parse_hypotheses(output_content)
    if hypotheses:
        reward = await embedding_similarity_reward(
            hypotheses, label, embed_host, embed_port
        )
    else:
        reward = 0.0

    trajectory_record = {
        "turn_id": 3,
        "agent_index": 0,
        "agent_id": agent_name,
        "agent_name": agent_name,
        "agent_role": "generator",
        "agent_input": prompt,
        "agent_output": output,
        "output_ids": output_ids,
        "sequence_ids": sequence_ids,
        "rollout_log_prob": extract_rollout_log_probs(
            resp.outputs[0], input_ids, output_ids, sampling_params
        ),
        "reward": reward,
        "metadata": {"label": label},
    }
    debug_entry = {
        "turn_id": 3,
        "role": "generator",
        "input": prompt,
        "output": output,
        "output_content": output_content,
        "reward": reward,
        "n_input_tokens": len(input_ids),
        "n_output_tokens": len(output_ids),
    }
    return TurnResult(
        trajectory_record=trajectory_record,
        debug_entry=debug_entry,
        output_content=output_content,
        reward=reward,
    )
