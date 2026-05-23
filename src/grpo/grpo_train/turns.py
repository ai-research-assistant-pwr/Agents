"""
Turn execution for the scientific hypothesis generation pipeline.
================================================================

``execute_turn`` orchestrates one step of the fixed-length trajectory:

  1. Build the ChatML prompt for the current step type.
  2. Call the LLM and collect the raw output text + token ids.
  3. Extract the relevant content from the output based on step type:
       retriever_search   → strip the output → use as search query
       retriever_message  → strip the output → use as synthesis message
       generator_ask      → strip the output → use as follow-up question
       generator_generate → _parse_hypotheses(output) → compute final reward
  4. For ``retriever_search``: execute the Weaviate search and store the result.
  5. For ``generator_generate``: compute embedding-based reward (0.0 if no
     valid hypotheses were parsed).
  6. Delegate to ``apply_step`` in transitions.py and return the next state.

All intermediate steps receive reward=0.0 in the trajectory record; the final
reward from ``generator_generate`` is propagated to all records by the workflow.
"""

import re
import random
from typing import Any, Dict, List, Optional

from src.grpo.grpo_train.prompts import build_agent_prompt
from src.grpo.grpo_train.records import (
    make_debug_entry,
    make_turn_record,
    tokenize,
    strip_thinking,
)
from src.grpo.grpo_train.rewards import _parse_hypotheses
from src.grpo.grpo_train.state import TrajectoryState
from src.grpo.grpo_train.tools import search_papers_tool
from src.grpo.grpo_train.transitions import apply_step, compute_final_reward

# ── emergent communication helpers ───────────────────────────────────────────

_MASK_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b")

def _inject_noise(
    text: str,
    noise_prob: float,
    mask_token: str = "[MASK]",
    span_probability: float = 0.15,
    max_span_length: int = 3,
) -> str:
    """Randomly masks word-tokens to force redundant, compositional communication.

    Only word tokens matching ``_MASK_PATTERN`` (alphabetic, length ≥ 3) are
    candidates for masking.  Numbers, punctuation, and special tokens are left
    intact so the channel does not become completely unreadable at high noise
    levels.

    Parameters
    ----------
    text            : Clean retriever message text (post strip_thinking).
    noise_prob      : Per-token probability of masking (Technique 2 λ_noise).
    mask_token      : Replacement string for masked tokens.
    span_probability: Probability that a masked position starts a multi-token
                      span (default 15 %).
    max_span_length : Maximum number of consecutive tokens in a masked span.
    """
    if noise_prob <= 0.0:
        return text

    matches = list(_MASK_PATTERN.finditer(text))
    if not matches:
        return text

    output = []
    last_end = 0
    i = 0

    while i < len(matches):
        match = matches[i]
        output.append(text[last_end : match.start()])

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


# ── turn execution ────────────────────────────────────────────────────────────

async def execute_turn(
    state: TrajectoryState,
    llm,
    tokenizer,
    sampling_params,
    agent_name: str,
    label: str,
    embed_host: str,
    embed_port: int,
    weaviate_url: str,
    similarity_weight: float = 0.7,
    diversity_weight: float = 0.3,
    rerank_host: Optional[str] = None,
    rerank_port: Optional[int] = None,
    groundedness_weight: float = 0.0,
    relevancy_weight: float = 0.0,
    apply_channel_noise: bool = False,
    noise_probability: float = 0.1,
    is_eval: bool = False,
) -> TrajectoryState:
    """
    Execute one trajectory step and return the next ``TrajectoryState``.
    """
    step = state.current_step
    prompt = build_agent_prompt(state)

    input_ids = tokenize(tokenizer, prompt)
    resp = await llm.generate_async.remote(
        prompt_ids=input_ids, sampling_params=sampling_params
    )
    raw_text = resp.outputs[0].text
    output: str = raw_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    output_ids: List[int] = list(resp.outputs[0].token_ids)

    # ── step-specific extraction and side-effects ─────────────────────────────
    reward: float = 0.0
    extra_debug: Dict[str, Any] = {
        "step": step,
        "turn_type": step,
        "agent_role": state.current_agent,
    }
    step_payload: Dict[str, Any] = {}

    if step == "retriever_search":
        query = strip_thinking(output)
        search_result = search_papers_tool(query, weaviate_url, embed_host, embed_port)
        step_payload = {"query": query, "search_result": search_result}
        extra_debug.update({"search_query": query, "search_result": search_result})

    elif step == "retriever_message":
        clean_message = strip_thinking(output)

        n_channel_tokens: int = len(
            tokenizer.encode(clean_message, add_special_tokens=False)
        )

        # apply_noise_this_step = apply_channel_noise and not is_eval
        # for inference on eval data
        apply_noise_this_step = apply_channel_noise

        if apply_noise_this_step:
            noisy_message = _inject_noise(clean_message, noise_probability)
            message_for_payload = noisy_message
            extra_debug.update(
                {
                    # ``output_content`` → clean text used by Technique 1 (length penalty)
                    "output_content": clean_message,
                    # ``noisy_output_content`` → what Generator actually sees (for CIC analysis)
                    "noisy_output_content": noisy_message,
                    "message": clean_message,
                    "n_channel_tokens": n_channel_tokens,
                    "noise_applied": True,
                }
            )
        else:
            message_for_payload = clean_message
            extra_debug.update(
                {
                    "output_content": clean_message,
                    "message": clean_message,
                    "n_channel_tokens": n_channel_tokens,
                    "noise_applied": False,
                }
            )

        step_payload = {"message": message_for_payload}

    elif step == "generator_ask":
        question = strip_thinking(output)
        step_payload = {"question": question}
        extra_debug.update({"question": question})

    elif step == "generator_generate":
        hypotheses = _parse_hypotheses(strip_thinking(output))
        (
            reward,
            sim_score,
            div_score,
            ground_score,
            relev_score,
        ) = await compute_final_reward(
            hypotheses,
            label,
            embed_host,
            embed_port,
            similarity_weight,
            diversity_weight,
            rerank_host=rerank_host,
            rerank_port=rerank_port,
            groundedness_weight=groundedness_weight,
            relevancy_weight=relevancy_weight,
            papers=state.papers,
            query=state.query,
        )
        step_payload = {"hypotheses": hypotheses}
        extra_debug.update(
            {
                "hypotheses": hypotheses,
                "similarity_score": sim_score,
                "diversity_score": div_score,
                "groundedness_score": ground_score,
                "relevancy_score": relev_score,
                "similarity_weight": similarity_weight,
                "diversity_weight": diversity_weight,
                "groundedness_weight": groundedness_weight,
                "relevancy_weight": relevancy_weight,
            }
        )

    turn_record = make_turn_record(
        turn_id=state.turn_id,
        agent_name=agent_name,
        role=state.current_agent,
        prompt=prompt,
        output=output,
        input_ids=input_ids,
        output_ids=output_ids,
        sampling_params=sampling_params,
        resp_output=resp.outputs[0],
        reward=reward,
        metadata={"label": label} if step == "generator_generate" else {},
    )
    debug_entry = make_debug_entry(
        turn_id=state.turn_id,
        role=state.current_agent,
        prompt=prompt,
        output=output,
        reward=reward,
        input_ids=input_ids,
        output_ids=output_ids,
        tool_name=step,
        extra=extra_debug,
    )

    return apply_step(state, step, step_payload, turn_record, debug_entry)