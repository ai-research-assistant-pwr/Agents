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
) -> TrajectoryState:
    """
    Execute one trajectory step and return the next ``TrajectoryState``.

    Parameters
    ----------
    state           : Current trajectory state.
    llm             : vLLM actor (Ray remote).
    tokenizer       : HuggingFace tokenizer.
    sampling_params : vLLM SamplingParams (or compatible dict).
    agent_name      : Human-readable agent identifier stored in records.
    label           : Ground-truth hypothesis string (used for terminal reward).
    embed_host      : Hostname of the embedding server.
    embed_port      : Port of the embedding server.
    weaviate_url    : Base URL of the Weaviate instance.
    similarity_weight, diversity_weight : Reward weighting coefficients.
    rerank_host     : Hostname of the reranker server (optional).
    rerank_port     : Port of the reranker server (optional).
    groundedness_weight : Weight for groundedness reward (reranker-based).
    relevancy_weight    : Weight for relevancy reward (reranker-based).
    """
    step = state.current_step
    prompt = build_agent_prompt(state)

    input_ids = tokenize(tokenizer, prompt)
    resp = await llm.generate_async.remote(
        prompt_ids=input_ids, sampling_params=sampling_params
    )
    output: str = resp.outputs[0].text
    output_ids: List[int] = list(resp.outputs[0].token_ids)

    # ── step-specific extraction and side-effects ─────────────────────────────
    reward: float = 0.0
    extra_debug: Dict[str, Any] = {"step": step}
    step_payload: Dict[str, Any] = {}  # passed to apply_step

    if step == "retriever_search":
        query = strip_thinking(output)
        search_result = search_papers_tool(query, weaviate_url, embed_host, embed_port)
        step_payload = {"query": query, "search_result": search_result}
        extra_debug.update({"search_query": query, "search_result": search_result})

    elif step == "retriever_message":
        message = strip_thinking(output)
        step_payload = {"message": message}
        extra_debug.update({"message": message})

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
