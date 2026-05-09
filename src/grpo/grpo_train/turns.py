"""
Turn execution for the scientific hypothesis generation pipeline.
================================================================

``execute_turn`` is the single public function here.  It orchestrates one
iteration of the trajectory loop:

  1. Build the prompt for the current agent.
  2. Call the LLM.
  3. Parse the tool call from the raw output.
  4. Validate that the tool is allowed this turn.
  5. Compute the reward (generate_hypotheses only).
  6. Delegate to ``apply_tool_call`` and return the next ``TrajectoryState``.

Any failure in steps 3–4 immediately terminates the trajectory with
``terminal_reason="bad_tool_call"`` and reward 0.
"""

from typing import Any, Dict, List

from src.grpo.grpo_train.prompts import build_agent_prompt
from src.grpo.grpo_train.records import (
    make_debug_entry,
    make_turn_record,
    parse_tool_call,
    tokenize,
)
from src.grpo.grpo_train.state import TrajectoryState
from src.grpo.grpo_train.tools import available_tools
from src.grpo.grpo_train.transitions import (
    _terminal_bad_call,
    apply_tool_call,
    compute_final_reward,
)


async def execute_turn(
    state: TrajectoryState,
    llm,
    tokenizer,
    sampling_params,
    agent_name: str,
    label: str,
    embed_host: str,
    embed_port: int,
    similarity_weight: float = 0.7,
    diversity_weight: float = 0.3,
) -> TrajectoryState:
    """
    Execute one trajectory turn and return the next ``TrajectoryState``.

    Parameters
    ----------
    state           : Current trajectory state.
    llm             : vLLM actor (Ray remote).
    tokenizer       : HuggingFace tokenizer.
    sampling_params : vLLM SamplingParams (or compatible dict).
    agent_name      : Human-readable agent identifier for records.
    label           : Ground-truth hypothesis string (used for reward).
    embed_host      : Hostname of the embedding server.
    embed_port      : Port of the embedding server.
    similarity_weight, diversity_weight : Reward weighting coefficients.
    """
    role = state.current_agent
    tools = available_tools(state)
    prompt = build_agent_prompt(state)

    input_ids = tokenize(tokenizer, prompt)
    resp = await llm.generate_async.remote(
        prompt_ids=input_ids, sampling_params=sampling_params
    )
    output: str = resp.outputs[0].text
    output_ids: List[int] = list(resp.outputs[0].token_ids)

    # ── parse tool call ───────────────────────────────────────────────────────
    parsed = parse_tool_call(output)

    if parsed is None:
        turn_record = make_turn_record(
            state.turn_id,
            agent_name,
            role,
            prompt,
            output,
            input_ids,
            output_ids,
            sampling_params,
            resp.outputs[0],
            reward=0.0,
        )
        debug_entry = make_debug_entry(
            state.turn_id,
            role,
            prompt,
            output,
            reward=0.0,
            input_ids=input_ids,
            output_ids=output_ids,
            tool_name=None,
            extra={"error": "missing_tool_call"},
        )
        return _terminal_bad_call(state, turn_record, debug_entry)

    tool_name, tool_args = parsed

    # ── validate tool is allowed this turn ────────────────────────────────────
    if tool_name not in tools:
        turn_record = make_turn_record(
            state.turn_id,
            agent_name,
            role,
            prompt,
            output,
            input_ids,
            output_ids,
            sampling_params,
            resp.outputs[0],
            reward=0.0,
        )
        debug_entry = make_debug_entry(
            state.turn_id,
            role,
            prompt,
            output,
            reward=0.0,
            input_ids=input_ids,
            output_ids=output_ids,
            tool_name=tool_name,
            extra={"error": f"tool_not_allowed: {tool_name} not in {tools}"},
        )
        return _terminal_bad_call(state, turn_record, debug_entry)

    # ── compute reward (only for the terminal generate_hypotheses call) ───────
    reward = 0.0
    extra_debug: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    if tool_name == "generate_hypotheses":
        hypotheses = tool_args.get("hypotheses", [])
        reward, sim_score, div_score = await compute_final_reward(
            hypotheses,
            label,
            embed_host,
            embed_port,
            similarity_weight,
            diversity_weight,
        )
        extra_debug = {
            "similarity_score": sim_score,
            "diversity_score": div_score,
            "similarity_weight": similarity_weight,
            "diversity_weight": diversity_weight,
        }
        metadata = {"label": label}

    turn_record = make_turn_record(
        state.turn_id,
        agent_name,
        role,
        prompt,
        output,
        input_ids,
        output_ids,
        sampling_params,
        resp.outputs[0],
        reward=reward,
        metadata=metadata,
    )
    debug_entry = make_debug_entry(
        state.turn_id,
        role,
        prompt,
        output,
        reward=reward,
        input_ids=input_ids,
        output_ids=output_ids,
        tool_name=tool_name,
        extra=extra_debug if extra_debug else None,
    )

    return apply_tool_call(state, tool_name, tool_args, turn_record, debug_entry)
