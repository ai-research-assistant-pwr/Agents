import copy
import json
import random
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import StructuredOutputsParams

from app.explorer.tools.tool_definitions import TOOL_DEFINITIONS, get_tool_executor

DEFAULT_MODEL = "Qwen/Qwen3-4B"
MAX_MODEL_LEN = 8192
MAX_ITERATIONS = 5
MAX_RESULTS_PER_TOOL = 10

llm: LLM | None = None
tokenizer: AutoTokenizer | None = None


def _load_model(model_name: str = DEFAULT_MODEL) -> tuple[LLM, AutoTokenizer]:
    global llm, tokenizer
    if llm is None:
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=model_name,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        )
        print("Model loaded.")
    return llm, tokenizer


def reload_model(model_name: str = DEFAULT_MODEL) -> tuple[LLM, AutoTokenizer]:
    global llm, tokenizer
    if llm is not None:
        del llm
        llm = None
        tokenizer = None
    return _load_model(model_name)


def load_prompt(prompt_path: str) -> dict[str, Any]:
    path = Path(prompt_path)
    if not path.exists():
        path = Path(__file__).parent.parent / "prompts" / prompt_path

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def format_prompt(
    prompt_config: dict[str, Any],
    user_query: str,
    papers_range: str,
    context: str,
    tools: str,
) -> str:
    system = prompt_config.get("system", "")
    task = prompt_config.get("task", "").format(user_query=user_query)
    loop = prompt_config.get("loop", "").format(range=papers_range)

    tools_section = f"\n## TOOLS\n{tools}" if tools else ""

    return f"""## ROLE
{system}

## TASK
{task}

## INSTRUCTIONS
{loop}

## CURRENT CONTEXT
{context}
{tools_section}

## OUTPUT INSTRUCTION
Return ONLY a valid JSON object.
Do not include any explanations, greetings, or extra newlines.

JSON output:"""


def _call_llm(
    llm_model: LLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    json_schema: dict[str, Any] | None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Call LLM and parse JSON response."""
    messages = [{"role": "user", "content": prompt}]

    processed = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    processed = tokenizer(processed, add_special_tokens=False).input_ids

    inputs = [TokensPrompt(prompt_token_ids=processed)]

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=512,
        structured_outputs=(
            StructuredOutputsParams(json=json_schema) if json_schema else None
        ),
        stop=["<|im_end|>"],
    )

    outputs = llm_model.generate(inputs, sampling_params, use_tqdm=False)

    try:
        return json.loads(outputs[0].outputs[0].text)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse JSON")
        return {}


def execute_tool_call(tool_name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute a tool and return results."""
    executor = get_tool_executor(tool_name)
    if executor is None:
        return [{"error": f"Tool {tool_name} not found"}]

    try:
        return executor(**args)
    except Exception as e:
        return [{"error": str(e)}]


def _format_tool_descriptions(available_tools: list[str]) -> str:
    """Format tool descriptions for prompt."""
    tool_descriptions = []
    for tool_name in available_tools:
        if tool_name in TOOL_DEFINITIONS:
            tool = TOOL_DEFINITIONS[tool_name]
            tool_descriptions.append(
                f"- {tool_name}: {tool['description']}\n"
                f"  Parameters: {json.dumps(tool['parameters'], indent=2)}"
            )
    return "\n".join(tool_descriptions)


def _format_tool_history(history: list[dict[str, Any]]) -> str:
    """Format tool usage history for prompt."""
    if not history:
        return "No tools used yet."

    lines = []
    for entry in history:
        tool = entry.get("tool", "unknown")
        iter_num = entry.get("iteration", "?")
        results_count = entry.get("results_count", 0)
        selected = entry.get("selected_node", "none")
        lines.append(
            f"- {tool} (iter {iter_num}): {results_count} results, selected: {selected}"
        )
    return "\n".join(lines)


def _format_paper_list(
    papers: list[dict[str, Any]],
    include_abstracts: bool = True,
    include_summary: bool = False,
) -> str:
    """Format papers as readable text for prompt."""
    lines = []
    for paper in papers:
        paper_id = paper.get("id", "unknown")
        title = paper.get("title", "No title")
        abstract = paper.get("abstract", "") if include_abstracts else ""
        summary = paper.get("summary", "") if include_summary else ""

        lines.append(f"ID: {paper_id}")
        lines.append(f"Title: {title}")
        if abstract:
            lines.append(f"Abstract: {abstract}")
        if summary:
            lines.append(f"Summary: \n{summary}")
        lines.append("")

    return "\n".join(lines)


def agentic_explorer(
    start_nodes: list[str],
    iterations: int,
    tools: list[str],
    tool_selection_prompt_path: str,
    node_filtering_prompt_path: str,
    user_query: str = "",
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_results_per_tool: int = MAX_RESULTS_PER_TOOL,
    include_abstracts: bool = True,
    selected_nodes_count_low: int = 2,
    selected_nodes_count_high: int = 3,
    include_summary: bool = False,
) -> list[dict[str, Any]]:
    """Agentic graph exploration with 2-phase LLM calls.

    Phase 1: Tool selection - LLM chooses which tool to use.
    Phase 2: Node filtering - LLM filters results and selects next node.
    """
    llm_model, tok = _load_model(model_name)

    tool_select_config = load_prompt(tool_selection_prompt_path)
    node_filter_config = load_prompt(node_filtering_prompt_path)

    tool_select_schema_orig = tool_select_config.get("json_schema", {})
    tool_select_schema = copy.deepcopy(tool_select_schema_orig)
    tool_select_schema["properties"]["selected_tool"]["enum"] = tools

    node_filter_schema_orig = node_filter_config.get("json_schema", {})
    node_filter_schema = copy.deepcopy(node_filter_schema_orig)

    low = min(selected_nodes_count_low, max_results_per_tool)
    high = min(selected_nodes_count_high, max_results_per_tool)
    papers_range = f"{low}-{high}"

    node_filter_schema["properties"]["selected_nodes"][
        "description"
    ] = f"List of paper IDs to add to visited ({low}-{high} papers)"

    tool_section = _format_tool_descriptions(tools)

    all_papers = []
    all_paper_ids: set[str] = set()

    for start_id in start_nodes:
        visited: set[str] = {start_id}
        current_node = start_id
        tool_history: list[dict[str, Any]] = []

        results = execute_tool_call(
            "bfs_from_papers",
            {
                "start_paper_ids": [start_id],
                "max_level": 0,
                "direction": "both",
            },
        )
        if results and "error" not in results[0] and results[0].get("id"):
            start_paper = results[0]
            if start_paper.get("id") not in all_paper_ids:
                all_paper_ids.add(start_paper.get("id"))
                all_papers.append(start_paper)

        for iteration in range(iterations):
            print(f"\n{'='*60}")
            print(f"Start: {start_id} | Iteration {iteration + 1}/{iterations}")
            print(f"Current node: {current_node}")
            print(f"Visited: {len(visited)} nodes")
            print(f"{'='*60}")

            if len(tools) == 1:
                selected_tool = tools[0]
                tool_reason = "Only one tool available"
            else:
                visited_list = sorted(visited)
                visited_str = ", ".join(visited_list)
                tool_history_str = _format_tool_history(tool_history)

                tool_loop_descriptions = []
                for tool_name in tools:
                    if tool_name in TOOL_DEFINITIONS:
                        desc = TOOL_DEFINITIONS[tool_name].get("description", "")
                        tool_loop_descriptions.append(f"- {tool_name}: {desc}")
                tool_loop_text = "\n".join(tool_loop_descriptions)

                tool_select_context = f"""Tool usage history:
{tool_history_str}

Current exploring node: {current_node}
Visited paper IDs: [{visited_str}]
(User query: {user_query})

Available tools:
{tool_loop_text}

IMPORTANT: You have already visited these papers. Do NOT select them again unless there are no other unvisited papers available."""

                tool_select_prompt = format_prompt(
                    tool_select_config,
                    user_query,
                    papers_range,
                    tool_select_context,
                    tool_section,
                )

                print(f"Call 1/2: Tool selection...")
                print("=" * 60)
                print("TOOL SELECTION PROMPT:")
                print(tool_select_prompt)
                print("=" * 60)
                tool_result = _call_llm(
                    llm_model, tok, tool_select_prompt, tool_select_schema, temperature
                )

                selected_tool = tool_result.get("selected_tool")
                tool_reason = tool_result.get("reason", "No reason provided")

                if not selected_tool or selected_tool not in tools:
                    print(f"ERROR: Invalid tool selected: {selected_tool}")
                    continue

                print(f"Selected tool: {selected_tool}")
                print(f"Reason: {tool_reason}")

            if selected_tool == "bfs_from_papers":
                tool_args = {
                    "start_paper_ids": [current_node],
                    "max_level": 1,
                    "direction": "both",
                }
            elif selected_tool == "random_walk":
                tool_args = {
                    "start_paper_ids": [current_node],
                    "steps": max_results_per_tool,
                    "direction": "both",
                }
            elif selected_tool == "ppr":
                tool_args = {
                    "start_paper_ids": [current_node],
                    "top_n": max_results_per_tool,
                    "direction": "both",
                }
            else:
                tool_args = {"start_paper_ids": [current_node]}

            print(f"Executing tool: {selected_tool} with args: {tool_args}")
            results = execute_tool_call(selected_tool, tool_args)

            if not results or "error" in results[0]:
                error_msg = (
                    results[0].get("error", "Unknown error")
                    if results
                    else "No results"
                )
                print(f"ERROR: Tool execution failed: {error_msg}")
                continue

            results = [
                r for r in results if selected_tool == "ppr" or r.get("level", 0) > 0
            ]
            if len(results) > max_results_per_tool:
                results = random.sample(results, max_results_per_tool)

            paper_list_str = _format_paper_list(
                results, include_abstracts, include_summary
            )
            visited_list = sorted(visited)
            visited_str = ", ".join(visited_list)

            unvisited_in_results = [r for r in results if r.get("id") not in visited]

            if not unvisited_in_results:
                print("No unvisited papers in results - backtracking...")
                tool_history.append(
                    {
                        "iteration": iteration + 1,
                        "tool": selected_tool,
                        "results_count": len(results),
                        "selected_node": "BACKTRACK",
                    }
                )
                if visited:
                    current_node = random.choice(list(visited))
                    print(f"Backtracked to: {current_node}")
                    continue
                else:
                    print("No visited nodes to backtrack to - stopping.")
                    break

            node_filter_context = f"""Current node: {current_node}
Tool used: {selected_tool}

RESULTS FROM TOOL:
{paper_list_str}

VISITED PAPER IDs (DO NOT SELECT THESE):
[{visited_str}]

(User query: {user_query})

IMPORTANT: Select ONLY papers that are NOT in the visited list above."""

            node_filter_prompt = format_prompt(
                node_filter_config,
                user_query,
                papers_range,
                node_filter_context,
                "",
            )

            print(f"Call 2/2: Node filtering...")
            print("=" * 60)
            print("NODE FILTERING PROMPT:")
            print(node_filter_prompt)
            print("=" * 60)
            filter_result = _call_llm(
                llm_model, tok, node_filter_prompt, node_filter_schema, temperature
            )

            selected_nodes = filter_result.get("selected_nodes", [])
            current_node = filter_result.get("current_node")

            print(f"Selected nodes: {selected_nodes}")
            print(f"Current node: {current_node}")

            if not selected_nodes or not current_node:
                print("ERROR: No nodes selected")
                continue

            for node_id in selected_nodes:
                if node_id not in visited:
                    visited.add(node_id)
                    selected_paper = next(
                        (r for r in results if r.get("id") == node_id), None
                    )
                    if selected_paper and selected_paper.get("id") not in all_paper_ids:
                        all_paper_ids.add(selected_paper.get("id"))
                        all_papers.append(selected_paper)

            print(f"Added to visited. Total visited: {len(visited)}")

            if current_node in visited:
                print(
                    f"Current node {current_node} is already visited - backtracking..."
                )
                tool_history.append(
                    {
                        "iteration": iteration + 1,
                        "tool": selected_tool,
                        "results_count": len(results),
                        "selected_node": "BACKTRACK",
                    }
                )
                if visited:
                    current_node = random.choice(list(visited))
                    print(f"Backtracked to: {current_node}")
                else:
                    print("No nodes to backtrack to - stopping.")
                    break
                continue

            tool_history.append(
                {
                    "iteration": iteration + 1,
                    "tool": selected_tool,
                    "results_count": len(results),
                    "selected_node": current_node,
                }
            )

    print(f"\n{'='*60}")
    print(f"Exploration complete. Total papers found: {len(all_papers)}")
    print(f"{'='*60}")

    return all_papers