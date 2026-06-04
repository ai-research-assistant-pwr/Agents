import csv
import json
import random
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from app.api_client.base import BaseAPIClient, Message
from app.explorer.tools.tool_definitions import TOOL_DEFINITIONS, get_tool_executor


class _ToolSelection(BaseModel):
    selected_tool: str
    reason: str


class _NodeFiltering(BaseModel):
    selected_nodes: list[int]
    current_node: int
    reason: str


MAX_ITERATIONS = 5
MAX_RESULTS_PER_TOOL = 10


PAPER_METRICS_MAP = {}

with open("data/citations_map.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        paper_id = str(row["paperId"])

        influential = (
            int(row["influentialCitationCount"])
            if row["influentialCitationCount"]
            else 0
        )

        citations = int(row["citationCount"]) if row["citationCount"] else 0

        date_str = row["publicationDate"]
        year = (
            int(date_str[:4])
            if date_str and len(date_str) >= 4 and date_str[:4].isdigit()
            else 0
        )

        PAPER_METRICS_MAP[paper_id] = (influential, citations, year)


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


def _call_api_llm(
    api_client: BaseAPIClient,
    prompt: str,
    response_schema: type[BaseModel] | None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Call LLM via API client and return parsed result as dict."""
    messages = [Message(role="user", content=prompt)]
    result = api_client.call(
        messages, response_schema=response_schema, temperature=temperature
    )
    if response_schema is not None:
        return result.content.model_dump()
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
    for i, paper in enumerate(papers):
        paper_id = paper.get("id", "unknown")
        title = paper.get("title", "No title")
        abstract = paper.get("abstract", "") if include_abstracts else ""
        summary = paper.get("summary", "") if include_summary else ""

        lines.append(f"[{i}] ID: {paper_id}")
        lines.append(f"    Title: {title}")
        if abstract:
            lines.append(f"    Abstract: {abstract}")
        if summary:
            lines.append(f"    Summary: \n{summary}")
        lines.append("")

    return "\n".join(lines)


def agentic_explorer(
    start_nodes: list[str],
    iterations: int,
    tools: list[str],
    tool_selection_prompt_path: str,
    node_filtering_prompt_path: str,
    api_client: BaseAPIClient,
    user_query: str = "",
    max_results_per_tool: int = 10,
    include_abstracts: bool = True,
    selected_nodes_count_low: int = 2,
    selected_nodes_count_high: int = 3,
    include_summary: bool = False,
    temperature: float | None = None,
) -> list[dict[str, Any]]:
    """Agentic graph exploration with 2-phase LLM calls.

    Phase 1: Tool selection - LLM chooses which tool to use.
    Phase 2: Node filtering - LLM filters results and selects next node.
    """
    tool_select_config = load_prompt(tool_selection_prompt_path)
    node_filter_config = load_prompt(node_filtering_prompt_path)

    low = min(selected_nodes_count_low, max_results_per_tool)
    high = min(selected_nodes_count_high, max_results_per_tool)
    papers_range = f"{low}-{high}"

    tool_section = _format_tool_descriptions(tools)

    all_papers = []
    all_paper_ids: set[str] = set()

    for start_id in start_nodes:
        visited: dict[str, str] = {}
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
            visited[start_paper["id"]] = start_paper.get("title", "")
            if start_paper.get("id") not in all_paper_ids:
                all_paper_ids.add(start_paper.get("id"))
                all_papers.append(start_paper)

        for iteration in range(iterations):
            print(f"\n{'='*60}")
            print(f"Start: {start_id} | Iteration {iteration + 1}/{iterations}")
            print(f"Current node: {current_node}")
            print(
                f"Visited: {len(visited)} nodes | Collected papers: {len(all_papers)}"
            )
            print(f"{'='*60}")

            if len(tools) == 1:
                selected_tool = tools[0]
                tool_reason = "Only one tool available"
            else:
                visited_str = "\n".join(
                    [
                        f"- ID: {pid} | Title: {title}"
                        for pid, title in sorted(visited.items())
                    ]
                )
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
Visited papers:
{visited_str}
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

                tool_result = _call_api_llm(
                    api_client,
                    tool_select_prompt,
                    _ToolSelection,
                    temperature=temperature,
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

            # -------------------------------------------------------------------------
            # POPRAWKA 2: Mądrzejsze filtrowanie. Odcinamy level 0 tylko z BFS,
            # by nie niszczyć wyników zrestartowanego Random Walk
            # -------------------------------------------------------------------------
            results = [
                r
                for r in results
                if selected_tool != "bfs_from_papers" or r.get("level", 0) > 0
            ]

            if len(results) > max_results_per_tool:
                # Pobieramy krotkę z RAM. Jeśli węzła nie ma w słowniku, dajemy mu (0, 0, 0)
                results.sort(
                    key=lambda x: PAPER_METRICS_MAP.get(str(x.get("id")), (0, 0, 0)),
                    reverse=True,
                )
                results = results[:max_results_per_tool]

            paper_list_str = _format_paper_list(
                results, include_abstracts, include_summary
            )
            visited_str = "\n".join(
                [
                    f"- ID: {pid} | Title: {title}"
                    for pid, title in sorted(visited.items())
                ]
            )

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

RESULTS FROM TOOL (index -> paper):
{paper_list_str}

VISITED PAPERS (DO NOT SELECT THESE):
{visited_str}

(User query: {user_query})

IMPORTANT: Select ONLY papers that are NOT in the visited list above."""

            node_filter_prompt = format_prompt(
                node_filter_config,
                user_query,
                papers_range,
                node_filter_context,
                "",
            )

            filter_result = _call_api_llm(
                api_client,
                node_filter_prompt,
                _NodeFiltering,
                temperature=temperature,
            )

            selected_indices = filter_result.get("selected_nodes", [])
            current_node_index = filter_result.get("current_node")

            selected_nodes = []
            for idx in selected_indices:
                if isinstance(idx, int) and 0 <= idx < len(results):
                    node_id = str(results[idx].get("id"))
                    if node_id and node_id not in visited:
                        selected_nodes.append(node_id)

            if isinstance(current_node_index, int) and 0 <= current_node_index < len(
                results
            ):
                proposed_current_node = str(results[current_node_index].get("id"))
            else:
                proposed_current_node = None

            print(f"Selected indices: {selected_indices} -> IDs: {selected_nodes}")
            print(f"Current index: {current_node_index} -> ID: {proposed_current_node}")

            if not selected_nodes or not proposed_current_node:
                print("ERROR: No valid nodes selected by LLM")
                if visited:
                    current_node = random.choice(list(visited))
                    print(f"Backtracked to: {current_node}")
                    continue
                else:
                    break

            # -------------------------------------------------------------------------
            # POPRAWKA 3: Logika gromadzenia węzłów i weryfikacja current_node
            # -------------------------------------------------------------------------
            if proposed_current_node in visited:
                print(
                    f"LLM Error: Proposed node {proposed_current_node} is already visited - backtracking..."
                )
                tool_history.append(
                    {
                        "iteration": iteration + 1,
                        "tool": selected_tool,
                        "results_count": len(results),
                        "selected_node": "BACKTRACK_LLM_ERROR",
                    }
                )
                current_node = random.choice(list(visited))
                continue

            # Ustawiamy nowy, bezpieczny current_node
            current_node = proposed_current_node

            # 1. Dodajemy do koszyka i odwiedzonych węzły poboczne (sąsiedztwo)
            for node_id in selected_nodes:
                if node_id == current_node:
                    continue  # Obsłużymy go za chwilę osobno

                selected_paper = next(
                    (r for r in results if r.get("id") == node_id), None
                )
                visited[node_id] = (
                    selected_paper.get("title", "") if selected_paper else ""
                )
                if selected_paper and selected_paper.get("id") not in all_paper_ids:
                    all_paper_ids.add(selected_paper.get("id"))
                    all_papers.append(selected_paper)

            # 2. DODAJEMY CURRENT NODE DO ODWIEDZONYCH (Kluczowe zabezpieczenie przed pętlą!)
            current_paper = next(
                (r for r in results if r.get("id") == current_node), None
            )
            visited[current_node] = (
                current_paper.get("title", "") if current_paper else ""
            )
            if current_paper and current_paper.get("id") not in all_paper_ids:
                all_paper_ids.add(current_paper.get("id"))
                all_papers.append(current_paper)

            print(f"Nodes successfully added. Total visited: {len(visited)}")

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
