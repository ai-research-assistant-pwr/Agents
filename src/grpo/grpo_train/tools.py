"""
Weaviate search tool and agent tool definitions for the scientific workflow.
============================================================================

Tool schemas
------------
  send_to_generator  – Retriever sends its synthesis to the generator.
  search_papers      – Retriever searches Weaviate for one additional paper
                       (limited to ``state.retriever_search_limit`` calls per
                       retriever turn; counter resets at the start of each new
                       retriever turn).
  generate_hypotheses– Generator outputs the final hypothesis list (terminal).
  ask_retriever      – Generator asks the retriever a follow-up question
                       (limited to ``state.ask_retriever_limit`` uses per
                       trajectory).

Tool availability is determined by ``available_tools(state)``, which reads
limits from the state so they can be configured at runtime via workflow_args
without touching source code.
"""

import json
from typing import Any, Dict, List, Optional

import weaviate

from src.grpo.grpo_train.state import TrajectoryState


# ── tool schemas ──────────────────────────────────────────────────────────────
# Descriptions that depend on runtime limits are formatted lazily in
# build_tool_section so the correct numbers are always shown.

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "send_to_generator": {
        "description": (
            "Send your synthesis or focused answer to the generator agent. "
            "Use this when you have extracted the relevant information from the papers."
        ),
        "parameters": {
            "message": "string — your complete synthesis or answer text",
        },
    },
    "search_papers": {
        "description": (
            "Search the paper database for one additional paper relevant to a specific "
            "aspect of the query. Use this to fill evidence gaps before sending your "
            "synthesis. Your full prior reasoning and the search result will be preserved "
            "in context. {limit_note}"
        ),
        "parameters": {
            "query": "string — a focused search query targeting a specific topic or mechanism",
        },
    },
    "generate_hypotheses": {
        "description": (
            "Output the final list of scientific hypotheses. "
            "Use this when you have enough information to generate well-grounded hypotheses."
        ),
        "parameters": {
            "hypotheses": (
                "array of strings — each element is one complete, self-contained hypothesis"
            ),
        },
    },
    "ask_retriever": {
        "description": (
            "Ask the retriever a single focused follow-up question to fill a critical "
            "evidence gap. {limit_note}"
        ),
        "parameters": {
            "question": "string — your precise, specific question for the retriever",
        },
    },
}


def _render_description(tool_name: str, state: TrajectoryState) -> str:
    """Return the tool description with any runtime limit notes filled in."""
    desc = TOOL_SCHEMAS[tool_name]["description"]
    if tool_name == "search_papers":
        remaining = state.retriever_search_limit - state.tool_usage.get(
            "search_papers", 0
        )
        note = f"Allowed {remaining} more time(s) this retriever turn."
        return desc.format(limit_note=note)
    if tool_name == "ask_retriever":
        remaining = state.ask_retriever_limit - state.tool_usage.get("ask_retriever", 0)
        note = f"Allowed {remaining} more time(s) per trajectory."
        return desc.format(limit_note=note)
    return desc.format(limit_note="")  # no placeholder in other tools


def build_tool_section(tools: List[str], state: TrajectoryState) -> str:
    """Render the tool-call instruction block for the given tool names."""
    lines = [
        "## Tool Use\n",
        "After your thinking, you MUST output exactly one tool call using this format:\n",
        '<tool_call>{"name": "<tool_name>", "arguments": {<arguments as JSON>}}</tool_call>\n',
        "Do not output anything after the closing </tool_call> tag.\n",
        "\n### Available tools:\n",
    ]
    for name in tools:
        schema = TOOL_SCHEMAS[name]
        description = _render_description(name, state)
        param_str = json.dumps(schema["parameters"], indent=4)
        lines.append(f"**{name}**")
        lines.append(f"  {description}")
        lines.append(f"  Parameters:\n{param_str}\n")
    return "\n".join(lines)


def available_tools(state: TrajectoryState) -> List[str]:
    """Return the list of tool names the current agent may call this turn."""
    if state.current_agent == "retriever":
        tools = ["send_to_generator"]
        if state.tool_usage.get("search_papers", 0) < state.retriever_search_limit:
            tools.append("search_papers")
        return tools

    # generator
    tools = ["generate_hypotheses"]
    if state.tool_usage.get("ask_retriever", 0) < state.ask_retriever_limit:
        tools.append("ask_retriever")
    return tools


# ── Weaviate search ───────────────────────────────────────────────────────────


def search_weaviate(query: str, n: int, weaviate_url: str) -> List[Dict[str, str]]:
    """
    Search the Weaviate database for papers relevant to the given query.

    The query is embedded using Weaviate's built-in text2vec vectoriser
    (the same model used when papers were indexed), so no separate embedding
    call is required here.

    Parameters
    ----------
    query       : Natural-language search query (e.g. the user research prompt).
    n           : Maximum number of results to return.
    weaviate_url: Base URL of the Weaviate instance, e.g. "http://localhost:8080".

    Returns
    -------
    List of dicts with keys "id", "title", "summary".
    """
    client = weaviate.Client(weaviate_url)

    result = (
        client.query.get("Paper", ["paper_id", "title", "summary"])
        .with_near_text({"concepts": [query]})
        .with_limit(n)
        .do()
    )

    papers: List[Dict[str, str]] = []
    hits: List[Dict[str, Any]] = result.get("data", {}).get("Get", {}).get("Paper", [])
    for hit in hits:
        papers.append(
            {
                "id": hit.get("paper_id", ""),
                "title": hit.get("title", ""),
                "summary": hit.get("summary", ""),
            }
        )
    return papers


def search_papers_tool(query: str, weaviate_url: str) -> str:
    """
    Tool-call wrapper around ``search_weaviate``.

    Always retrieves exactly 1 paper and formats it as a human-readable string
    suitable for injection into the retriever's context as a tool-result message.

    Returns a plain string — either the formatted paper or an error note if
    Weaviate returned no results.
    """
    papers = search_weaviate(query, n=1, weaviate_url=weaviate_url)
    if not papers:
        return "No results found for the given query."
    p = papers[0]
    title = p.get("title", "Unknown title")
    summary = p.get("summary", "No summary available.")
    return f"[Search result]\nTitle: {title}\n\n{summary.strip()}"
