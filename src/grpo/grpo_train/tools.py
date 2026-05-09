"""
Weaviate search tool and agent tool definitions for the scientific workflow.
============================================================================

Tool schemas
------------
  send_to_generator   – Retriever sends its synthesis to the generator.
  generate_hypotheses – Generator outputs the final hypothesis list (terminal).
  ask_retriever       – Generator asks the retriever a follow-up question
                        (limited to ``ASK_RETRIEVER_LIMIT`` uses per trajectory).

Tool availability is determined by ``available_tools(state)`` which enforces
per-agent and per-trajectory limits.
"""

import json
from typing import Any, Dict, List

import weaviate

from src.grpo.grpo_train.state import TrajectoryState


# ── tool limits ───────────────────────────────────────────────────────────────

ASK_RETRIEVER_LIMIT: int = 1

# ── tool schemas ──────────────────────────────────────────────────────────────

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
            f"evidence gap. Allowed at most {ASK_RETRIEVER_LIMIT} time(s) per trajectory."
        ),
        "parameters": {
            "question": "string — your precise, specific question for the retriever",
        },
    },
}


def build_tool_section(tools: List[str]) -> str:
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
        param_str = json.dumps(schema["parameters"], indent=4)
        lines.append(f"**{name}**")
        lines.append(f"  {schema['description']}")
        lines.append(f"  Parameters:\n{param_str}\n")
    return "\n".join(lines)


def available_tools(state: TrajectoryState) -> List[str]:
    """Return the list of tool names the current agent may call this turn."""
    if state.current_agent == "retriever":
        return ["send_to_generator"]

    # generator always has generate_hypotheses; ask_retriever only while under limit
    tools = ["generate_hypotheses"]
    if state.tool_usage.get("ask_retriever", 0) < ASK_RETRIEVER_LIMIT:
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
