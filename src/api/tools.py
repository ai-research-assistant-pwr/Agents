"""OpenAI tool definitions and tool execution helpers for chat continuation."""

from api.models import SessionOut

CHAT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "compare_hypotheses",
            "description": (
                "Get a structured side-by-side comparison of the two generated hypotheses "
                "including their mechanisms, predictions, and key differences."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_knowledge_graph_context",
            "description": (
                "Retrieve the full knowledge graph context: discovered communities, "
                "key concepts, source papers, and inter-node relations."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_related_literature",
            "description": "Search for papers and concepts related to a topic in the knowledge graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The concept or topic to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def execute_tool(name: str, args: dict, session: SessionOut) -> str:
    if name == "compare_hypotheses":
        lines = []
        for h in session.hypotheses:
            lines += [
                f"Hypothesis {h.id} — {h.headline}",
                f"  Classification: {h.classification}",
                f"  Mechanism: {h.statement}",
                f"  Prediction: {h.falsifiablePrediction}",
                f"  Evidence: {', '.join(h.drawnFrom)}",
                "",
            ]
        if session.selectedId:
            lines.append(f"User selected Hypothesis {session.selectedId}.")
            if session.rationale:
                lines.append(f"Rationale: {session.rationale}")
        return "\n".join(lines)

    if name == "get_knowledge_graph_context":
        nodes_by_type: dict[str, list[str]] = {}
        if session.knowledgeGraph:
            for n in session.knowledgeGraph.nodes:
                nodes_by_type.setdefault(n.type, []).append(n.label)
        lines = [f"Communities ({len(session.exploration.communities)}): "
                 f"{', '.join(session.exploration.communities)}"]
        for t in ("concept", "paper"):
            if t in nodes_by_type:
                labels = nodes_by_type[t][:8]
                lines.append(f"{t.capitalize()}s: {', '.join(labels)}")
        lines.append(
            f"Total: {session.exploration.nodesTraversed} nodes, "
            f"{session.exploration.relations} relations"
        )
        return "\n".join(lines)

    if name == "search_related_literature":
        query = args.get("query", "")
        return (
            f"[Mock search for '{query}']\n"
            "• Induction heads and attention circuit formation (Olsson et al. 2022)\n"
            "• Grokking: generalisation beyond overfitting (Power et al. 2022)\n"
            "• Emergent abilities critique (Schaeffer et al. 2023)\n"
            "Set PIPELINE_USE_MOCK=false with a Weaviate instance for real results."
        )

    return f"Unknown tool: {name}"
