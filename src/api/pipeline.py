"""Pipeline entry point for the hypothesis generation API.

Set PIPELINE_USE_MOCK=false (plus the appropriate API keys and DB connections)
to swap in the real pipeline.  Mock mode is the default and requires no
external services.

Chat continuation uses OpenAI-compatible function/tool calling so the LLM
can retrieve structured context about the session while composing replies.
Any endpoint compatible with the OpenAI Chat Completions API works:
  - OpenAI (set OPENAI_API_KEY)
  - vLLM   (set OPENAI_BASE_URL=http://host:port/v1  and OPENAI_API_KEY=dummy)
  - Ollama  (set OPENAI_BASE_URL=http://localhost:11434/v1)
  - Groq, Together AI, Cerebras, etc.

See INTEGRATION.md for full setup instructions.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

from api.models import (
    ExplorationStats,
    HypothesisOut,
    KGEdge,
    KGNode,
    KnowledgeGraph,
    Message,
    ReasoningStep,
    SessionOut,
)

# ---------------------------------------------------------------------------
# Mock hypotheses
# ---------------------------------------------------------------------------

_H_A = HypothesisOut(
    id="A",
    classification="Mechanistic · capacity-side",
    headline="Phase transition in induction-head circuit formation",
    statement=(
        "Emergent in-context learning corresponds to a sharp phase transition in "
        "induction-head circuit formation. Below a critical product of depth and "
        "width, multi-step attention compositions are sample-inefficient and "
        "decoupled; above it, they crystallise into a stable circuit that supports "
        "in-context pattern matching at near-Bayes-optimal rates on synthetic tasks."
    ),
    drawnFrom=["induction heads", "grokking", "circuit formation", "phase transitions"],
    falsifiablePrediction=(
        "Synthetic-task ICL accuracy shows a sharp discontinuity locatable within "
        "a ±5% parameter band across model families."
    ),
)

_H_B = HypothesisOut(
    id="B",
    classification="Statistical · measurement-side",
    headline="Apparent emergence as evaluation discontinuity",
    statement=(
        "The emergence pattern is an artefact of discontinuous evaluation. "
        "Pretraining corpora cross a coverage-density threshold at which few-shot "
        "meta-patterns become reliably retrievable; underlying capacity itself "
        "improves smoothly. Replacing exact-match metrics with continuous proxies "
        "should reveal a smooth curve and no critical scale."
    ),
    drawnFrom=[
        "emergence critique",
        "metric continuity",
        "distributional learning",
        "data composition",
    ],
    falsifiablePrediction=(
        "No smooth continuous metric reveals a discontinuity at the same critical "
        "scale across at least three model families."
    ),
)

# ---------------------------------------------------------------------------
# Mock reasoning trace
# ---------------------------------------------------------------------------

_MOCK_TRACE = [
    ReasoningStep(
        step="Search",
        description="Semantic search over 50 K paper embeddings",
        durationSec=0.12,
        details=(
            "Found 10 candidate papers via Weaviate vector search using a "
            "Qwen3-Embedding-4B encoder.  Top result cosine similarity: 0.91."
        ),
    ),
    ReasoningStep(
        step="Explorer",
        description="BFS traversal of the Neo4j knowledge graph",
        durationSec=2.31,
        details=(
            "Starting from 10 seed papers, traversed 127 nodes and 43 typed "
            "relations at depth ≤ 2.  Ran Louvain clustering — discovered "
            "4 communities: Mechanistic interpretability, Scaling laws, "
            "Training dynamics, Evaluation methods."
        ),
    ),
    ReasoningStep(
        step="Retriever",
        description="LLM filtering of retrieved context",
        durationSec=1.85,
        details=(
            "Condensed 127 nodes into a 2 400-token structured context summary, "
            "preserving two evidence threads: mechanistic (induction-head circuits) "
            "and statistical (metric discontinuity critique)."
        ),
    ),
    ReasoningStep(
        step="Generator",
        description="Hypothesis synthesis from curated context",
        durationSec=3.42,
        details=(
            "Prompted the LLM with the research question and the retriever summary. "
            "Generated 2 competing hypotheses grounded in distinct subgraphs: "
            "capacity-side (A) vs. measurement-side (B)."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Mock knowledge graph — includes paper nodes with metadata
# ---------------------------------------------------------------------------

_MOCK_KG = KnowledgeGraph(
    nodes=[
        KGNode(id="query", label="Research Query", type="query"),
        # Communities
        KGNode(id="c_mech", label="Mechanistic Interpretability", type="community"),
        KGNode(id="c_scale", label="Scaling Laws", type="community"),
        KGNode(id="c_train", label="Training Dynamics", type="community"),
        KGNode(id="c_eval", label="Evaluation Methods", type="community"),
        # Concepts
        KGNode(id="n_ind", label="Induction Heads", type="concept"),
        KGNode(id="n_circ", label="Circuit Formation", type="concept"),
        KGNode(id="n_phase", label="Phase Transitions", type="concept"),
        KGNode(id="n_grok", label="Grokking", type="concept"),
        KGNode(id="n_icl", label="Emergent ICL", type="concept"),
        KGNode(id="n_metric", label="Metric Continuity", type="concept"),
        KGNode(id="n_data", label="Data Composition", type="concept"),
        KGNode(id="n_dist", label="Distributional Learning", type="concept"),
        # Papers
        KGNode(
            id="p_olsson",
            label="Olsson et al. 2022",
            type="paper",
            paperId="arXiv:2209.11895",
            abstract=(
                "We present evidence that 'induction heads' are the mechanism by which "
                "transformers perform in-context learning. We find a striking correlation "
                "between the phase change in loss curves described in previous work and "
                "the formation of induction heads. The mechanism is present across a wide "
                "range of transformer models."
            ),
            summary=(
                "Demonstrates that induction heads form during a sharp phase transition "
                "in training and are the core circuit enabling in-context pattern matching "
                "across transformer families."
            ),
        ),
        KGNode(
            id="p_wei",
            label="Wei et al. 2022",
            type="paper",
            paperId="arXiv:2206.07682",
            abstract=(
                "We investigate emergent abilities of large language models — abilities "
                "not present in smaller-scale models that appear when model scale reaches "
                "a critical threshold. We discuss potential explanations and implications "
                "of these phenomena for future research."
            ),
            summary=(
                "Documents over 100 emergent abilities that appear unpredictably in large "
                "models and proposes that scale is the primary driver."
            ),
        ),
        KGNode(
            id="p_schaeffer",
            label="Schaeffer et al. 2023",
            type="paper",
            paperId="arXiv:2304.15004",
            abstract=(
                "Are emergent abilities of large language models a mirage? We present an "
                "alternative explanation: apparent emergent abilities are a consequence of "
                "the researcher's choice of metric rather than fundamental changes in model "
                "behaviour. Using continuous metrics, model improvements are smooth and "
                "predictable."
            ),
            summary=(
                "Argues that emergence is an artifact of discontinuous evaluation metrics; "
                "continuous metrics reveal smooth, predictable capability scaling."
            ),
        ),
    ],
    edges=[
        KGEdge(source="query", target="c_mech", relation="explores"),
        KGEdge(source="query", target="c_scale", relation="explores"),
        KGEdge(source="query", target="c_train", relation="explores"),
        KGEdge(source="query", target="c_eval", relation="explores"),
        KGEdge(source="c_mech", target="n_ind", relation="contains"),
        KGEdge(source="c_mech", target="n_circ", relation="contains"),
        KGEdge(source="c_mech", target="n_phase", relation="contains"),
        KGEdge(source="c_mech", target="p_olsson", relation="cites"),
        KGEdge(source="c_scale", target="n_grok", relation="contains"),
        KGEdge(source="c_scale", target="n_icl", relation="contains"),
        KGEdge(source="c_scale", target="p_wei", relation="cites"),
        KGEdge(source="c_train", target="n_phase", relation="supports"),
        KGEdge(source="c_eval", target="n_metric", relation="contains"),
        KGEdge(source="c_eval", target="n_data", relation="contains"),
        KGEdge(source="c_eval", target="n_dist", relation="contains"),
        KGEdge(source="c_eval", target="p_schaeffer", relation="cites"),
        KGEdge(source="n_ind", target="n_icl", relation="causes"),
        KGEdge(source="n_phase", target="n_grok", relation="predicts"),
        KGEdge(source="n_metric", target="n_icl", relation="explains"),
        KGEdge(source="p_olsson", target="n_ind", relation="studies"),
        KGEdge(source="p_wei", target="n_icl", relation="studies"),
        KGEdge(source="p_schaeffer", target="n_metric", relation="studies"),
    ],
)

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

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


def _execute_tool(name: str, args: dict, session: SessionOut) -> str:
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
        # Real mode: call Weaviate / Neo4j full-text search
        return (
            f"[Mock search for '{query}']\n"
            "• Induction heads and attention circuit formation (Olsson et al. 2022)\n"
            "• Grokking: generalisation beyond overfitting (Power et al. 2022)\n"
            "• Emergent abilities critique (Schaeffer et al. 2023)\n"
            "Set PIPELINE_USE_MOCK=false with a Weaviate instance for real results."
        )

    return f"Unknown tool: {name}"


def _build_system_prompt(session: SessionOut) -> str:
    lines = [
        "You are a scientific research assistant helping to develop a hypothesis "
        "generated by an AI pipeline.",
        "",
        f"## Research Question\n{session.question}",
        "",
        "## Generated Hypotheses",
    ]
    for h in session.hypotheses:
        lines += [
            f"\n### Hypothesis {h.id} — {h.headline}",
            f"Classification: {h.classification}",
            f"Statement: {h.statement}",
            f"Falsifiable prediction: {h.falsifiablePrediction}",
            f"Evidence drawn from: {', '.join(h.drawnFrom)}",
        ]

    if session.selectedId:
        selected = next((h for h in session.hypotheses if h.id == session.selectedId), None)
        lines += [
            "",
            f"## User's Selection",
            f"The user chose Hypothesis {session.selectedId}"
            + (f" — {selected.headline}" if selected else ""),
        ]
        if session.rationale:
            lines.append(f"Rationale: {session.rationale}")

    lines += [
        "",
        "## Knowledge Graph Context",
        f"Communities explored: {', '.join(session.exploration.communities)}",
        f"Nodes traversed: {session.exploration.nodesTraversed}",
        f"Relations: {session.exploration.relations}",
        "",
        "## Instructions",
        "- Help the user develop and extend their chosen hypothesis through rigorous scientific discussion.",
        "- Reference evidence from the knowledge graph when relevant.",
        "- Use the available tools when you need to retrieve additional context.",
        "- Suggest specific, falsifiable experiments where appropriate.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat entry point
# ---------------------------------------------------------------------------


def generate_chat_reply(session: SessionOut, user_message: str) -> str:
    if os.getenv("PIPELINE_USE_MOCK", "true").lower() != "false":
        return _mock_chat_reply(session, user_message)
    return _real_chat_reply(session, user_message)


def _mock_chat_reply(session: SessionOut, user_message: str) -> str:
    selected = next((h for h in session.hypotheses if h.id == session.selectedId), None)
    headline = selected.headline if selected else "the hypothesis"
    turn = len([m for m in session.messages if m.role == "user"])
    intro = f"(Turn {turn}) Building on **{headline}**" if turn > 1 else f"Building on **{headline}**"
    snippet = user_message[:80]
    return (
        f'{intro}: your question — "{snippet}" — points toward an important extension. '
        "The key mechanism connects to the evidence from the explorer subgraph: the same "
        "circuits that stabilise in-context learning also gate the phase transition you "
        f"identified. The communities ({', '.join(session.exploration.communities[:2])}) "
        "provide converging evidence.\n\n"
        "*[Mock response. Set PIPELINE_USE_MOCK=false with an OpenAI-compatible endpoint "
        "for real AI-generated continuations — see INTEGRATION.md.]*"
    )


def _real_chat_reply(session: SessionOut, user_message: str) -> str:
    """Context-aware chat using OpenAI-compatible tool calling.

    Works with any OpenAI-compatible endpoint:
      - OpenAI:  set OPENAI_API_KEY
      - vLLM:    set OPENAI_BASE_URL=http://host:8000/v1  OPENAI_API_KEY=dummy
      - Ollama:  set OPENAI_BASE_URL=http://localhost:11434/v1

    Model is read from CHAT_MODEL (default: gpt-4o-mini).
    """
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
        base_url=os.getenv("OPENAI_BASE_URL"),  # None → use OpenAI directly
    )
    model = os.getenv("CHAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    messages: list[dict] = [{"role": "system", "content": _build_system_prompt(session)}]

    # Replay last 20 messages (10 turns) to stay within context limits
    for msg in session.messages[-20:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": user_message})

    # Tool-calling loop (capped at 5 iterations to prevent runaway)
    last_content = ""
    for _ in range(5):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=CHAT_TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        last_content = choice.message.content or ""

        if choice.finish_reason == "stop" or not choice.message.tool_calls:
            return last_content

        # Append assistant turn with tool_calls
        messages.append(
            {
                "role": "assistant",
                "content": last_content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ],
            }
        )

        # Execute each tool and append results
        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _execute_tool(tc.function.name, args, session)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return last_content or "Context gathered. Please ask your follow-up question."


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(question: str) -> SessionOut:
    if os.getenv("PIPELINE_USE_MOCK", "true").lower() != "false":
        return _run_mock(question)
    return _run_real(question)


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------


def _run_mock(question: str) -> SessionOut:
    t0 = time.monotonic()
    session_id = str(uuid4())
    elapsed = round(time.monotonic() - t0 + 0.1, 3)

    return SessionOut(
        sessionId=session_id,
        question=question,
        createdAt=datetime.now(timezone.utc).isoformat(),
        exploration=ExplorationStats(
            nodesTraversed=127,
            relations=43,
            sourcePapers=8,
            clusters=4,
            communities=[
                "Mechanistic interpretability",
                "Scaling laws",
                "Training dynamics",
                "Evaluation methods",
            ],
            durationSec=elapsed,
        ),
        hypotheses=[_H_A, _H_B],
        reasoningTrace=_MOCK_TRACE,
        knowledgeGraph=_MOCK_KG,
    )


# ---------------------------------------------------------------------------
# Real pipeline implementation
# ---------------------------------------------------------------------------


def _run_real(question: str) -> SessionOut:
    """Run the full pipeline with real models and knowledge graph.

    Requires: config/app/config.yaml, API keys, and DB connections.
    See INTEGRATION.md for full setup instructions.
    """
    _ensure_src_on_path()

    from app.app import App  # noqa: PLC0415
    from app.config import load_config  # noqa: PLC0415
    from app.retriever.api_llm_retriever import APILLMRetriever  # noqa: PLC0415
    from app.generator.api_llm_generator import APILLMGenerator  # noqa: PLC0415

    config = load_config("config/app/config.yaml")
    provider = config.get("api_client", {}).get("type", "openai")
    model = config.get("api_client", {}).get("model", "gpt-4o-mini")
    api_client = _build_api_client(provider, model)

    explorer = _build_explorer(config)
    search_explorer = _build_search(config)

    app = App(
        search_explorer=search_explorer,
        explorer=explorer,
        retriever=APILLMRetriever(api_client=api_client),
        generator=APILLMGenerator(api_client=api_client),
        config_path="config/app/config.yaml",
    )

    t0 = time.monotonic()
    result = app.run(question)
    elapsed = round(time.monotonic() - t0, 3)

    trace = [
        ReasoningStep(
            step="Pipeline",
            description="Full search → explorer → retriever → generator run",
            durationSec=elapsed,
            details=f"Model: {result.metadata.get('model', 'unknown')}",
        )
    ]

    return SessionOut(
        sessionId=str(uuid4()),
        question=question,
        createdAt=datetime.now(timezone.utc).isoformat(),
        exploration=ExplorationStats(durationSec=elapsed),
        hypotheses=_wrap_hypotheses(result.hypotheses),
        reasoningTrace=trace,
    )


def _build_api_client(provider: str, model: str):
    if provider in ("openai", "openai_compatible"):
        from app.api_client.openai_compatible_client import OpenAICompatibleClient  # noqa: PLC0415
        return OpenAICompatibleClient(model=model)
    if provider == "google":
        from app.api_client.google_client import GoogleAPIClient  # noqa: PLC0415
        return GoogleAPIClient(model=model)
    if provider == "cerebras":
        from app.api_client.cerebras_client import CerebrasAPIClient  # noqa: PLC0415
        return CerebrasAPIClient(model=model)
    raise ValueError(f"Unknown provider: {provider!r}")


def _build_explorer(config: dict):
    from app.explorer.const_explorer import ConstExplorer  # noqa: PLC0415

    t = config.get("explorer", {}).get("type", "const")
    if t == "neo4j_bfs":
        from app.explorer.neo4j_bfs_explorer import Neo4jBFSExplorer  # noqa: PLC0415
        return Neo4jBFSExplorer(config)
    if t == "neo4j_pagerank":
        from app.explorer.neo4j_pagerank_explorer import Neo4jPageRankExplorer  # noqa: PLC0415
        return Neo4jPageRankExplorer(config)
    if t == "neo4j_random_walk":
        from app.explorer.neo4j_random_walk_explorer import Neo4jRandomWalkExplorer  # noqa: PLC0415
        return Neo4jRandomWalkExplorer(config)
    if t == "agentic":
        from app.explorer.agentic_explorer import AgenticExplorer  # noqa: PLC0415
        return AgenticExplorer(config)
    return ConstExplorer()


def _build_search(config: dict):
    t = config.get("search", {}).get("type", "const")
    if t == "weaviate":
        from app.explorer.weaviate_search_explorer import WeaviateSearchExplorer  # noqa: PLC0415
        return WeaviateSearchExplorer(config)
    return _NullSearch()


class _NullSearch:
    def search(self, prompt: str) -> list[str]:  # noqa: ARG002
        return []


def _ensure_src_on_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _wrap_hypotheses(raw: list[str]) -> list[HypothesisOut]:
    ids = ["A", "B"]
    out: list[HypothesisOut] = []
    for i, text in enumerate(raw[:2]):
        headline = text.split(". ", 1)[0].strip()[:120]
        out.append(
            HypothesisOut(
                id=ids[i],  # type: ignore[arg-type]
                classification=f"Hypothesis {ids[i]}",
                headline=headline,
                statement=text,
                drawnFrom=[],
                falsifiablePrediction="",
            )
        )
    while len(out) < 2:
        idx = len(out)
        out.append(
            HypothesisOut(
                id=ids[idx],  # type: ignore[arg-type]
                classification=f"Hypothesis {ids[idx]}",
                headline="(not generated)",
                statement="",
                drawnFrom=[],
                falsifiablePrediction="",
            )
        )
    return out
