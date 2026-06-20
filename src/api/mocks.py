"""Mock pipeline data and replies used when `PIPELINE_USE_MOCK=true`."""

import time
from datetime import datetime, timezone
from uuid import uuid4

from api.models import (
    ExplorationStats,
    HypothesisOut,
    KGEdge,
    KGNode,
    KnowledgeGraph,
    ReasoningStep,
    SessionOut,
)

H_A = HypothesisOut(
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

H_B = HypothesisOut(
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

MOCK_TRACE = [
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

MOCK_KG = KnowledgeGraph(
    nodes=[
        KGNode(id="query", label="Research Query", type="query"),
        KGNode(id="c_mech", label="Mechanistic Interpretability", type="community"),
        KGNode(id="c_scale", label="Scaling Laws", type="community"),
        KGNode(id="c_train", label="Training Dynamics", type="community"),
        KGNode(id="c_eval", label="Evaluation Methods", type="community"),
        KGNode(id="n_ind", label="Induction Heads", type="concept"),
        KGNode(id="n_circ", label="Circuit Formation", type="concept"),
        KGNode(id="n_phase", label="Phase Transitions", type="concept"),
        KGNode(id="n_grok", label="Grokking", type="concept"),
        KGNode(id="n_icl", label="Emergent ICL", type="concept"),
        KGNode(id="n_metric", label="Metric Continuity", type="concept"),
        KGNode(id="n_data", label="Data Composition", type="concept"),
        KGNode(id="n_dist", label="Distributional Learning", type="concept"),
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


def mock_chat_reply(session: SessionOut, user_message: str) -> str:
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


def run_mock(question: str) -> SessionOut:
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
        hypotheses=[H_A, H_B],
        reasoningTrace=MOCK_TRACE,
        knowledgeGraph=MOCK_KG,
    )
