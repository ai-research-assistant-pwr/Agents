"""
Run the retriever-message agent (Qwen3-32B via vLLM) over pre-generated queries.

For each query the script:
  1. Looks up the referenced neighbour papers (capped at --max-papers, default 8).
  2. Builds a messages list (system + user) using the retriever_message_system prompt
     and formats the neighbour papers as search results.
  3. Calls Qwen/Qwen3-32B hosted locally via vLLM (OpenAI-compatible API).
  4. Saves the query, reasoning trace, and synthesised message to a CSV.

Prerequisites – start the vLLM server first:
    vllm serve Qwen/Qwen3-32B \\
      --tensor-parallel-size 4 \\
      --reasoning-parser qwen3

Usage:
    python scripts/rl/run_retriever_inference.py \\
        --queries data/rl_queries.csv \\
        --out data/rl_retriever_outputs.csv

    python scripts/rl/run_retriever_inference.py \\
        --queries data/rl_queries.csv \\
        --out data/rl_retriever_outputs.csv \\
        --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen3-32B \\
        --max-papers 8 \\
        --workers 8 \\
        --n 100
"""

import argparse
import csv
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]

csv.field_size_limit(10_000_000)

PAPERS_CSV = ROOT / "data/11_neo4j_papers.csv"

# ---------------------------------------------------------------------------
# Prompt (copied from src/grpo/grpo_train/agents.py – AgentPrompts.retriever_message_system)
# ---------------------------------------------------------------------------

RETRIEVER_MESSAGE_SYSTEM = (
    "You are an Expert Scientific Retriever Agent. You have just performed a "
    "literature search and must now synthesize the results for the hypothesis generator.\n\n"
    "## Your Objectives\n\n"
    "1. Extract the most relevant information for the research query from the search results.\n"
    "2. Capture key scientific elements: core findings, causal mechanisms, relationships "
    "between variables, boundary conditions, and open questions.\n"
    "3. Distinguish established findings from speculative claims.\n"
    "4. Note any gaps or contradictions in the evidence.\n"
    "5. Describe methodologies, datasets, and experimental setups in sufficient detail.\n"
    "6. Explain all technical terms, acronyms, and domain-specific concepts so the "
    "generator can understand them without consulting the source papers.\n\n"
    "## Output Format\n\n"
    "Output ONLY your synthesis — a rich, detailed, well-organized set of paragraphs. "
    "Your synthesis MUST be long and comprehensive: aim for at least 20–30 sentences. "
    "Cover findings from multiple retrieved papers, elaborate on mechanisms and evidence, "
    "and surface nuances, contradictions, and open questions. "
    "Scientific precision and depth are more important than brevity — do NOT summarize to a few sentences. "
    "Do NOT generate hypotheses yourself. "
    "Do NOT include any preamble like 'Here is my synthesis:'."
)

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _format_search_results(papers: list[dict]) -> str:
    parts = []
    for i, p in enumerate(papers, 1):
        title = p["title"] or "(no title)"
        body = p["summary"] or p["abstract"] or "(no content available)"
        parts.append(f"[Search result {i}]\n**{title}**\n{body}")
    return "\n\n".join(parts) if parts else "(No search results available.)"


def build_messages(query: str, neighbour_papers: list[dict]) -> list[dict]:
    result_block = _format_search_results(neighbour_papers)
    user_content = f"Research query:\n{query}\n\nSearch results:\n{result_block}"
    return [
        {"role": "system", "content": RETRIEVER_MESSAGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_papers() -> dict[str, dict]:
    papers: dict[str, dict] = {}
    with open(PAPERS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["paperId:ID(Paper)"]
            papers[pid] = {
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "summary": row.get("summary", ""),
            }
    return papers


def load_queries(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> tuple[str, str]:
    """Return (reasoning, message) from the model."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.95,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
    )
    choice = response.choices[0].message
    reasoning = getattr(choice, "reasoning", "") or ""
    content = choice.content or ""
    return reasoning.strip(), content.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retriever-message inference with Qwen3-32B over pre-generated queries."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=ROOT / "data/rl_queries.csv",
        help="Input queries CSV (output of create_queries.py).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/rl_retriever_outputs.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible base URL (default: http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name as registered in vLLM (default: Qwen/Qwen3-32B).",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=8,
        help="Maximum number of neighbour papers to include as context (default: 8).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of parallel inference threads (default: 64).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Maximum number of queries to process (default: all).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for neighbour sampling."
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    client = OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=3600)

    print("Loading papers …")
    papers = load_papers()
    print(f"  {len(papers)} papers loaded.")

    print(f"Loading queries from {args.queries} …")
    queries = load_queries(args.queries)
    print(f"  {len(queries)} queries loaded.")

    if args.n is not None:
        queries = queries[: args.n]
        print(f"  Limiting to {len(queries)} queries (--n {args.n}).")

    if not queries:
        print("No queries found. Exiting.")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "paper_id",
        "paper_title",
        "user_query",
        "referenced_paper_ids",
        "input_messages",
        "reasoning",
        "retriever_message",
    ]

    write_lock = threading.Lock()
    completed = 0
    total = len(queries)

    def process(row: dict, index: int) -> bool:
        paper_id = row.get("paper_id", "")
        paper_title = row.get("paper_title", "")
        user_query = row.get("user_query", "").strip()
        ref_ids_raw = row.get("referenced_paper_ids", "")

        ref_ids = [r for r in ref_ids_raw.split("|") if r] if ref_ids_raw else []

        neighbour_papers = [papers[rid] for rid in ref_ids if rid in papers]
        if len(neighbour_papers) > args.max_papers:
            neighbour_papers = rng.sample(neighbour_papers, args.max_papers)

        print(
            f"\n[{index}/{total}] {paper_title[:70]!r} "
            f"({len(neighbour_papers)} neighbour papers)"
        )

        messages = build_messages(user_query, neighbour_papers)

        try:
            reasoning, message = call_model(
                client, args.model, messages, args.max_tokens
            )
            print(
                f"  reasoning: {len(reasoning)} chars | message: {len(message)} chars"
            )
        except Exception as exc:
            print(f"  ERROR: {exc} — skipping.")
            return False

        out_row = {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "user_query": user_query,
            "referenced_paper_ids": ref_ids_raw,
            "input_messages": json.dumps(messages, ensure_ascii=False),
            "reasoning": reasoning,
            "retriever_message": message,
        }
        with write_lock:
            writer.writerow(out_row)
            f.flush()
        return True

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process, row, i): row
                for i, row in enumerate(queries, 1)
            }
            for future in as_completed(futures):
                if future.result():
                    completed += 1

    print(f"\nDone. {completed}/{total} samples saved to {args.out}")


if __name__ == "__main__":
    main()
