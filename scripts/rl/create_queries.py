"""
Generate a dataset of user queries derived directly from target papers.

Unlike create_grounded_dataset.py, this script does not use neighbouring/referenced
papers for context. Instead, the LLM reads the target paper itself, internally
identifies the paper's main hypothesis, and then formulates a query for which
that hypothesis would be the ideal answer.

Usage:
    python scripts/rl/create_queries.py --n 100 --out data/rl_queries.csv
    python scripts/rl/create_queries.py --n 100 --out data/rl_queries.csv --model gpt-4o-mini
"""

import argparse
import csv
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]

csv.field_size_limit(10_000_000)

PAPERS_CSV = ROOT / "data/graph/11_neo4j_papers.csv"
REFS_CSV = ROOT / "data/graph/11_neo4j_references.csv"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QUERY_SYSTEM_PROMPT = """
Role: You are generating training queries for a scientific hypothesis generation system.
Input: The abstract and summary of a research paper.
Task: First, identify the main hypothesis of the paper. Then, write a single-sentence \
query a scientist would naturally ask, for which that hypothesis is the ideal answer.

Step 1 — Identify the hypothesis:
- Determine the core testable claim, prediction, or novel contribution the paper proposes.
- Focus on what the authors argue, claim to demonstrate, or set out to test — not \
  background facts or general statements.
- If the paper introduces a new method, the hypothesis should center on describing \
  that method and why it is expected to work, not merely on its results.

Step 2 — Write the query:
- Hide the Novelty: Do not reveal the hypothesis's core contribution in the query. \
  The query must be broad enough not to spoil the answer, yet specific enough to \
  naturally elicit it.
- Seek Directions, Not Direct Fixes: Frame the query around exploring problems or \
  addressing limitations (e.g., "What are promising directions for addressing X?" or \
  "What hypotheses could advance our understanding of Y?"). Do not ask for a direct solution.
- Don't look for answers: The query should be open-ended and exploratory, not asking \
  for a specific answer. Avoid phrasing that implies a known answer exists.
- Keep it Natural: Write exactly one sentence. Do not use meta-phrases like \
  "based on the paper provided."
- Strict Output: Output ONLY the query itself. No preambles, labels, chain-of-thought, \
  or commentary — just the final query sentence.
"""

QUERY_USER_TEMPLATE = (
    "## Paper Abstract\n{abstract}\n\n"
    "## Paper Summary\n{summary}\n\n"
    "---\n\n"
    "First identify the main hypothesis of this paper, then write the single-sentence "
    "query a scientist would send to a hypothesis-generation system to elicit that hypothesis."
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def build_reference_index() -> dict[str, list[str]]:
    """Return {paper_id: [referenced_paper_id, ...]} for all papers."""
    index: dict[str, list[str]] = {}
    if not REFS_CSV.exists():
        return index
    with open(REFS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = row[":START_ID(Paper)"]
            dst = row[":END_ID(Paper)"]
            index.setdefault(src, []).append(dst)
    return index


def load_all_papers() -> dict[str, dict]:
    """Return {paper_id: {title, abstract, summary}} for all papers."""
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


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def generate_query(client: OpenAI, model: str, abstract: str, summary: str) -> str:
    response = client.responses.create(
        model=model,
        instructions=QUERY_SYSTEM_PROMPT,
        input=QUERY_USER_TEMPLATE.format(
            abstract=abstract or "(not available)",
            summary=summary or "(not available)",
        ),
        reasoning={"effort": "medium"},
    )
    return response.output_text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate queries directly from target papers."
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of samples to generate."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/rl_queries.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-mini",
        help="OpenAI model to use for query generation (default: gpt-5.4-mini).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel threads for LLM calls (default: 8).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    client = OpenAI()

    print("Loading reference index …")
    ref_index = build_reference_index()

    print("Loading papers …")
    papers = load_all_papers()

    eligible = [
        pid
        for pid, p in papers.items()
        if (p["abstract"] or p["summary"]) and len(ref_index.get(pid, [])) >= 3
    ]
    print(
        f"  {len(eligible)} papers with at least an abstract or summary and >= 3 references found."
    )

    if not eligible:
        print("No eligible papers found. Exiting.")
        sys.exit(1)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "paper_id",
        "paper_title",
        "referenced_paper_ids",
        "abstract",
        "summary",
        "user_query",
    ]

    already_used: set[str] = set()
    generated = 0

    # Sample all paper IDs upfront
    rng.shuffle(eligible)
    selected = eligible[: args.n]

    write_lock = threading.Lock()

    def process(paper_id: str, index: int) -> bool:
        paper = papers[paper_id]
        print(f"\n[{index}/{args.n}] Paper: {paper['title'][:80]!r}")
        try:
            user_query = generate_query(
                client, args.model, paper["abstract"], paper["summary"]
            )
            print(f"  Query: {user_query[:120]} …")
        except Exception as exc:
            print(f"  ERROR: {exc} — skipping this paper.")
            return False

        row = {
            "paper_id": paper_id,
            "paper_title": paper["title"],
            "referenced_paper_ids": "|".join(ref_index.get(paper_id, [])),
            "abstract": paper["abstract"],
            "summary": paper["summary"],
            "user_query": user_query,
        }
        with write_lock:
            writer.writerow(row)
            f.flush()
        return True

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process, pid, i): pid
                for i, pid in enumerate(selected, 1)
            }
            for future in as_completed(futures):
                if future.result():
                    generated += 1

    print(f"\nDone. {generated} samples saved to {out_path}")


if __name__ == "__main__":
    main()
