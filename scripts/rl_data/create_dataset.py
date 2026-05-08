"""
Generate a dataset of (user_query, hypothesis, paper_id, referenced_paper_ids) rows
for training a scientific hypothesis generation system.

For each sample:
  1. Randomly select a paper that references at least 3 other papers.
  2. Call an LLM to extract the main scientific hypothesis from the paper's
     summary + abstract.
  3. Call an LLM to generate a user query for which the extracted hypothesis
     would be the ideal answer.

Usage:
    python scripts/rl/create_dataset.py --n 100 --out data/rl_dataset.csv
    python scripts/rl/create_dataset.py --n 100 --out data/rl_dataset.csv --model gpt-4o-mini
"""

import argparse
import csv
import random
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]

csv.field_size_limit(10_000_000)

PAPERS_CSV = ROOT / "data/graph/11_neo4j_papers.csv"
REFS_CSV = ROOT / "data/graph/11_neo4j_references.csv"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

HYPOTHESIS_SYSTEM_PROMPT = """
You are a scientific paper analyst. Your task is to extract the main scientific \
hypothesis from a research paper.

A scientific hypothesis is a testable claim, prediction, or explanation that the \
paper proposes to investigate or validate. Focus on what the authors argue, claim \
to demonstrate, or set out to test — not background facts or general statements.

Instructions:
- Extract between 1 hypothesis from the paper.
- Extract hypothesis that is introduced by the paper, not those that are only cited from other work.
- Hypothesis must be a concise, self-contained declarative statement.
- Every newly introduced term, dataset or method that is essential to understanding the hypothesis should be \
  explained within the hypothesis. Similarly, each reference to external datasets, methods, or concepts should be explained within the hypothesis. \
  The hypothesis should be understandable to a domain expert without requiring them to read the full paper. \
  After creating hypothesis, step in to the role of a domain expert and ask yourself: "If I only had this hypothesis statement, would I be able to \
  design an experiment to test it?" If the answer is no, revise the hypothesis to include necessary explanations or clarifications.
- Avoid generating simple hypothesis that are too general or obvious. Focus on extracting the specific, testable claims that the paper is centered around.
- Hypothesis can contain even few sentences (up to 3) if necessary to fully capture the claim and its context.
- Do not include general background statements, conclusions without a claim, or \
  methodology descriptions.
- Write only hypothesis text, don't include any preamble, commentary, or explanation. The output should be a clean, concise hypothesis statement that stands on its own.
"""

HYPOTHESIS_USER_TEMPLATE = (
    "## Paper Abstract\n{abstract}\n\n"
    "## Paper Summary\n{summary}\n\n"
    "---\n\n"
    "Extract the main scientific hypothesis from this paper."
)

QUERY_SYSTEM_PROMPT = (
    "You are an expert at formulating research questions. "
    "Given a scientific hypothesis, generate a vague, high-level user query "
    "that a researcher might type into a hypothesis-generation system.\n\n"
    "The query should:\n"
    "- Be phrased as a broad research question or area of interest, NOT as a "
    "specific question about the hypothesis itself.\n"
    "- Mention only the general research domain or problem space — do NOT "
    "include specific mechanisms, methods, variables, or conclusions from "
    "the hypothesis.\n"
    "- Be open-ended enough that many different hypotheses could plausibly "
    "answer it, while still being relevant to this one.\n"
    "- Be self-contained and understandable without prior context.\n"
    "- Be 1-2 sentences at most.\n\n"
    "Output ONLY the query. Do not add any preamble or commentary."
)

QUERY_USER_TEMPLATE = (
    "## Scientific Hypothesis\n{hypothesis}\n\n"
    "---\n\n"
    "Generate a vague, high-level user query for which the hypothesis above "
    "would be a fitting (but not obvious) answer."
)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def build_reference_index() -> dict[str, list[str]]:
    """Return {paper_id: [referenced_paper_id, ...]} for all papers."""
    index: dict[str, list[str]] = {}
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
# LLM calls
# ---------------------------------------------------------------------------


def extract_hypothesis(client: OpenAI, model: str, abstract: str, summary: str) -> str:
    response = client.responses.create(
        model=model,
        instructions=HYPOTHESIS_SYSTEM_PROMPT,
        input=HYPOTHESIS_USER_TEMPLATE.format(
            abstract=abstract or "(not available)",
            summary=summary or "(not available)",
        ),
        reasoning={"effort": "medium"},
    )
    return response.output_text.strip()


def generate_query(client: OpenAI, model: str, hypothesis: str) -> str:
    response = client.responses.create(
        model=model,
        instructions=QUERY_SYSTEM_PROMPT,
        input=QUERY_USER_TEMPLATE.format(hypothesis=hypothesis),
        reasoning={"effort": "medium"},
    )
    return response.output_text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hypothesis dataset from graph data."
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of samples to generate."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/rl_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-mini",
        help="OpenAI model to use (default: gpt-5.4-mini).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--min-refs",
        type=int,
        default=3,
        help="Minimum number of references a paper must have (default: 3).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    client = OpenAI()

    print("Loading reference index …")
    ref_index = build_reference_index()

    print("Loading papers …")
    papers = load_all_papers()

    # Filter to papers with at least --min-refs references that are also in papers.csv
    eligible = [
        pid
        for pid, refs in ref_index.items()
        if len(refs) >= args.min_refs and pid in papers
    ]
    print(f"  {len(eligible)} papers with >= {args.min_refs} references found.")

    if len(eligible) == 0:
        print("No eligible papers found. Exiting.")
        sys.exit(1)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "paper_id",
        "paper_title",
        "referenced_paper_ids",
        "hypothesis",
        "user_query",
    ]

    already_used: set[str] = set()
    generated = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while generated < args.n:
            remaining_eligible = [p for p in eligible if p not in already_used]
            if not remaining_eligible:
                print(
                    "Warning: ran out of unique eligible papers before reaching N. "
                    f"Generated {generated} samples."
                )
                break

            paper_id = rng.choice(remaining_eligible)
            already_used.add(paper_id)
            paper = papers[paper_id]
            ref_ids = ref_index[paper_id]

            print(
                f"\n[{generated + 1}/{args.n}] Paper: {paper['title'][:80]!r} "
                f"({len(ref_ids)} refs)"
            )

            try:
                hypothesis = extract_hypothesis(
                    client, args.model, paper["abstract"], paper["summary"]
                )
                print(f"  Hypothesis: {hypothesis[:120]} …")

                user_query = generate_query(client, args.model, hypothesis)
                print(f"  Query     : {user_query[:120]} …")
            except Exception as exc:
                print(f"  ERROR: {exc} — skipping this paper.")
                continue

            writer.writerow(
                {
                    "paper_id": paper_id,
                    "paper_title": paper["title"],
                    "referenced_paper_ids": "|".join(ref_ids),
                    "hypothesis": hypothesis,
                    "user_query": user_query,
                }
            )
            f.flush()
            generated += 1

    print(f"\nDone. {generated} samples saved to {out_path}")


if __name__ == "__main__":
    main()
