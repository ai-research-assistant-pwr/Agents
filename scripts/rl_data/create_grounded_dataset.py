"""
Generate a grounded dataset of (user_query, hypothesis, paper_id, referenced_paper_ids)
rows for training a scientific hypothesis generation system.

Differs from create_dataset.py in query generation: the LLM is given summaries of
up to 10 referenced articles and role-plays as a scientist who knows that literature
but has not yet seen the target hypothesis. It then formulates a query to a
hypothesis-generation system — a query for which the target hypothesis is the
ideal answer.

Usage:
    python scripts/rl/create_grounded_dataset.py --n 100 --out data/rl_grounded_dataset.csv
    python scripts/rl/create_grounded_dataset.py --n 100 --out data/rl_grounded_dataset.csv --model gpt-4o-mini
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

MAX_CONTEXT_PAPERS = 15

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
- Extract 1 hypothesis from the paper.
- Extract hypothesis that is introduced by the paper, not those that are only cited from other work.
- Hypothesis must be a concise, self-contained declarative statement.
- Every newly introduced term, dataset or method that is essential to understanding the hypothesis should be \
  explained within the hypothesis. Similarly, each reference to external datasets, methods, or concepts should be explained within the hypothesis. \
  The hypothesis should be understandable to a domain expert without requiring them to read the full paper. \
  After creating hypothesis, step in to the role of a domain expert and ask yourself: "If I only had this hypothesis statement, would I be able to \
  design an experiment to test it?" If the answer is no, revise the hypothesis to include necessary explanations or clarifications.
- Avoid generating simple hypothesis that are too general or obvious. Focus on extracting the specific, testable claims that the paper is centered around.
- Do not include general background statements, conclusions without a claim, or \
  methodology descriptions.
- If the paper introduces some new methodology, that accomplishes improved results on some \
  task, thet the hypothesis should focus mostly on descrbing the methodology, not the results. \
  The hypothesis should be about the method itself, not just the fact that it achieved better performance. \
  For example, a good hypothesis would be: "We propose a novel method X that integrates techniques A and B in a unique way, allowing it to effectively leverage both structured and unstructured data for improved performance on task Y."
- Split hypothesis into multiple sentences. Most papers will have complex hypotheses and writing few sentences will make them easier to understand.
- Write only hypothesis text, don't include any preamble, commentary, or explanation. The output should be a clean, concise hypothesis statement that stands on its own. \
  For example, don't write "The hypothesis of this paper is: ..." or "The main claim of this article is that ...". Just write the hypothesis itself, without any leading phrases or framing.
"""

HYPOTHESIS_USER_TEMPLATE = (
    "## Paper Abstract\n{abstract}\n\n"
    "## Paper Summary\n{summary}\n\n"
    "---\n\n"
    "Extract the main scientific hypothesis from this paper."
)

GROUNDED_QUERY_SYSTEM_PROMPT = """
Role: You are generating training queries for a scientific hypothesis generation system.
Input: A target hypothesis and summaries of its referenced papers.
Task: Write a single-sentence query a scientist would naturally ask, for which the target hypothesis is the ideal answer.

Guidelines:

-Hide the Novelty: Identify the hypothesis's core contribution (e.g., a new method or novel connection). Do not reveal this in the query. The query must be broad enough to not spoil the answer, yet specific enough to naturally elicit it.
-Seek Directions, Not Direct Fixes: Frame the query around exploring problems or addressing limitations (e.g., "What are promising directions for addressing X?" or "What hypotheses could advance our understanding of Y?"). Do not ask for a direct solution.
-Don't look for answers: The query should be open-ended and exploratory, not asking for a specific answer or solution. Avoid phrasing that implies there is a known answer to the question (e.g. "What hypothesis could explain X?").
-Keep it Natural: Write exactly one sentence. You may reference the context papers, but never use meta-phrases like "based on the summaries provided."
-Strict Output: Output ONLY the query itself. No preambles, labels, or commentary.
"""

GROUNDED_QUERY_USER_TEMPLATE = (
    "## Summaries of Related Papers\n\n"
    "{paper_summaries}\n\n"
    "---\n\n"
    "## Target Hypothesis (for your reference only — do NOT copy or paraphrase it)\n"
    "{hypothesis}\n\n"
    "---\n\n"
    "As a scientist who has read the papers above but does not know the target "
    "hypothesis, write the query you would send to a hypothesis-generation system "
    "to elicit a hypothesis like the one above."
)


def _format_paper_summaries(ref_papers: list[dict]) -> str:
    parts = []
    for i, p in enumerate(ref_papers, 1):
        title = p["title"] or "(no title)"
        summary = p["summary"] or p["abstract"] or "(no summary available)"
        parts.append(f"[{i}] **{title}**\n{summary}")
    return "\n\n".join(parts)


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


def generate_grounded_query(
    client: OpenAI,
    model: str,
    hypothesis: str,
    ref_papers: list[dict],
) -> str:
    paper_summaries = _format_paper_summaries(ref_papers)
    response = client.responses.create(
        model=model,
        instructions=GROUNDED_QUERY_SYSTEM_PROMPT,
        input=GROUNDED_QUERY_USER_TEMPLATE.format(
            paper_summaries=paper_summaries,
            hypothesis=hypothesis,
        ),
        reasoning={"effort": "medium"},
    )
    return response.output_text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate grounded hypothesis dataset from graph data."
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of samples to generate."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/rl_grounded_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--hypothesis-model",
        type=str,
        default="gpt-5.4",
        help="OpenAI model to use for hypothesis extraction (default: gpt-5.4).",
    )
    parser.add_argument(
        "--query-model",
        type=str,
        default="gpt-5.4-mini",
        help="OpenAI model to use for query generation (default: gpt-5.4-mini).",
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

            # Gather referenced paper data (only those present in papers.csv)
            ref_papers_data = [papers[rid] for rid in ref_ids if rid in papers]
            # Cap at MAX_CONTEXT_PAPERS, shuffled so we don't always pick the same ones
            if len(ref_papers_data) > MAX_CONTEXT_PAPERS:
                ref_papers_data = rng.sample(ref_papers_data, MAX_CONTEXT_PAPERS)

            print(
                f"\n[{generated + 1}/{args.n}] Paper: {paper['title'][:80]!r} "
                f"({len(ref_ids)} refs, {len(ref_papers_data)} used as context)"
            )

            try:
                hypothesis = extract_hypothesis(
                    client, args.hypothesis_model, paper["abstract"], paper["summary"]
                )
                print(f"  Hypothesis: {hypothesis[:120]} …")

                user_query = generate_grounded_query(
                    client, args.query_model, hypothesis, ref_papers_data
                )
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
