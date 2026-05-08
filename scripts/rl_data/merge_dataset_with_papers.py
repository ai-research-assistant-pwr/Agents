"""
Transform a grounded dataset CSV into the v2 format.

Input columns:  paper_id, paper_title, referenced_paper_ids, hypothesis, user_query
Output columns: user_query, hypothesis, metadata

`metadata` is a JSON object:
  {
    "papers": [
      {"id": "<paper_id>", "title": "<title>", "summary": "<summary>"},
      ...
    ]
  }

Each entry in `papers` corresponds to one of the referenced papers (pipe-separated
in `referenced_paper_ids`) that is present in the papers CSV.

Usage:
    python scripts/rl/transform_to_v2.py \
        --input  data/rl_grounded_dataset.csv \
        --papers data/graph/11_neo4j_papers.csv \
        --out    data/rl_grounded_dataset_v2.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

csv.field_size_limit(10_000_000)


def load_papers(papers_csv: Path) -> dict[str, dict]:
    """Return {paper_id: {title, summary}} for all papers."""
    papers: dict[str, dict] = {}
    with open(papers_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["paperId:ID(Paper)"]
            papers[pid] = {
                "title": row.get("title", "") or "",
                "summary": row.get("summary", "") or row.get("abstract", "") or "",
            }
    return papers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform grounded dataset CSV to v2 format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/rl_grounded_dataset.csv",
        help="Input grounded dataset CSV (default: data/rl_grounded_dataset.csv).",
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=ROOT / "data/graph/11_neo4j_papers.csv",
        help="Papers CSV with id/title/summary (default: data/graph/11_neo4j_papers.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/rl_grounded_dataset_merged.csv",
        help="Output CSV path (default: data/rl_grounded_dataset_merged.csv).",
    )
    args = parser.parse_args()

    print(f"Loading papers from {args.papers} …")
    papers = load_papers(args.papers)
    print(f"  {len(papers)} papers loaded.")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0

    with (
        open(args.input, newline="", encoding="utf-8") as fin,
        open(args.out, "w", newline="", encoding="utf-8") as fout,
    ):
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(
            fout, fieldnames=["user_query", "hypothesis", "metadata"]
        )
        writer.writeheader()

        for row in reader:
            ref_ids = [rid for rid in row["referenced_paper_ids"].split("|") if rid]
            ref_papers = [
                {
                    "id": rid,
                    "title": papers[rid]["title"],
                    "summary": papers[rid]["summary"],
                }
                for rid in ref_ids
                if rid in papers
            ]

            if not ref_papers:
                print(
                    f"  Warning: no referenced papers found for {row['paper_id']!r} — skipping."
                )
                skipped += 1
                continue

            metadata = json.dumps({"papers": ref_papers}, ensure_ascii=False)
            writer.writerow(
                {
                    "user_query": row["user_query"],
                    "hypothesis": row["hypothesis"],
                    "metadata": metadata,
                }
            )
            written += 1

    print(f"\nDone. {written} rows written to {args.out} ({skipped} skipped).")


if __name__ == "__main__":
    main()
