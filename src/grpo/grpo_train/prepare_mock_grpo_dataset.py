"""
Prepare GRPO training dataset from:
  - rl_grounded_dataset.csv  (queries, paper references, ground-truth hypotheses)
  - 11_neo4j_papers.csv      (paper summaries indexed by paperId)

Output: data/datasets/grpo_exp_dataset/grpo_data.json
Each record:
  {
    "id":       <paper_id from rl_grounded_dataset>,
    "query":    <user_query>,
    "label":    <ground-truth hypothesis>,
    "metadata": {"papers": [{"id", "title", "summary"}, ...]}
  }
"""

import os
import sys
import csv
import json
import random

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRPO_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.dirname(GRPO_DIR)
AGENTS_DIR = os.path.dirname(SRC_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(AGENTS_DIR, "data")
QUERIES_CSV = os.path.join(DATA_DIR, "rl_grounded_dataset.csv")
PAPERS_CSV = os.path.join(DATA_DIR, "11_neo4j_papers.csv")

OUTPUT_DIR = os.path.join(AGENTS_DIR, "data", "datasets", "grpo_exp_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "grpo_data.json")

MAX_PAPERS_PER_QUERY = 8
MAX_RECORDS = None  # set to an int to cap dataset size (None = all)
RANDOM_SEED = 42

csv.field_size_limit(10**7)


def load_papers(path: str) -> dict:
    """Return {paperId: {"id", "title", "summary"}} from 11_neo4j_papers.csv."""
    papers = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("paperId:ID(Paper)", "").strip()
            if not pid:
                continue
            summary = row.get("summary", "").strip()
            if not summary:
                summary = row.get("abstract", "").strip()
            papers[pid] = {
                "id": pid,
                "title": row.get("title", "").strip(),
                "summary": summary,
            }
    return papers


def load_queries(path: str) -> list:
    """Return list of dicts from rl_grounded_dataset.csv."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def main():
    print("=== Preparing GRPO dataset from rl_grounded_dataset + neo4j_papers ===")

    for p in [QUERIES_CSV, PAPERS_CSV]:
        if not os.path.exists(p):
            print(f"ERROR: file not found: {p}")
            sys.exit(1)

    print(f"Loading papers from {PAPERS_CSV} …")
    papers_db = load_papers(PAPERS_CSV)
    print(f"  Loaded {len(papers_db)} papers.")

    print(f"Loading queries from {QUERIES_CSV} …")
    queries = load_queries(QUERIES_CSV)
    print(f"  Loaded {len(queries)} query records.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = []
    missing_paper_counts = []

    for row in queries:
        paper_id = row.get("paper_id", "").strip()
        user_query = row.get("user_query", "").strip()
        hypothesis = row.get("hypothesis", "").strip()
        ref_ids_raw = row.get("referenced_paper_ids", "")

        if not user_query or not hypothesis:
            continue

        ref_ids = [pid.strip() for pid in ref_ids_raw.split("|") if pid.strip()]
        context_papers = []
        missing = 0
        for pid in ref_ids[:MAX_PAPERS_PER_QUERY]:
            if pid in papers_db:
                context_papers.append(papers_db[pid])
            else:
                missing += 1

        missing_paper_counts.append(missing)

        records.append(
            {
                "id": paper_id,
                "query": user_query,
                "label": hypothesis,
                "metadata": {"papers": context_papers},
            }
        )

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    if MAX_RECORDS is not None:
        records = records[:MAX_RECORDS]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    avg_missing = sum(missing_paper_counts) / max(len(missing_paper_counts), 1)
    print(f"  Avg missing papers per query: {avg_missing:.1f}")
    print(f"Saved {len(records)} records → {OUTPUT_FILE}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
