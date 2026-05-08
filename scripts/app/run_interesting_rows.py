"""Run the hypothesis generation pipeline over all rows in data/interesting_rows.csv.

For each query the pipeline is executed twice, each time using a *different*
randomly-chosen model from the MODELS pool.  The paper summaries stored in the
``metadata`` column are formatted and passed directly to the retriever as
context, bypassing the Weaviate explorer entirely.

Output: outputs/interesting_rows_<timestamp>.csv
One row per generated hypothesis, with columns:
    query, hypothesis, hypothesis_num, model, num_papers, run_index

Usage:
    python scripts/app/run_interesting_rows.py
    python scripts/app/run_interesting_rows.py --output path/to/out.csv
    python scripts/app/run_interesting_rows.py --runs-per-query 3

Requires the API key for each provider listed in MODELS to be set in the
environment or in a .env file at the project root (GOOGLE_API_KEY,
OPENAI_API_KEY, or CEREBRAS_API_KEY).
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

import pandas as pd
from dotenv import load_dotenv

# Resolve project root and add src/ to path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

env_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(env_path)

from app import App
from app.api_client.base import BaseAPIClient
from app.api_client.cerebras_client import CerebrasAPIClient
from app.api_client.google_client import GoogleAPIClient
from app.api_client.openai_client import OpenAIAPIClient
from app.explorer.const_explorer import ConstExplorer
from app.generator.api_llm_generator import APILLMGenerator
from app.retriever.api_llm_retriever import APILLMRetriever

CONFIG_PATH = "config/app/config.yaml"
INPUT_CSV = "data/interesting_rows.csv"
RUNS_PER_QUERY = 2  # default: run the pipeline twice per query

# Maps provider name -> client class.
PROVIDERS: dict[str, type[BaseAPIClient]] = {
    "google": GoogleAPIClient,
    "openai": OpenAIAPIClient,
    "cerebras": CerebrasAPIClient,
}

# Model pool.  Adjust names / weights as needed.
MODELS: list[dict] = [
    {"name": "gemini-3-flash-preview", "provider": "google", "weight": 1},
    {"name": "gpt-5.4", "provider": "openai", "weight": 1},
    {"name": "gpt-5.4-nano", "provider": "openai", "weight": 1},
    {"name": "llama3.1-8b", "provider": "cerebras", "weight": 1},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def select_models_for_runs(n: int) -> list[dict]:
    """Pick *n* models from MODELS (without immediate repetition when possible).

    For n == 2 this guarantees two *different* models as long as the pool has
    at least 2 distinct entries.  For larger n it simply shuffles and cycles.
    """
    if len(MODELS) < 2 or n == 1:
        weights = [m["weight"] for m in MODELS]
        return random.choices(MODELS, weights=weights, k=n)

    # Build a weighted list and keep drawing without repeating the last pick.
    chosen: list[dict] = []
    last: dict | None = None
    for _ in range(n):
        candidates = [m for m in MODELS if m is not last] or MODELS
        weights = [m["weight"] for m in candidates]
        pick = random.choices(candidates, weights=weights, k=1)[0]
        chosen.append(pick)
        last = pick
    return chosen


def build_api_client(model_cfg: dict) -> BaseAPIClient:
    client_cls = PROVIDERS[model_cfg["provider"]]
    return client_cls(model=model_cfg["name"])


def format_papers_as_context(papers: list[dict]) -> str:
    """Format a list of paper dicts (id, title, summary) into a context string.

    The format mirrors what WeaviateExplorer produces so the retriever prompt
    works without modification.
    """
    sections: list[str] = []
    for paper in papers:
        header = f"[paper] {paper.get('title', 'Untitled')} (id={paper.get('id', 'unknown')})"
        summary = paper.get("summary", "No summary available.")
        sections.append(f"{header}\n{summary}")
    return "\n\n---\n\n".join(sections) if sections else "No papers provided."


def parse_metadata_papers(metadata_str: str) -> list[dict]:
    """Parse the JSON metadata column and return the list of paper dicts."""
    try:
        meta = json.loads(metadata_str)
        return meta.get("papers", [])
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pipeline over interesting_rows.csv with provided paper context."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_CSV,
        help=f"Path to the input CSV (default: {INPUT_CSV}).",
    )
    parser.add_argument(
        "--runs-per-query",
        type=int,
        default=RUNS_PER_QUERY,
        help=f"How many pipeline runs per query, each with a different model (default: {RUNS_PER_QUERY}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output CSV. Defaults to outputs/interesting_rows_<timestamp>.csv.",
    )
    parser.add_argument(
        "--refinement-turns",
        type=int,
        default=0,
        help="Number of retriever<->generator refinement rounds (default: 0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Maximum number of parallel threads (default: 8).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-task worker
# ---------------------------------------------------------------------------


def run_single(
    q_idx: int,
    total_queries: int,
    query: str,
    context: str,
    num_papers: int,
    run_idx: int,
    runs_per_query: int,
    model_cfg: dict,
    refinement_turns: int,
    print_lock: Lock,
) -> tuple[list[dict], dict | None]:
    """Execute one pipeline run and return (hypothesis_rows, error_or_None)."""
    model_name = model_cfg["name"]
    provider = model_cfg["provider"]

    with print_lock:
        print(
            f"[q{q_idx + 1}/{total_queries} run{run_idx}/{runs_per_query}]"
            f"  model={model_name} ({provider})"
            f"  query={query[:80]}{'...' if len(query) > 80 else ''}"
        )

    t0 = time.time()
    try:
        api_client = build_api_client(model_cfg)
        retriever = APILLMRetriever(api_client=api_client)
        generator = APILLMGenerator(api_client=api_client)
        explorer = ConstExplorer()

        app = App(
            explorer=explorer,
            retriever=retriever,
            generator=generator,
            config_path=CONFIG_PATH,
        )
        app.config.setdefault("pipeline", {})
        app.config["pipeline"]["save_steps"] = False
        app.config["pipeline"]["refinement_turns"] = refinement_turns

        result = app.run(query, context=context)
        elapsed = time.time() - t0

        actual_model = result.metadata.get("model", model_name)
        with print_lock:
            print(
                f"  -> done [q{q_idx + 1} run{run_idx}]"
                f"  {elapsed:.1f}s  model={actual_model}"
                f"  hypotheses={len(result.hypotheses)}"
            )

        rows = [
            {
                "query": query,
                "hypothesis": hypothesis,
                "hypothesis_num": h_num,
                "model": actual_model,
                "num_papers": num_papers,
                "run_index": run_idx,
            }
            for h_num, hypothesis in enumerate(result.hypotheses, start=1)
        ]
        return rows, None

    except Exception as exc:
        elapsed = time.time() - t0
        with print_lock:
            print(f"  -> ERROR [q{q_idx + 1} run{run_idx}]  {elapsed:.1f}s  {exc}")
        return [], {
            "query_index": q_idx,
            "run_index": run_idx,
            "model": model_name,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load input data
    abs_input = (
        args.input
        if os.path.isabs(args.input)
        else os.path.join(PROJECT_ROOT, args.input)
    )
    df = pd.read_csv(abs_input)
    required_cols = {"user_query", "metadata"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # Prepare output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(PROJECT_ROOT) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"interesting_rows_{timestamp}.csv"

    model_pool_str = ", ".join(f"{m['name']}({m['provider']})" for m in MODELS)
    total_queries = len(df)
    print(f"Input CSV       : {abs_input}")
    print(f"Rows            : {total_queries}")
    print(f"Runs per query  : {args.runs_per_query}")
    print(f"Workers         : {args.workers}")
    print(f"Refinements     : {args.refinement_turns}")
    print(f"Model pool      : {model_pool_str}")
    print(f"Output          : {output_path}")
    print()

    # Build all (query, run) tasks upfront so we can assign models once
    # and submit everything to the thread pool.
    tasks = []
    for q_idx, row in df.iterrows():
        query = str(row["user_query"])
        papers = parse_metadata_papers(row["metadata"])
        context = format_papers_as_context(papers)
        num_papers = len(papers)
        run_models = select_models_for_runs(args.runs_per_query)
        for run_idx, model_cfg in enumerate(run_models, start=1):
            tasks.append(
                {
                    "q_idx": q_idx,
                    "query": query,
                    "context": context,
                    "num_papers": num_papers,
                    "run_idx": run_idx,
                    "model_cfg": model_cfg,
                }
            )

    print(f"Total tasks     : {len(tasks)}  (submitting with {args.workers} workers)")
    print()

    print_lock = Lock()
    rows_out: list[dict] = []
    errors: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_single,
                t["q_idx"],
                total_queries,
                t["query"],
                t["context"],
                t["num_papers"],
                t["run_idx"],
                args.runs_per_query,
                t["model_cfg"],
                args.refinement_turns,
                print_lock,
            ): t
            for t in tasks
        }

        for future in as_completed(futures):
            hyp_rows, error = future.result()
            rows_out.extend(hyp_rows)
            if error:
                errors.append(error)

    # Sort for deterministic ordering: by original query index, then run, then hypothesis
    q_order = {str(row["user_query"]): i for i, row in df.iterrows()}
    rows_out.sort(
        key=lambda r: (q_order.get(r["query"], 0), r["run_index"], r["hypothesis_num"])
    )

    # Write output CSV
    out_df = pd.DataFrame(
        rows_out,
        columns=[
            "query",
            "hypothesis",
            "hypothesis_num",
            "model",
            "num_papers",
            "run_index",
        ],
    )
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

    print()
    print("=" * 60)
    print(
        f"Finished {total_queries} queries ({len(tasks)} runs)  |  errors: {len(errors)}"
    )
    print(f"Total hypotheses written : {len(rows_out)}")
    print(f"Results saved to         : {output_path}")

    if errors:
        print(f"\nFailed runs ({len(errors)}):")
        for e in errors:
            print(
                f"  [q={e['query_index']} run={e['run_index']} model={e['model']}]"
                f" -> {e['error']}"
            )


if __name__ == "__main__":
    main()
