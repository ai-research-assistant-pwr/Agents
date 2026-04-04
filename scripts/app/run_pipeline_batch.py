"""Run the hypothesis generation pipeline over a sample of queries from a CSV file.

Usage:
    python scripts/app/run_pipeline_batch.py
    python scripts/app/run_pipeline_batch.py --csv data/synthetic_prompts_gemini-3-flash-preview_simpler.csv
    python scripts/app/run_pipeline_batch.py --sample-size 20 --seed 123
    python scripts/app/run_pipeline_batch.py --sample-size 5 --save-steps --refinement-turns 1

Explorer is selected from config (explorer.type):
    - "weaviate"  WeaviateExplorer  (Qwen3-Embedding-8B + Weaviate vector DB)
    - "const"     ConstExplorer     (placeholder, for testing without a DB)

The API client randomly selects a model on each call weighted by the MODELS
table below. Add or adjust entries there to change the pool.

Results are written to a CSV file in the outputs/ directory (one row per query).

Requires GOOGLE_API_KEY to be set in the environment or in a .env file at the
project root.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

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
from app.api_client.google_client import GoogleAPIClient
from app.api_client.random_client import RandomAPIClient
from app.config import load_config
from app.explorer.const_explorer import ConstExplorer
from app.explorer.weaviate_explorer import WeaviateExplorer
from app.generator.api_llm_generator import APILLMGenerator
from app.retriever.api_llm_retriever import APILLMRetriever

CONFIG_PATH = "config/app/config.yaml"
DEFAULT_CSV = "data/synthetic_prompts_gemini-3-flash-preview_simpler.csv"
QUERY_COLUMN = "generated_prompt"
DEFAULT_SAMPLE_SIZE = 2

# Provider classes available for random routing.
PROVIDERS: dict[str, type] = {
    "google": GoogleAPIClient,
}

# Model pool used by RandomAPIClient.
# weight controls relative selection probability (higher = more likely).
MODELS: dict[str, dict] = {
    "gemini-3-flash-preview": {"provider": "google", "weight": 7},
    "gemini-3.1-flash-lite-preview": {"provider": "google", "weight": 30},
}


def build_explorer(config: dict):
    """Instantiate the explorer specified by config[explorer][type]."""
    explorer_type = config.get("explorer", {}).get("type", "const")
    if explorer_type == "weaviate":
        return WeaviateExplorer(config)
    if explorer_type == "const":
        const_text = config.get("explorer", {}).get("const_text")
        return ConstExplorer(text=const_text)
    raise ValueError(f"Unknown explorer type: {explorer_type!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pipeline over a sample of queries from a CSV file."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help=f"Path to the CSV file with queries (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--query-column",
        type=str,
        default=QUERY_COLUMN,
        help=f"Column name containing the query text (default: {QUERY_COLUMN}).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of rows to sample from the CSV (default: {DEFAULT_SAMPLE_SIZE}). "
             "Pass -1 to run all rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--save-steps",
        action="store_true",
        help="Save intermediate pipeline step outputs to disk for each query.",
    )
    parser.add_argument(
        "--refinement-turns",
        type=int,
        default=0,
        help="Number of retriever<->generator refinement rounds (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the results CSV. Defaults to outputs/batch_<timestamp>.csv.",
    )
    return parser.parse_args()


def load_queries(csv_path: str, query_column: str, sample_size: int, seed: int) -> pd.DataFrame:
    """Load and optionally sample rows from the CSV file.

    Returns a DataFrame with at least the query column. All original columns
    are kept so that metadata can be written to the output file.
    """
    abs_path = csv_path if os.path.isabs(csv_path) else os.path.join(PROJECT_ROOT, csv_path)
    df = pd.read_csv(abs_path)

    if query_column not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Column '{query_column}' not found in {csv_path}.\n"
            f"Available columns: {available}"
        )

    if sample_size == -1 or sample_size >= len(df):
        return df.reset_index(drop=True)

    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: GOOGLE_API_KEY is not set.")
        print(
            "Set it in your environment or add it to a .env file at the project root."
        )
        sys.exit(1)

    config = load_config(CONFIG_PATH)
    explorer_type = config.get("explorer", {}).get("type", "const")

    # Load queries
    df = load_queries(args.csv, args.query_column, args.sample_size, args.seed)
    n_queries = len(df)

    model_pool = ", ".join(f"{name}(w={cfg['weight']})" for name, cfg in MODELS.items())
    print(f"Explorer        : {explorer_type}")
    print(f"Model pool      : {model_pool}")
    print(f"Refinements     : {args.refinement_turns}")
    print(f"Save steps      : {args.save_steps}")
    print(f"CSV file        : {args.csv}")
    print(f"Query column    : {args.query_column}")
    print(f"Sample size     : {n_queries} queries (seed={args.seed})")
    print()

    # Wire up the pipeline components (shared across all queries)
    api_client = RandomAPIClient(providers=PROVIDERS, models=MODELS)
    explorer = build_explorer(config)
    retriever = APILLMRetriever(api_client=api_client)
    generator = APILLMGenerator(api_client=api_client)

    app = App(
        explorer=explorer,
        retriever=retriever,
        generator=generator,
        config_path=CONFIG_PATH,
    )
    app.config.setdefault("pipeline", {})
    app.config["pipeline"]["save_steps"] = args.save_steps
    app.config["pipeline"]["refinement_turns"] = args.refinement_turns

    # Prepare output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(PROJECT_ROOT) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"batch_{timestamp}.csv"

    # Collect results
    results: list[dict] = []
    errors: list[dict] = []

    for idx, row in df.iterrows():
        query = row[args.query_column]
        print(f"[{idx + 1}/{n_queries}] Running pipeline...")
        print(f"  Query: {str(query)[:120]}{'...' if len(str(query)) > 120 else ''}")

        t0 = time.time()
        try:
            result = app.run(query)
            elapsed = time.time() - t0

            record = row.to_dict()
            record["hypotheses"] = " | ".join(result.hypotheses)
            record["model_used"] = result.metadata.get("model", "unknown")
            record["elapsed_seconds"] = round(elapsed, 2)
            record["status"] = "ok"
            results.append(record)

            print(f"  Done in {elapsed:.1f}s  |  model={record['model_used']}  |  hypotheses={len(result.hypotheses)}")
            for i, h in enumerate(result.hypotheses, start=1):
                print(f"    {i}. {h[:100]}{'...' if len(h) > 100 else ''}")

        except Exception as exc:
            elapsed = time.time() - t0
            record = row.to_dict()
            record["hypotheses"] = ""
            record["model_used"] = ""
            record["elapsed_seconds"] = round(elapsed, 2)
            record["status"] = f"error: {exc}"
            results.append(record)
            errors.append({"index": idx, "query": query, "error": str(exc)})
            print(f"  ERROR after {elapsed:.1f}s: {exc}")

        print()

    # Write results CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

    print("=" * 60)
    print(f"Finished {n_queries} queries  |  errors: {len(errors)}")
    print(f"Results saved to: {output_path}")
    if errors:
        print(f"\nFailed queries ({len(errors)}):")
        for e in errors:
            print(f"  [{e['index']}] {str(e['query'])[:80]} -> {e['error']}")


if __name__ == "__main__":
    main()
