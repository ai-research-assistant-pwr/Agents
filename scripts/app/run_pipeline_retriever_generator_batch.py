"""Run the simplified hypothesis generation pipeline over a sample of queries
from a CSV file using a single RetrieverGenerator agent (no separate
retriever + generator).

Usage:
    python scripts/app/run_pipeline_retriever_generator_batch.py
    python scripts/app/run_pipeline_retriever_generator_batch.py --csv data/synthetic_prompts.csv
    python scripts/app/run_pipeline_retriever_generator_batch.py --sample-size 20 --seed 123
    python scripts/app/run_pipeline_retriever_generator_batch.py --sample-size 5 --save-steps

Results are written to a CSV file in the outputs/ directory (one row per query).
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

env_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(env_path)

from app.api_client.base import BaseAPIClient
from app.api_client.cerebras_client import CerebrasAPIClient
from app.api_client.google_client import GoogleAPIClient
from app.api_client.openai_client import OpenAIAPIClient
from app.config import load_config
from app.explorer.agentic_explorer import AgenticExplorer
from app.explorer.const_explorer import ConstExplorer
from app.explorer.neo4j_bfs_explorer import Neo4jBFSExplorer
from app.explorer.neo4j_pagerank_explorer import Neo4jPageRankExplorer
from app.explorer.neo4j_random_walk_explorer import Neo4jRandomWalkExplorer
from app.explorer.none_explorer import NoneExplorer
from app.explorer.weaviate_explorer import WeaviateExplorer
from app.explorer.weaviate_search_explorer import WeaviateSearchExplorer
from app.retriever_generator.api_llm_retriever_generator import APILLMRetrieverGenerator

CONFIG_PATH = "config/app/config.yaml"
DEFAULT_CSV = "data/prompts.csv"
QUERY_COLUMN = "generated_prompt"
DEFAULT_SAMPLE_SIZE = 8

PROVIDERS: dict[str, type[BaseAPIClient]] = {
    "google": GoogleAPIClient,
    "openai": OpenAIAPIClient,
    "cerebras": CerebrasAPIClient,
}

MODELS: list[dict] = [
    {"name": "gemini-3.5-flash", "provider": "google", "weight": 1},
]


def select_model() -> dict:
    weights = [m["weight"] for m in MODELS]
    return random.choices(MODELS, weights=weights, k=1)[0]


def build_api_client(model_cfg: dict) -> BaseAPIClient:
    client_cls = PROVIDERS[model_cfg["provider"]]
    return client_cls(model=model_cfg["name"])


def build_search(config: dict):
    search_type = config.get("search", {}).get("type")
    if search_type == "weaviate":
        return WeaviateSearchExplorer(config)
    raise ValueError(f"Unknown search type: {search_type!r}")


def build_explorer(config: dict):
    explorer_type = config.get("explorer", {}).get("type", "const")
    if explorer_type == "weaviate":
        return WeaviateExplorer(config)
    if explorer_type == "neo4j_bfs":
        return Neo4jBFSExplorer(config)
    if explorer_type == "neo4j_random_walk":
        return Neo4jRandomWalkExplorer(config)
    if explorer_type == "neo4j_pagerank":
        return Neo4jPageRankExplorer(config)
    if explorer_type == "agentic":
        return AgenticExplorer(config)
    if explorer_type == "const":
        const_text = config.get("explorer", {}).get("const_text")
        return ConstExplorer(text=const_text)
    if explorer_type == "none":
        return NoneExplorer(config)
    raise ValueError(f"Unknown explorer type: {explorer_type!r}")


def _save_step(save_dir: Path, name: str, data: dict) -> None:
    path = save_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the simplified pipeline over a sample of queries "
        "(single RetrieverGenerator agent)."
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
        help=f"Number of rows to sample (default: {DEFAULT_SAMPLE_SIZE}). "
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
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write all results into. "
        "CSV will be saved as results.csv and per-query step folders "
        "will be placed here. Overrides --output.",
    )
    return parser.parse_args()


def load_queries(
    csv_path: str, query_column: str, sample_size: int, seed: int
) -> pd.DataFrame:
    abs_path = (
        csv_path if os.path.isabs(csv_path) else os.path.join(PROJECT_ROOT, csv_path)
    )
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

    config = load_config(CONFIG_PATH)
    search_type = config.get("search", {}).get("type", "weaviate")
    explorer_type = config.get("explorer", {}).get("type", "const")

    df = load_queries(args.csv, args.query_column, args.sample_size, args.seed)
    n_queries = len(df)

    model_pool = ", ".join(
        f"{m['name']}(provider={m['provider']}, w={m['weight']})" for m in MODELS
    )
    print(f"Search          : {search_type}")
    print(f"Explorer        : {explorer_type}")
    print(f"Model pool      : {model_pool}")
    print(f"Save steps      : {args.save_steps}")
    print(f"CSV file        : {args.csv}")
    print(f"Query column    : {args.query_column}")
    print(f"Sample size     : {n_queries} queries (seed={args.seed})")
    print()

    search_explorer = build_search(config)
    explorer = build_explorer(config)

    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(PROJECT_ROOT) / "outputs" / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "results.csv"

    results: list[dict] = []
    errors: list[dict] = []

    for idx, row in df.iterrows():
        query = row[args.query_column]

        model_cfg = select_model()
        model_name = model_cfg["name"]
        provider = model_cfg["provider"]

        print(f"[{idx + 1}/{n_queries}] Running pipeline...")
        print(f"  Query : {str(query)[:120]}{'...' if len(str(query)) > 120 else ''}")
        print(f"  Model : {model_name}  (provider={provider})")

        t0 = time.time()
        try:
            save_dir = None
            if args.save_steps:
                save_dir = run_dir / f"query_{idx}"
                save_dir.mkdir(parents=True, exist_ok=True)

            api_client = build_api_client(model_cfg)
            retriever_generator = APILLMRetrieverGenerator(
                api_client=api_client,
            )

            paper_ids = search_explorer.search(query)
            if save_dir:
                _save_step(save_dir, "01_search", {"paper_ids": paper_ids})

            explorer_output = explorer.explore(query, paper_ids)
            if save_dir:
                _save_step(save_dir, "02_explorer", asdict(explorer_output))

            result = retriever_generator.generate(query, explorer_output)
            if save_dir:
                _save_step(save_dir, "05_generator", asdict(result))

            elapsed = time.time() - t0

            record = row.to_dict()
            record["hypotheses"] = " | ".join(result.hypotheses)
            record["model_used"] = result.metadata.get("model", "unknown")
            record["elapsed_seconds"] = round(elapsed, 2)
            record["status"] = "ok"
            results.append(record)

            print(
                f"  Done in {elapsed:.1f}s  |  model={record['model_used']}  |  "
                f"hypotheses={len(result.hypotheses)}"
            )
            for i, h in enumerate(result.hypotheses, start=1):
                print(f"    {i}. {h[:100]}{'...' if len(h) > 100 else ''}")

        except Exception as exc:
            elapsed = time.time() - t0
            record = row.to_dict()
            record["hypotheses"] = ""
            record["model_used"] = model_name
            record["elapsed_seconds"] = round(elapsed, 2)
            record["status"] = f"error: {exc}"
            results.append(record)
            errors.append({"index": idx, "query": query, "error": str(exc)})
            print(f"  ERROR after {elapsed:.1f}s: {exc}")

        print()

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
