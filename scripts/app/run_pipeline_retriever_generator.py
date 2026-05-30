"""Run the simplified hypothesis generation pipeline for a single query
using a single RetrieverGenerator agent (no separate retriever + generator).

Usage:
    python scripts/app/run_pipeline_retriever_generator.py --query "Your research question here"
    python scripts/app/run_pipeline_retriever_generator.py  # uses default query

Search is selected from config (search.type):
    - "weaviate"  WeaviateSearchExplorer  (Qwen3 embedding + reranking)

Explorer is selected from config (explorer.type):
    - "weaviate"        WeaviateExplorer
    - "neo4j_bfs"       Neo4jBFSExplorer
    - "neo4j_random_walk"  Neo4jRandomWalkExplorer
    - "neo4j_pagerank"  Neo4jPageRankExplorer
    - "agentic"         AgenticExplorer
    - "const"           ConstExplorer
    - "none"            NoneExplorer
"""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

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
DEFAULT_QUERY = "I am interested in Mixtures of Experts (MoE) models for efficient inference. What are some recent research papers on this topic, and what hypotheses can we generate about future directions in this area?"

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
        description="Run the simplified hypothesis generation pipeline "
        "(single RetrieverGenerator agent)."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Research question to generate hypotheses for.",
    )
    parser.add_argument(
        "--save-steps",
        action="store_true",
        help="Save intermediate pipeline step outputs to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(CONFIG_PATH)
    search_type = config.get("search", {}).get("type", "weaviate")
    explorer_type = config.get("explorer", {}).get("type", "const")

    model_cfg = select_model()
    model_name = model_cfg["name"]
    provider = model_cfg["provider"]

    print(f"Search        : {search_type}")
    print(f"Explorer      : {explorer_type}")
    print(f"Model         : {model_name}  (provider={provider})")
    print(f"Save steps    : {args.save_steps}")
    print(f"Query         : {args.query}")
    print()

    api_client = build_api_client(model_cfg)
    search_explorer = build_search(config)
    explorer = build_explorer(config)
    retriever_generator = APILLMRetrieverGenerator(
        api_client=api_client,
    )

    print("Running simplified pipeline (search → explorer → retriever_generator)...")
    print("-" * 60)

    save_dir = None
    if args.save_steps:
        save_root = config.get("pipeline", {}).get("save_dir", "outputs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(PROJECT_ROOT) / save_root / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)

    paper_ids = search_explorer.search(args.query)
    if save_dir:
        _save_step(save_dir, "01_search", {"paper_ids": paper_ids})

    explorer_output = explorer.explore(args.query, paper_ids)
    if save_dir:
        _save_step(save_dir, "02_explorer", asdict(explorer_output))

    result = retriever_generator.generate(args.query, explorer_output)
    if save_dir:
        _save_step(save_dir, "05_generator", asdict(result))

    if save_dir:
        print(f"Step outputs saved to: {save_dir}")

    print(f"Model used    : {result.metadata.get('model', 'unknown')}")
    print()
    print("Generated hypotheses:")
    print("-" * 60)
    for i, hypothesis in enumerate(result.hypotheses, start=1):
        print(f"{i}. {hypothesis}")


if __name__ == "__main__":
    main()
