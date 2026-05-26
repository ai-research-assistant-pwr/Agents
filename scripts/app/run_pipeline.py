"""Run the hypothesis generation pipeline for a single query.

Usage:
    python scripts/app/run_pipeline.py --query "Your research question here"
    python scripts/app/run_pipeline.py --query "Your research question" --save-steps
    python scripts/app/run_pipeline.py  # uses default query

Search is selected from config (search.type):
    - "weaviate"  WeaviateSearchExplorer  (Qwen3 embedding + reranking)

Explorer is selected from config (explorer.type):
    - "weaviate"        WeaviateExplorer       (direct vector search to content)
    - "neo4j_bfs"       Neo4jBFSExplorer       (BFS traversal from paper IDs)
    - "neo4j_random_walk"  Neo4jRandomWalkExplorer (Random walk traversal)
    - "neo4j_pagerank"  Neo4jPageRankExplorer  (Personalized PageRank)
    - "agentic"         AgenticExplorer        (LLM-driven tool selection)
    - "const"           ConstExplorer          (placeholder, for testing without a DB)
    - "none"            NoneExplorer           (pass-through, fetches paper metadata from Weaviate)

A model is randomly selected once at startup from the MODELS list below,
weighted by the 'weight' field. The matching API client is then instantiated
and used for the entire run. Add or adjust entries in MODELS to change the pool.

Requires the API key for the selected provider to be set in the environment or
in a .env file at the project root (GOOGLE_API_KEY, OPENAI_API_KEY, or
CEREBRAS_API_KEY).
"""

import argparse
import os
import random
import sys

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
from app.config import load_config
from app.explorer.agentic_explorer import AgenticExplorer
from app.explorer.const_explorer import ConstExplorer
from app.explorer.none_explorer import NoneExplorer
from app.explorer.neo4j_bfs_explorer import Neo4jBFSExplorer
from app.explorer.neo4j_pagerank_explorer import Neo4jPageRankExplorer
from app.explorer.neo4j_random_walk_explorer import Neo4jRandomWalkExplorer
from app.explorer.weaviate_explorer import WeaviateExplorer
from app.explorer.weaviate_search_explorer import WeaviateSearchExplorer
from app.generator.api_llm_generator import APILLMGenerator
from app.retriever.api_llm_retriever import APILLMRetriever

CONFIG_PATH = "config/app/config.yaml"
DEFAULT_QUERY = "I am interested in Mixtures of Experts (MoE) models for efficient inference. What are some recent research papers on this topic, and what hypotheses can we generate about future directions in this area?"

# Maps provider name -> client class.
PROVIDERS: dict[str, type[BaseAPIClient]] = {
    "google": GoogleAPIClient,
    "openai": OpenAIAPIClient,
    "cerebras": CerebrasAPIClient,
}

# Model pool: each entry needs 'name', 'provider', and 'weight'.
# weight controls relative selection probability (higher = more likely).
MODELS: list[dict] = [
    {"name": "gemini-3-flash-preview", "provider": "google", "weight": 1},
    {"name": "gemini-3.1-flash-lite-preview", "provider": "google", "weight": 1},
    # {"name": "gpt-5.4-mini", "provider": "openai", "weight": 1},
]


def select_model() -> dict:
    """Randomly pick one model config from MODELS, respecting weights."""
    weights = [m["weight"] for m in MODELS]
    return random.choices(MODELS, weights=weights, k=1)[0]


def build_api_client(model_cfg: dict) -> BaseAPIClient:
    """Instantiate the API client for the given model config dict."""
    client_cls = PROVIDERS[model_cfg["provider"]]
    return client_cls(model=model_cfg["name"])


def build_search(config: dict):
    """Instantiate the search explorer specified by config[search][type]."""
    search_type = config.get("search", {}).get("type")
    if search_type == "weaviate":
        return WeaviateSearchExplorer(config)
    raise ValueError(f"Unknown search type: {search_type!r}")


def build_explorer(config: dict):
    """Instantiate the explorer specified by config[explorer][type]."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scientific hypothesis generation pipeline."
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
    parser.add_argument(
        "--refinement-turns",
        type=int,
        default=0,
        help="Number of retriever<->generator refinement rounds (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(CONFIG_PATH)
    search_type = config.get("search", {}).get("type", "weaviate")
    explorer_type = config.get("explorer", {}).get("type", "const")

    # Select a model once for this run.
    model_cfg = select_model()
    model_name = model_cfg["name"]
    provider = model_cfg["provider"]

    print(f"Search        : {search_type}")
    print(f"Explorer      : {explorer_type}")
    print(f"Model         : {model_name}  (provider={provider})")
    print(f"Refinements   : {args.refinement_turns}")
    print(f"Save steps    : {args.save_steps}")
    print(f"Query         : {args.query}")
    print()

    # Instantiate the client (raises ValueError if the API key is missing).
    api_client = build_api_client(model_cfg)
    search_explorer = build_search(config)
    explorer = build_explorer(config)
    retriever = APILLMRetriever(api_client=api_client)
    generator = APILLMGenerator(api_client=api_client)

    app = App(
        search_explorer=search_explorer,
        explorer=explorer,
        retriever=retriever,
        generator=generator,
        config_path=CONFIG_PATH,
    )

    # Override config values from CLI flags
    app.config.setdefault("pipeline", {})
    app.config["pipeline"]["save_steps"] = args.save_steps
    app.config["pipeline"]["refinement_turns"] = args.refinement_turns

    print("Running pipeline...")
    print("-" * 60)

    result = app.run(args.query)

    print(f"Model used   : {result.metadata.get('model', 'unknown')}")
    print()
    print("Generated hypotheses:")
    print("-" * 60)
    for i, hypothesis in enumerate(result.hypotheses, start=1):
        print(f"{i}. {hypothesis}")

    if args.save_steps:
        save_dir = app.config.get("pipeline", {}).get("save_dir", "outputs")
        print()
        print(f"Step outputs saved to: {os.path.join(PROJECT_ROOT, save_dir)}/")


if __name__ == "__main__":
    main()
