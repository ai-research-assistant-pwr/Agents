"""Run the hypothesis generation pipeline for a single query.

Usage:
    python scripts/app/run_pipeline.py --query "Your research question here"
    python scripts/app/run_pipeline.py --query "Your research question" --save-steps
    python scripts/app/run_pipeline.py  # uses default query

Explorer is selected from config (explorer.type):
    - "weaviate"  WeaviateExplorer  (Qwen3-Embedding-8B + Weaviate vector DB)
    - "const"     ConstExplorer     (placeholder, for testing without a DB)

The API client randomly selects a model on each call weighted by the MODELS
table below. Add or adjust entries there to change the pool.

Requires GOOGLE_API_KEY to be set in the environment or in a .env file at the
project root.
"""

import argparse
import os
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
from app.api_client.google_client import GoogleAPIClient
from app.api_client.random_client import RandomAPIClient
from app.config import load_config
from app.explorer.const_explorer import ConstExplorer
from app.explorer.weaviate_explorer import WeaviateExplorer
from app.generator.api_llm_generator import APILLMGenerator
from app.retriever.api_llm_retriever import APILLMRetriever

CONFIG_PATH = "config/app/config.yaml"
DEFAULT_QUERY = (
    "What are the mechanisms by which transformer attention heads specialize "
    "during pre-training, and how does this relate to emergent capabilities?"
)

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

    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: GOOGLE_API_KEY is not set.")
        print(
            "Set it in your environment or add it to a .env file at the project root."
        )
        sys.exit(1)

    config = load_config(CONFIG_PATH)
    explorer_type = config.get("explorer", {}).get("type", "const")

    model_pool = ", ".join(f"{name}(w={cfg['weight']})" for name, cfg in MODELS.items())
    print(f"Explorer     : {explorer_type}")
    print(f"Model pool   : {model_pool}")
    print(f"Refinements  : {args.refinement_turns}")
    print(f"Save steps   : {args.save_steps}")
    print(f"Query        : {args.query}")
    print()

    # Wire up the pipeline components
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
