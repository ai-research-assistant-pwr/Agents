"""Evaluate hypotheses from a single pipeline run using LLM judges.

Usage:
    python scripts/hypotheses_evaluation/evaluate_run.py
    python scripts/hypotheses_evaluation/evaluate_run.py --run-dir outputs/20260404_130041
    python scripts/hypotheses_evaluation/evaluate_run.py --run-dir outputs/20260404_130041 --model gemini-2.5-flash
    python scripts/hypotheses_evaluation/evaluate_run.py --provider openai --model gpt-4o

When --run-dir is omitted the script automatically picks the most recently
modified run subdirectory inside outputs/.

Each generated hypothesis is scored on three metrics:
    Groundedness (0-4) — how well the hypothesis is grounded in the
                          evidence produced by the retriever.
    Relevancy    (0-4) — how relevant the hypothesis is to the user query.
    Clarity      (0-3) — how clear and well-expressed the hypothesis is.

If the run includes retriever-refinement turns the last refined retriever
output is used as the evidence source for groundedness evaluation.

Requires GOOGLE_API_KEY (for --provider google) or OPENAI_API_KEY (for
--provider openai) to be set in the environment or in a .env file at the
project root.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import types as _types

from dotenv import load_dotenv

# Resolve project root and add src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

# Stub out the weaviate package so that importing app.api_client.google_client
# does not fail in environments where weaviate is not installed.
# The evaluation script only needs the API client, not the explorer.
_weaviate = _types.ModuleType("weaviate")
_weaviate_classes = _types.ModuleType("weaviate.classes")
_weaviate_classes_query = _types.ModuleType("weaviate.classes.query")
_weaviate_classes_query.MetadataQuery = None  # type: ignore[attr-defined]
_weaviate.classes = _weaviate_classes  # type: ignore[attr-defined]
sys.modules.setdefault("weaviate", _weaviate)
sys.modules.setdefault("weaviate.classes", _weaviate_classes)
sys.modules.setdefault("weaviate.classes.query", _weaviate_classes_query)

from app.api_client.base import BaseAPIClient
from app.api_client.google_client import GoogleAPIClient
from app.api_client.openai_client import OpenAIAPIClient
from hypotheses_evaluation import (
    ClarityJudge,
    GroundednessJudge,
    JudgeResult,
    RelevancyJudge,
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODELS = {
    "google": "gemini-3-flash-preview",
    "openai": "gpt-4o",
}
PROVIDERS = list(DEFAULT_MODELS.keys())
DEFAULT_PROVIDER = "google"

# Width of the label column in the per-hypothesis table.
_LABEL_W = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _find_latest_run(outputs_dir: Path) -> Path:
    """Return the most recently modified timestamped run subdirectory."""
    candidates = sorted(
        (d for d in outputs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run subdirectories found in {outputs_dir}")
    return candidates[0]


def _find_last_retriever_file(run_dir: Path) -> Path:
    """Return the last retriever output file in the run directory.

    Handles both the plain ``02_retriever.json`` and any
    ``03_retriever_refinement_turn_N.json`` files produced by the
    refinement loop, always returning the one that feeds the generator.
    """
    refinement_files = sorted(run_dir.glob("03_retriever_refinement_turn_*.json"))
    if refinement_files:
        return refinement_files[-1]
    plain = run_dir / "02_retriever.json"
    if plain.exists():
        return plain
    raise FileNotFoundError(f"No retriever output file found in {run_dir}")


def _resolve_run_dir(arg: str | None) -> Path:
    if arg is None:
        run_dir = _find_latest_run(OUTPUTS_DIR)
        print(
            f"No --run-dir specified. Using latest run: {run_dir.relative_to(PROJECT_ROOT)}"
        )
        return run_dir

    path = Path(arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_rule(char: str = "-", width: int = 70) -> None:
    print(char * width)


def _print_score_row(label: str, result: JudgeResult, max_score: int) -> None:
    bar_filled = "#" * result.score
    bar_empty = "." * (max_score - result.score)
    bar = f"[{bar_filled}{bar_empty}]"
    score_str = f"{result.score}/{max_score}"
    print(f"  {label:<{_LABEL_W}} {score_str:>4}  {bar}  ({result.model})")
    # Wrap reasoning at ~65 chars
    reasoning = result.reasoning.strip().replace("\n", " ")
    indent = " " * (_LABEL_W + 10)
    words = reasoning.split()
    line, lines = [], []
    for word in words:
        if sum(len(w) + 1 for w in line) + len(word) > 65:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    for ln in lines:
        print(f"{indent}{ln}")


def _print_summary(all_scores: list[dict]) -> None:
    """Print a compact summary table of mean scores across all hypotheses."""
    if not all_scores:
        return
    metrics = ["groundedness", "relevancy", "clarity"]
    max_scores = {"groundedness": 4, "relevancy": 4, "clarity": 3}
    _print_rule("=")
    print("SUMMARY")
    _print_rule("=")
    header = f"  {'Hypothesis':<14}" + "".join(
        f"  {m.capitalize():>13}" for m in metrics
    )
    print(header)
    _print_rule()
    for i, scores in enumerate(all_scores, start=1):
        row = f"  {f'#{i}':<14}"
        for m in metrics:
            s = scores[m]
            mx = max_scores[m]
            row += f"  {s}/{mx}{'':>10}"
        print(row)
    _print_rule()
    means_row = f"  {'Mean':<14}"
    for m in metrics:
        mean = sum(s[m] for s in all_scores) / len(all_scores)
        mx = max_scores[m]
        means_row += f"  {mean:.2f}/{mx}{'':>7}"
    print(means_row)
    _print_rule("=")


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def evaluate_run(run_dir: Path, model: str, api_client: BaseAPIClient) -> None:
    # --- Load pipeline artifacts ---
    explorer_data = _load_json(run_dir / "01_explorer.json")
    retriever_path = _find_last_retriever_file(run_dir)
    retriever_data = _load_json(retriever_path)
    generator_path = run_dir / "04_generator.json"
    if not generator_path.exists():
        print(f"ERROR: Generator output not found at {generator_path}.")
        print("The run may be incomplete (pipeline did not finish).")
        sys.exit(1)
    generator_data = _load_json(generator_path)

    query: str = explorer_data["metadata"]["prompt"]
    evidence: str = retriever_data["content"]
    hypotheses: list[str] = generator_data["hypotheses"]

    # --- Summarise what we loaded ---
    _print_rule("=")
    print(f"Run directory : {run_dir.relative_to(PROJECT_ROOT)}")
    print(f"Evidence from : {retriever_path.name}")
    print(f"Hypotheses    : {len(hypotheses)}")
    print(f"Judge model   : {model}")
    _print_rule("=")
    print(f"Query: {query}")
    _print_rule()

    # --- Instantiate judges ---
    groundedness_judge = GroundednessJudge(api_client)
    relevancy_judge = RelevancyJudge(api_client)
    clarity_judge = ClarityJudge(api_client)

    # --- Evaluate each hypothesis ---
    all_scores: list[dict] = []

    for i, hypothesis in enumerate(hypotheses, start=1):
        print(f"\nHypothesis {i}/{len(hypotheses)}")
        _print_rule()
        print(hypothesis)
        _print_rule()
        print("  Evaluating...")

        g_result = groundedness_judge.judge(hypothesis=hypothesis, evidence=evidence)
        r_result = relevancy_judge.judge(hypothesis=hypothesis, query=query)
        c_result = clarity_judge.judge(hypothesis=hypothesis)

        print()
        _print_score_row("Groundedness", g_result, max_score=4)
        _print_score_row("Relevancy", r_result, max_score=4)
        _print_score_row("Clarity", c_result, max_score=3)

        all_scores.append(
            {
                "groundedness": g_result.score,
                "relevancy": r_result.score,
                "clarity": c_result.score,
            }
        )

    print()
    _print_summary(all_scores)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM judges on hypotheses from a saved pipeline run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Path to the pipeline run directory to evaluate "
            "(e.g. outputs/20260404_130041). "
            "Relative paths are resolved from the project root. "
            "Defaults to the most recently modified run in outputs/."
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=PROVIDERS,
        default=DEFAULT_PROVIDER,
        help=f"LLM provider to use for evaluation (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model to use for all three judges. "
            f"Defaults depend on provider: "
            + ", ".join(f"{k}={v}" for k, v in DEFAULT_MODELS.items())
            + "."
        ),
    )
    return parser.parse_args()


def _build_api_client(provider: str, model: str) -> BaseAPIClient:
    """Instantiate the appropriate API client for the given provider."""
    if provider == "google":
        return GoogleAPIClient(model=model)
    elif provider == "openai":
        return OpenAIAPIClient(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _check_api_key(provider: str) -> None:
    """Verify the required API key is present in the environment."""
    key_map = {
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    key_name = key_map[provider]
    if key_name not in os.environ:
        print(f"ERROR: {key_name} is not set.")
        print(
            "Set it in your environment or add it to a .env file at the project root."
        )
        sys.exit(1)


def main() -> None:
    args = parse_args()

    provider: str = args.provider
    model: str = args.model or DEFAULT_MODELS[provider]

    _check_api_key(provider)

    run_dir = _resolve_run_dir(args.run_dir)
    api_client = _build_api_client(provider, model)
    evaluate_run(run_dir=run_dir, model=model, api_client=api_client)


if __name__ == "__main__":
    main()
