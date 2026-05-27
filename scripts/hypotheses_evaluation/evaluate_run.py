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
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Resolve project root and add src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from weaviate.collections.classes.filters import Filter

from app.api_client.base import BaseAPIClient
from app.api_client.google_client import GoogleAPIClient
from app.api_client.openai_client import OpenAIAPIClient
from app.explorer.tools.weaviate_tools import (
    _get_weaviate_client,
    _load_embedding_model,
)
from hypotheses_evaluation import (
    ClarityJudge,
    GroundednessJudge,
    JudgeResult,
    RelevancyJudge,
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODELS = {
    "google": "gemini-3.5-flash",
    "openai": "gpt-5.4-mini",
}
PROVIDERS = list(DEFAULT_MODELS.keys())
DEFAULT_PROVIDER = "openai"

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

    Handles both the plain ``03_retriever.json`` and any
    ``04_retriever_refinement_turn_N.json`` files produced by the
    refinement loop, always returning the one that feeds the generator.
    """
    refinement_files = sorted(run_dir.glob("04_retriever_refinement_turn_*.json"))
    if refinement_files:
        return refinement_files[-1]
    plain = run_dir / "03_retriever.json"
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


def _calculate_diversity(hypotheses: list[str]) -> float:
    if len(hypotheses) < 2:
        return 0.0

    model = _load_embedding_model()
    embeddings = model.embed(hypotheses)
    embeddings = np.array([e.outputs.embedding for e in embeddings])

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    similarity_matrix = embeddings @ embeddings.T

    n = len(hypotheses)
    diversities = []
    for i in range(n):
        for j in range(i + 1, n):
            diversities.append(1 - similarity_matrix[i, j])

    return np.mean(diversities) if diversities else 0.0


def _get_existing_hypotheses(paper_ids: list[str]) -> list[str]:
    if not paper_ids:
        return []

    weaviate_client = _get_weaviate_client()
    collection = weaviate_client.collections.get("ResearchPapers")

    type_filter = Filter.by_property("type").equal("HYPOTHESIS")
    paper_filter = Filter.by_property("paperId").contains_any(paper_ids)
    combined_filter = type_filter & paper_filter

    response = collection.query.fetch_objects(
        filters=combined_filter,
        limit=1000,
        return_properties=["content"],
    )

    return [obj.properties.get("content", "") for obj in response.objects]


def _calculate_novelty(
    generated_hypotheses: list[str], existing_hypotheses: list[str]
) -> float:
    if not generated_hypotheses or not existing_hypotheses:
        return 0.0

    model = _load_embedding_model()

    generated_embeddings = model.embed(generated_hypotheses)
    generated_embeddings = np.array([e.outputs.embedding for e in generated_embeddings])
    generated_embeddings = generated_embeddings / np.linalg.norm(
        generated_embeddings, axis=1, keepdims=True
    )

    existing_embeddings = model.embed(existing_hypotheses)
    existing_embeddings = np.array([e.outputs.embedding for e in existing_embeddings])
    existing_embeddings = existing_embeddings / np.linalg.norm(
        existing_embeddings, axis=1, keepdims=True
    )

    similarities = []
    for gen_emb in generated_embeddings:
        for exist_emb in existing_embeddings:
            similarities.append(1 - np.dot(gen_emb, exist_emb))

    return np.mean(similarities) if similarities else 0.0


def _print_summary(all_scores: list[dict], diversity: float, novelty: float) -> None:
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
            s = scores[m].score
            mx = max_scores[m]
            row += f"  {s}/{mx}{'':>10}"
        print(row)
    _print_rule()
    means_row = f"  {'Mean':<14}"
    for m in metrics:
        mean = sum(s[m].score for s in all_scores) / len(all_scores)
        mx = max_scores[m]
        means_row += f"  {mean:.2f}/{mx}{'':>7}"
    print(means_row)
    print(f"  {'Diversity':<14}  {diversity:.2f}/1.00")
    print(f"  {'Novelty':<14}  {novelty:.2f}/1.00")
    _print_rule("=")


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def evaluate_run(run_dir: Path, model: str, api_client: BaseAPIClient) -> None:
    # --- Load pipeline artifacts ---
    explorer_data = _load_json(run_dir / "02_explorer.json")
    retriever_path = _find_last_retriever_file(run_dir)
    retriever_data = _load_json(retriever_path)
    generator_path = run_dir / "05_generator.json"
    if not generator_path.exists():
        print(f"ERROR: Generator output not found at {generator_path}.")
        print("The run may be incomplete (pipeline did not finish).")
        sys.exit(1)
    generator_data = _load_json(generator_path)

    query: str = explorer_data["metadata"]["prompt"]
    evidence: str = retriever_data["content"]
    hypotheses: list[str] = generator_data["hypotheses"]
    explorer_metadata = explorer_data.get("metadata", {})
    paper_ids = [p.get("id") for p in explorer_metadata.get("papers", [])]

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
                "groundedness": g_result,
                "relevancy": r_result,
                "clarity": c_result,
            }
        )

    print()
    diversity = _calculate_diversity(hypotheses)
    existing_hypotheses = _get_existing_hypotheses(paper_ids)
    novelty = _calculate_novelty(hypotheses, existing_hypotheses)
    _print_summary(all_scores, diversity, novelty)

    # --- Save evaluation results to run directory ---
    results = {
        "metadata": {
            "judge_model": model,
            "num_hypotheses": len(hypotheses),
            "evaluated_at": datetime.now().isoformat(),
        },
        "hypotheses": [
            {
                "index": i,
                "hypothesis": h,
                "groundedness": {
                    "score": all_scores[i - 1]["groundedness"].score,
                    "reasoning": all_scores[i - 1]["groundedness"].reasoning,
                },
                "relevancy": {
                    "score": all_scores[i - 1]["relevancy"].score,
                    "reasoning": all_scores[i - 1]["relevancy"].reasoning,
                },
                "clarity": {
                    "score": all_scores[i - 1]["clarity"].score,
                    "reasoning": all_scores[i - 1]["clarity"].reasoning,
                },
            }
            for i, h in enumerate(hypotheses, start=1)
        ],
        "summary": {
            "mean_groundedness": round(
                sum(s["groundedness"].score for s in all_scores) / len(all_scores), 2
            ),
            "mean_relevancy": round(
                sum(s["relevancy"].score for s in all_scores) / len(all_scores), 2
            ),
            "mean_clarity": round(
                sum(s["clarity"].score for s in all_scores) / len(all_scores), 2
            ),
            "diversity": round(diversity, 2),
            "novelty": round(novelty, 2),
        },
    }

    save_path = run_dir / "06_evaluation.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {save_path.relative_to(PROJECT_ROOT)}")


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
