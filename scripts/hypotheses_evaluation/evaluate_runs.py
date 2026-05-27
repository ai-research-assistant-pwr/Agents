"""Evaluate hypotheses from all pipeline runs inside a given directory.

Usage:
    python scripts/hypotheses_evaluation/evaluate_runs.py --runs-dir outputs/batch_20260409
    python scripts/hypotheses_evaluation/evaluate_runs.py --runs-dir outputs/batch_20260409 --model gemini-2.5-flash
    python scripts/hypotheses_evaluation/evaluate_runs.py --provider openai --model gpt-4o
    python scripts/hypotheses_evaluation/evaluate_runs.py --runs-dir outputs/batch_20260409 --workers 8

Each subdirectory of --runs-dir is treated as an individual pipeline run and
evaluated in turn.  When --runs-dir is omitted the script falls back to the
top-level outputs/ directory and evaluates every run found there.

Runs are evaluated in parallel using a thread pool (--workers controls the
degree of parallelism; default is 4).  Each run's output is buffered and
printed atomically once that run finishes, so the console output remains
readable even under high concurrency.

Each generated hypothesis is scored on three metrics:
    Groundedness (0-4) — how well the hypothesis is grounded in the
                          evidence produced by the retriever.
    Relevancy    (0-4) — how relevant the hypothesis is to the user query.
    Clarity      (0-3) — how clear and well-expressed the hypothesis is.

A per-run summary and a final aggregate summary across all runs are printed at
the end.

Requires GOOGLE_API_KEY (for --provider google) or OPENAI_API_KEY (for
--provider openai) to be set in the environment or in a .env file at the
project root.
"""

import argparse
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from dotenv import load_dotenv

# Resolve project root and add src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

import numpy as np
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

_LABEL_W = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _find_run_dirs(runs_dir: Path) -> list[Path]:
    """Return all subdirectories of *runs_dir* sorted by name."""
    dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    print(f"dirs {dirs}")
    if not dirs:
        raise FileNotFoundError(f"No run subdirectories found in {runs_dir}")
    return dirs


def _find_last_retriever_file(run_dir: Path) -> Path:
    refinement_files = sorted(run_dir.glob("03_retriever_refinement_turn_*.json"))
    if refinement_files:
        return refinement_files[-1]
    plain = run_dir / "03_retriever.json"
    if plain.exists():
        return plain
    raise FileNotFoundError(f"No retriever output file found in {run_dir}")


def _resolve_runs_dir(arg: str | None) -> Path:
    if arg is None:
        print(
            f"No --runs-dir specified. Using default: {OUTPUTS_DIR.relative_to(PROJECT_ROOT)}"
        )
        return OUTPUTS_DIR

    path = Path(arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Runs directory not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_rule(char: str = "-", width: int = 70, out: IO[str] = sys.stdout) -> None:
    print(char * width, file=out)


def _print_score_row(
    label: str, result: JudgeResult, max_score: int, out: IO[str] = sys.stdout
) -> None:
    bar_filled = "#" * result.score
    bar_empty = "." * (max_score - result.score)
    bar = f"[{bar_filled}{bar_empty}]"
    score_str = f"{result.score}/{max_score}"
    print(f"  {label:<{_LABEL_W}} {score_str:>4}  {bar}  ({result.model})", file=out)
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
        print(f"{indent}{ln}", file=out)


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


def _print_run_summary(
    all_scores: list[dict], diversity: float, novelty: float, out: IO[str] = sys.stdout
) -> None:
    """Print a compact summary table of mean scores for a single run."""
    if not all_scores:
        return
    metrics = ["groundedness", "relevancy", "clarity"]
    max_scores = {"groundedness": 4, "relevancy": 4, "clarity": 3}
    _print_rule("=", out=out)
    print("RUN SUMMARY", file=out)
    _print_rule("=", out=out)
    header = f"  {'Hypothesis':<14}" + "".join(
        f"  {m.capitalize():>13}" for m in metrics
    )
    print(header, file=out)
    _print_rule(out=out)
    for i, scores in enumerate(all_scores, start=1):
        row = f"  {f'#{i}':<14}"
        for m in metrics:
            s = scores[m].score
            mx = max_scores[m]
            row += f"  {s}/{mx}{'':>10}"
        print(row, file=out)
    _print_rule(out=out)
    means_row = f"  {'Mean':<14}"
    for m in metrics:
        mean = sum(s[m].score for s in all_scores) / len(all_scores)
        mx = max_scores[m]
        means_row += f"  {mean:.2f}/{mx}{'':>7}"
    print(means_row, file=out)
    print(f"  {'Diversity':<14}  {diversity:.2f}/1.00", file=out)
    print(f"  {'Novelty':<14}  {novelty:.2f}/1.00", file=out)
    _print_rule("=", out=out)


def _print_aggregate_summary(run_results: list[dict]) -> None:
    """Print a cross-run aggregate summary table."""
    if not run_results:
        return
    metrics = ["groundedness", "relevancy", "clarity"]
    max_scores = {"groundedness": 4, "relevancy": 4, "clarity": 3}

    _print_rule("*", 70)
    print("AGGREGATE SUMMARY  (all runs)")
    _print_rule("*", 70)
    header = (
        f"  {'Run':<20}"
        + "".join(f"  {m.capitalize():>13}" for m in metrics)
        + "  Diversity  Novelty"
    )
    print(header)
    _print_rule()

    grand: dict[str, list[float]] = {m: [] for m in metrics}
    grand_diversity: list[float] = []
    grand_novelty: list[float] = []

    for result in run_results:
        name = result["run_name"]
        scores = result["mean_scores"]
        row = f"  {name:<20}"
        for m in metrics:
            mean = scores[m]
            mx = max_scores[m]
            row += f"  {mean:.2f}/{mx}{'':>7}"
        div = result.get("diversity", 0.0)
        nov = result.get("novelty", 0.0)
        row += f"  {div:<9.2f}  {nov:<7.2f}"
        print(row)
        for m in metrics:
            grand[m].append(scores[m])
        grand_diversity.append(div)
        grand_novelty.append(nov)

    _print_rule()
    grand_row = f"  {'Grand Mean':<20}"
    for m in metrics:
        gm = sum(grand[m]) / len(grand[m]) if grand[m] else 0.0
        mx = max_scores[m]
        grand_row += f"  {gm:.2f}/{mx}{'':>7}"
    gdiv = sum(grand_diversity) / len(grand_diversity) if grand_diversity else 0.0
    gnov = sum(grand_novelty) / len(grand_novelty) if grand_novelty else 0.0
    grand_row += f"  {gdiv:<9.2f}  {gnov:<7.2f}"
    print(grand_row)
    _print_rule("*", 70)


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def evaluate_run(
    run_dir: Path,
    model: str,
    groundedness_judge: GroundednessJudge,
    relevancy_judge: RelevancyJudge,
    clarity_judge: ClarityJudge,
    out: IO[str] = sys.stdout,
) -> dict | None:
    """Evaluate a single run directory.

    Returns a dict with keys ``query``, ``hypotheses``, ``mean_scores``,
    ``diversity``, and ``novelty``, or None if the run is incomplete.
    """

    try:
        explorer_data = _load_json(run_dir / "02_explorer.json")
        retriever_path = _find_last_retriever_file(run_dir)
        retriever_data = _load_json(retriever_path)
    except (FileNotFoundError, KeyError) as exc:
        print(f"  SKIP: could not load run artifacts — {exc}", file=out)
        return None

    generator_path = run_dir / "05_generator.json"
    if not generator_path.exists():
        print(f"  SKIP: generator output not found (run may be incomplete).", file=out)
        return None

    generator_data = _load_json(generator_path)

    query: str = explorer_data["metadata"]["prompt"]
    evidence: str = retriever_data["content"]
    hypotheses: list[str] = generator_data["hypotheses"]
    explorer_metadata = explorer_data.get("metadata", {})
    paper_ids = [p.get("id") for p in explorer_metadata.get("papers", [])]

    _print_rule("=", out=out)
    print(f"Run directory : {run_dir.relative_to(PROJECT_ROOT)}", file=out)
    print(f"Evidence from : {retriever_path.name}", file=out)
    print(f"Hypotheses    : {len(hypotheses)}", file=out)
    print(f"Judge model   : {model}", file=out)
    _print_rule("=", out=out)
    print(f"Query: {query}", file=out)
    _print_rule(out=out)

    hypothesis_results: list[dict] = []
    all_scores: list[dict] = []

    for i, hypothesis in enumerate(hypotheses, start=1):
        print(f"\nHypothesis {i}/{len(hypotheses)}", file=out)
        _print_rule(out=out)
        print(hypothesis, file=out)
        _print_rule(out=out)
        print("  Evaluating...", file=out)

        g_result = groundedness_judge.judge(hypothesis=hypothesis, evidence=evidence)
        r_result = relevancy_judge.judge(hypothesis=hypothesis, query=query)
        c_result = clarity_judge.judge(hypothesis=hypothesis)

        print(file=out)
        _print_score_row("Groundedness", g_result, max_score=4, out=out)
        _print_score_row("Relevancy", r_result, max_score=4, out=out)
        _print_score_row("Clarity", c_result, max_score=3, out=out)

        hypothesis_results.append(
            {
                "text": hypothesis,
                "groundedness": {
                    "score": g_result.score,
                    "reasoning": g_result.reasoning,
                },
                "relevancy": {"score": r_result.score, "reasoning": r_result.reasoning},
                "clarity": {"score": c_result.score, "reasoning": c_result.reasoning},
            }
        )

        all_scores.append(
            {
                "groundedness": g_result,
                "relevancy": r_result,
                "clarity": c_result,
            }
        )

    diversity = _calculate_diversity(hypotheses)
    existing_hypotheses = _get_existing_hypotheses(paper_ids)
    novelty = _calculate_novelty(hypotheses, existing_hypotheses)

    print(file=out)
    _print_run_summary(all_scores, diversity, novelty, out=out)

    metrics = ["groundedness", "relevancy", "clarity"]
    mean_scores = {
        m: sum(s[m].score for s in all_scores) / len(all_scores) for m in metrics
    }

    # --- Save per-run evaluation results ---
    save_payload = {
        "metadata": {
            "judge_model": model,
            "num_hypotheses": len(hypotheses),
            "evaluated_at": datetime.now().isoformat(),
        },
        "hypotheses": [
            {
                "index": i,
                "hypothesis": h["text"],
                "groundedness": {
                    "score": h["groundedness"]["score"],
                    "reasoning": h["groundedness"]["reasoning"],
                },
                "relevancy": {
                    "score": h["relevancy"]["score"],
                    "reasoning": h["relevancy"]["reasoning"],
                },
                "clarity": {
                    "score": h["clarity"]["score"],
                    "reasoning": h["clarity"]["reasoning"],
                },
            }
            for i, h in enumerate(hypothesis_results, start=1)
        ],
        "summary": {
            "mean_groundedness": round(mean_scores["groundedness"], 2),
            "mean_relevancy": round(mean_scores["relevancy"], 2),
            "mean_clarity": round(mean_scores["clarity"], 2),
            "diversity": round(diversity, 2),
            "novelty": round(novelty, 2),
        },
    }

    save_path = run_dir / "06_evaluation.json"
    with open(save_path, "w") as f:
        json.dump(save_payload, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {save_path.relative_to(PROJECT_ROOT)}", file=out)

    return {
        "query": query,
        "hypotheses": hypothesis_results,
        "mean_scores": mean_scores,
        "diversity": diversity,
        "novelty": novelty,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_results(runs_dir: Path, model: str, run_results: list[dict]) -> Path:
    """Serialise *run_results* to JSON and write it inside *runs_dir*.

    The output file is named ``evaluation_results.json``.  Any existing file
    with that name is overwritten.
    """
    metrics = ["groundedness", "relevancy", "clarity"]

    # Compute aggregate means across all evaluated runs
    if run_results:
        aggregate_mean_scores = {
            m: sum(r["mean_scores"][m] for r in run_results) / len(run_results)
            for m in metrics
        }
    else:
        aggregate_mean_scores = {m: None for m in metrics}

    # Compute aggregate diversity and novelty across all runs
    diversities = [
        r.get("diversity", 0.0) for r in run_results if r.get("diversity") is not None
    ]
    novelties = [
        r.get("novelty", 0.0) for r in run_results if r.get("novelty") is not None
    ]

    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "runs_dir": str(runs_dir.relative_to(PROJECT_ROOT)),
        "runs": run_results,
        "aggregate_mean_scores": aggregate_mean_scores,
        "aggregate_diversity": (
            round(sum(diversities) / len(diversities), 2) if diversities else None
        ),
        "aggregate_novelty": (
            round(sum(novelties) / len(novelties), 2) if novelties else None
        ),
    }

    out_path = runs_dir / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM judges on hypotheses from all pipeline runs inside a directory."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help=(
            "Path to a directory whose subdirectories are individual pipeline runs "
            "(e.g. outputs/batch_20260409). "
            "Relative paths are resolved from the project root. "
            "Defaults to outputs/."
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
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Maximum number of runs to evaluate in parallel (default: 8). "
            "Each worker uses a separate API client. "
            "Set to 1 to disable parallelism."
        ),
    )
    return parser.parse_args()


def _evaluate_run_worker(
    run_dir: Path,
    model: str,
    provider: str,
    run_index: int,
    total_runs: int,
) -> tuple[Path, dict | None, str]:
    """Worker that runs in a thread pool, capturing all output to a buffer.

    Creates its own API client and judge instances so that concurrent workers
    do not share any mutable state.  Returns the run directory, the result
    dict (or None if the run was skipped), and the captured output string.
    """
    buf = io.StringIO()
    print(f"\n{'#' * 70}", file=buf)
    print(f"# Run {run_index}/{total_runs}: {run_dir.name}", file=buf)
    print(f"{'#' * 70}", file=buf)

    api_client = _build_api_client(provider, model)
    groundedness_judge = GroundednessJudge(api_client)
    relevancy_judge = RelevancyJudge(api_client)
    clarity_judge = ClarityJudge(api_client)

    result = evaluate_run(
        run_dir=run_dir,
        model=model,
        groundedness_judge=groundedness_judge,
        relevancy_judge=relevancy_judge,
        clarity_judge=clarity_judge,
        out=buf,
    )
    return run_dir, result, buf.getvalue()


def main() -> None:
    args = parse_args()

    provider: str = args.provider
    model: str = args.model or DEFAULT_MODELS[provider]
    workers: int = max(1, args.workers)

    _check_api_key(provider)

    runs_dir = _resolve_runs_dir(args.runs_dir)
    run_dirs = _find_run_dirs(runs_dir)

    print(
        f"Found {len(run_dirs)} run(s) in {runs_dir.relative_to(PROJECT_ROOT)} "
        f"(workers={workers})"
    )

    print_lock = threading.Lock()
    results_map: dict[Path, dict | None] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_run = {
            executor.submit(
                _evaluate_run_worker, run_dir, model, provider, idx, len(run_dirs)
            ): run_dir
            for idx, run_dir in enumerate(run_dirs, start=1)
        }

        for future in as_completed(future_to_run):
            run_dir_key = future_to_run[future]
            try:
                _, result, output = future.result()
            except Exception as exc:
                with print_lock:
                    print(f"\nERROR evaluating {run_dir_key.name}: {exc}")
                results_map[run_dir_key] = None
            else:
                with print_lock:
                    print(output, end="")
                results_map[run_dir_key] = result

    # Restore submission order for the aggregate summary and saved JSON.
    run_results: list[dict] = []
    for run_dir in run_dirs:
        result = results_map.get(run_dir)
        if result is not None:
            run_results.append({"run_name": run_dir.name, **result})

    print()
    _print_aggregate_summary(run_results)

    out_path = _save_results(runs_dir=runs_dir, model=model, run_results=run_results)
    print(f"\nResults saved to: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
