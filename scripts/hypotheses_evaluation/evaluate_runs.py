"""Evaluate hypotheses from all pipeline runs inside a given directory.

Usage:
    python scripts/hypotheses_evaluation/evaluate_runs.py --runs-dir outputs/batch_20260409
    python scripts/hypotheses_evaluation/evaluate_runs.py --runs-dir outputs/batch_20260409 --model gemini-2.5-flash

Each subdirectory of --runs-dir is treated as an individual pipeline run and
evaluated in turn.  When --runs-dir is omitted the script falls back to the
top-level outputs/ directory and evaluates every run found there.

Each generated hypothesis is scored on three metrics:
    Groundedness (0-4) — how well the hypothesis is grounded in the
                          evidence produced by the retriever.
    Relevancy    (0-4) — how relevant the hypothesis is to the user query.
    Clarity      (0-3) — how clear and well-expressed the hypothesis is.

A per-run summary and a final aggregate summary across all runs are printed at
the end.

Requires GOOGLE_API_KEY to be set in the environment or in a .env file at
the project root.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import types as _types

from dotenv import load_dotenv

# Resolve project root and add src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

# Stub out the weaviate package so that importing app.api_client.google_client
# does not fail in environments where weaviate is not installed.
_weaviate = _types.ModuleType("weaviate")
_weaviate_classes = _types.ModuleType("weaviate.classes")
_weaviate_classes_query = _types.ModuleType("weaviate.classes.query")
_weaviate_classes_query.MetadataQuery = None  # type: ignore[attr-defined]
_weaviate.classes = _weaviate_classes  # type: ignore[attr-defined]
sys.modules.setdefault("weaviate", _weaviate)
sys.modules.setdefault("weaviate.classes", _weaviate_classes)
sys.modules.setdefault("weaviate.classes.query", _weaviate_classes_query)

from app.api_client.google_client import GoogleAPIClient
from hypotheses_evaluation import (
    ClarityJudge,
    GroundednessJudge,
    JudgeResult,
    RelevancyJudge,
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

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
    if not dirs:
        raise FileNotFoundError(f"No run subdirectories found in {runs_dir}")
    return dirs


def _find_last_retriever_file(run_dir: Path) -> Path:
    refinement_files = sorted(run_dir.glob("03_retriever_refinement_turn_*.json"))
    if refinement_files:
        return refinement_files[-1]
    plain = run_dir / "02_retriever.json"
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


def _print_rule(char: str = "-", width: int = 70) -> None:
    print(char * width)


def _print_score_row(label: str, result: JudgeResult, max_score: int) -> None:
    bar_filled = "#" * result.score
    bar_empty = "." * (max_score - result.score)
    bar = f"[{bar_filled}{bar_empty}]"
    score_str = f"{result.score}/{max_score}"
    print(f"  {label:<{_LABEL_W}} {score_str:>4}  {bar}  ({result.model})")
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


def _print_run_summary(all_scores: list[dict]) -> None:
    """Print a compact summary table of mean scores for a single run."""
    if not all_scores:
        return
    metrics = ["groundedness", "relevancy", "clarity"]
    max_scores = {"groundedness": 4, "relevancy": 4, "clarity": 3}
    _print_rule("=")
    print("RUN SUMMARY")
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


def _print_aggregate_summary(run_results: list[dict]) -> None:
    """Print a cross-run aggregate summary table."""
    if not run_results:
        return
    metrics = ["groundedness", "relevancy", "clarity"]
    max_scores = {"groundedness": 4, "relevancy": 4, "clarity": 3}

    _print_rule("*", 70)
    print("AGGREGATE SUMMARY  (all runs)")
    _print_rule("*", 70)
    header = f"  {'Run':<20}" + "".join(f"  {m.capitalize():>13}" for m in metrics)
    print(header)
    _print_rule()

    grand: dict[str, list[float]] = {m: [] for m in metrics}

    for result in run_results:
        name = result["run_name"]
        scores = result["mean_scores"]
        row = f"  {name:<20}"
        for m in metrics:
            mean = scores[m]
            mx = max_scores[m]
            row += f"  {mean:.2f}/{mx}{'':>7}"
        print(row)
        for m in metrics:
            grand[m].append(scores[m])

    _print_rule()
    grand_row = f"  {'Grand Mean':<20}"
    for m in metrics:
        gm = sum(grand[m]) / len(grand[m]) if grand[m] else 0.0
        mx = max_scores[m]
        grand_row += f"  {gm:.2f}/{mx}{'':>7}"
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
) -> dict | None:
    """Evaluate a single run directory.

    Returns a dict with keys ``query``, ``hypotheses``, and ``mean_scores``,
    or None if the run is incomplete / cannot be loaded.
    """

    try:
        explorer_data = _load_json(run_dir / "01_explorer.json")
        retriever_path = _find_last_retriever_file(run_dir)
        retriever_data = _load_json(retriever_path)
    except (FileNotFoundError, KeyError) as exc:
        print(f"  SKIP: could not load run artifacts — {exc}")
        return None

    generator_path = run_dir / "04_generator.json"
    if not generator_path.exists():
        print(f"  SKIP: generator output not found (run may be incomplete).")
        return None

    generator_data = _load_json(generator_path)

    query: str = explorer_data["metadata"]["prompt"]
    evidence: str = retriever_data["content"]
    hypotheses: list[str] = generator_data["hypotheses"]

    _print_rule("=")
    print(f"Run directory : {run_dir.relative_to(PROJECT_ROOT)}")
    print(f"Evidence from : {retriever_path.name}")
    print(f"Hypotheses    : {len(hypotheses)}")
    print(f"Judge model   : {model}")
    _print_rule("=")
    print(f"Query: {query}")
    _print_rule()

    hypothesis_results: list[dict] = []

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

    # Flat score dicts for the summary helpers
    all_scores = [
        {
            "groundedness": h["groundedness"]["score"],
            "relevancy": h["relevancy"]["score"],
            "clarity": h["clarity"]["score"],
        }
        for h in hypothesis_results
    ]

    print()
    _print_run_summary(all_scores)

    metrics = ["groundedness", "relevancy", "clarity"]
    mean_scores = {m: sum(s[m] for s in all_scores) / len(all_scores) for m in metrics}

    return {
        "query": query,
        "hypotheses": hypothesis_results,
        "mean_scores": mean_scores,
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

    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "runs_dir": str(runs_dir.relative_to(PROJECT_ROOT)),
        "runs": run_results,
        "aggregate_mean_scores": aggregate_mean_scores,
    }

    out_path = runs_dir / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model to use for all three judges (default: {DEFAULT_MODEL}).",
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

    runs_dir = _resolve_runs_dir(args.runs_dir)
    run_dirs = _find_run_dirs(runs_dir)

    print(f"Found {len(run_dirs)} run(s) in {runs_dir.relative_to(PROJECT_ROOT)}")

    api_client = GoogleAPIClient(model=args.model)
    groundedness_judge = GroundednessJudge(api_client)
    relevancy_judge = RelevancyJudge(api_client)
    clarity_judge = ClarityJudge(api_client)

    run_results: list[dict] = []

    for idx, run_dir in enumerate(run_dirs, start=1):
        print(f"\n{'#' * 70}")
        print(f"# Run {idx}/{len(run_dirs)}: {run_dir.name}")
        print(f"{'#' * 70}")

        result = evaluate_run(
            run_dir=run_dir,
            model=args.model,
            groundedness_judge=groundedness_judge,
            relevancy_judge=relevancy_judge,
            clarity_judge=clarity_judge,
        )

        if result is not None:
            run_results.append({"run_name": run_dir.name, **result})

    print()
    _print_aggregate_summary(run_results)

    out_path = _save_results(
        runs_dir=runs_dir, model=args.model, run_results=run_results
    )
    print(f"\nResults saved to: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
