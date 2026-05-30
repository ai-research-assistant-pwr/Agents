"""Compute Vendi Score diversity for hypotheses from pipeline run(s).

Computes the Vendi Score (Friedman & Dieng, 2023) using one of two
methods:

  - embedding (default): cosine-similarity matrix from SPECTER2 embeddings.
  - judge:              pairwise binary similarity from a local LLM judge
                        (Qwen/Qwen2.5-7B-Instruct via vLLM).

Usage:
    # Single run (embedding method)
    python scripts/hypotheses_evaluation/calculate_diversity.py \\
        --run-dir outputs/20260404_130041

    # Batch with judge method
    python scripts/hypotheses_evaluation/calculate_diversity.py \\
        --runs-dir outputs/batch_20260409 --method judge

    # Custom batch size (embedding only)
    python scripts/hypotheses_evaluation/calculate_diversity.py \\
        --run-dir outputs/20260404_130041 --batch-size 32

Requires a GPU with CUDA for reasonable performance.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from hypotheses_evaluation.diversity import (
    calculate_diversity as compute_vendi_embedding,
)
from hypotheses_evaluation.diversity import (
    calculate_judge_diversity as compute_vendi_judge,
)
from hypotheses_evaluation.diversity import (
    calculate_judge_diversity_gemini as compute_vendi_gemini,
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _find_latest_run(outputs_dir: Path) -> Path:
    candidates = sorted(
        (d for d in outputs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run subdirectories found in {outputs_dir}")
    return candidates[0]


def _find_run_dirs(runs_dir: Path) -> list[Path]:
    dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No run subdirectories found in {runs_dir}")
    return dirs


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


def _get_hypotheses(run_dir: Path) -> list[str] | None:
    generator_path = run_dir / "05_generator.json"
    if not generator_path.exists():
        print(f"  SKIP: {run_dir.name} — generator output not found.")
        return None
    data = _load_json(generator_path)
    hypotheses = data.get("hypotheses")
    if not hypotheses:
        print(f"  SKIP: {run_dir.name} — no hypotheses found.")
        return None
    return hypotheses


METHOD_LABELS = {
    "embedding": "all-mpnet-base-v2",
    "judge": "Qwen/Qwen3-4B-Instruct-2507",
    "gemini": "gemini-3.5-flash",
}


def _save_vendi_score(
    run_dir: Path, score: float, num_hypotheses: int, method: str
) -> Path:
    payload = {
        "method": method,
        "model": METHOD_LABELS[method],
        "num_hypotheses": num_hypotheses,
        "vendi_score": round(score, 4),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_path = run_dir / "07_diversity_vendi.json"
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return save_path


def _compute(method: str, hypotheses: list[str], batch_size: int) -> float:
    if method == "embedding":
        return compute_vendi_embedding(hypotheses, batch_size=batch_size)
    elif method == "judge":
        return compute_vendi_judge(hypotheses)
    elif method == "gemini":
        return compute_vendi_gemini(hypotheses)
    else:
        raise ValueError(f"Unknown method: {method}")


def process_run(run_dir: Path, batch_size: int, method: str) -> dict | None:
    hypotheses = _get_hypotheses(run_dir)
    if hypotheses is None:
        return None

    print(f"  Hypotheses: {len(hypotheses)}")
    score = _compute(method, hypotheses, batch_size)
    print(f"  Vendi Score ({method}): {score:.4f}")

    save_path = _save_vendi_score(run_dir, score, len(hypotheses), method)
    print(f"  Saved to: {save_path.relative_to(PROJECT_ROOT)}")

    return {
        "run": run_dir.name,
        "method": method,
        "vendi_score": score,
        "num_hypotheses": len(hypotheses),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Vendi Score diversity for pipeline run(s)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a single pipeline run directory. Defaults to the latest run.",
    )
    group.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Path to a directory of pipeline run subdirectories.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="embedding",
        choices=["embedding", "judge", "gemini"],
        help="Diversity method: 'embedding' (SPECTER2 cosine) or 'judge' (LLM pairwise).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding encoding (default: 64, embedding method only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method = args.method

    if args.runs_dir is not None:
        runs_dir = _resolve_runs_dir(args.runs_dir)
        run_dirs = _find_run_dirs(runs_dir)
        print(f"Found {len(run_dirs)} run(s) in {runs_dir.relative_to(PROJECT_ROOT)}")
        print(f"Method: {method}")
        results: list[dict] = []
        for run_dir in run_dirs:
            print(f"\nProcessing: {run_dir.name}")
            result = process_run(run_dir, args.batch_size, method)
            if result is not None:
                results.append(result)
        print(f"\n{'=' * 50}")
        print(f"Processed {len(results)} run(s)")
        for r in results:
            print(
                f"  {r['run']}: Vendi Score = {r['vendi_score']:.4f}  (n={r['num_hypotheses']})"
            )

        if results:
            mean_vendi = sum(r["vendi_score"] for r in results) / len(results)
            print(f"\nMean Vendi Score across {len(results)} run(s): {mean_vendi:.4f}")

            aggregate = {
                "method": method,
                "model": METHOD_LABELS[method],
                "num_runs": len(results),
                "mean_vendi_score": round(mean_vendi, 4),
                "runs": results,
                "computed_at": datetime.now(timezone.utc).isoformat(),
            }
            agg_path = runs_dir / "vendi_diversity_aggregate.json"
            with open(agg_path, "w", encoding="utf-8") as fh:
                json.dump(aggregate, fh, indent=2, ensure_ascii=False)
            print(f"Aggregate saved to: {agg_path.relative_to(PROJECT_ROOT)}")
    else:
        run_dir = _resolve_run_dir(args.run_dir)
        print(f"Processing: {run_dir.relative_to(PROJECT_ROOT)}")
        print(f"Method: {method}")
        process_run(run_dir, args.batch_size, method)


if __name__ == "__main__":
    main()
