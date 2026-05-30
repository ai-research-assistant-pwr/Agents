"""Aggregate numeric scores from all 06_evaluation.json files in a runs directory.

Usage:
    python scripts/hypotheses_evaluation/aggregate_evaluations.py --runs-dir outputs/none_1
    python scripts/hypotheses_evaluation/aggregate_evaluations.py --runs-dir outputs/none_1 --output results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_FILENAME = "06_evaluation.json"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate numeric scores from evaluation results."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        required=True,
        help="Directory containing run subfolders with 06_evaluation.json files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <runs-dir>/aggregated_evaluations.json).",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    args = parse_args()

    runs_dir = resolve_path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: runs directory not found: {runs_dir}")
        sys.exit(1)

    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"ERROR: no subdirectories found in {runs_dir}")
        sys.exit(1)

    aggregated = []

    for run_dir in run_dirs:
        eval_path = run_dir / EVALUATION_FILENAME
        if not eval_path.exists():
            logger.warning("Missing %s in %s, skipping", EVALUATION_FILENAME, run_dir.name)
            continue

        with open(eval_path, encoding="utf-8") as f:
            data = json.load(f)

        groundedness = [h["groundedness"]["score"] for h in data["hypotheses"]]
        relevancy = [h["relevancy"]["score"] for h in data["hypotheses"]]
        clarity = [h["clarity"]["score"] for h in data["hypotheses"]]

        aggregated.append(
            {
                "run": run_dir.name,
                "groundedness": groundedness,
                "relevancy": relevancy,
                "clarity": clarity,
                "diversity": data["summary"]["diversity"],
                "novelty": data["summary"]["novelty"],
            }
        )

    output_path = resolve_path(args.output) if args.output else runs_dir / "aggregated_evaluations.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    print(f"Aggregated {len(aggregated)} run(s) to {output_path}")


if __name__ == "__main__":
    main()
