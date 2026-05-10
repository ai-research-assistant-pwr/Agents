import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from hypotheses_evaluation import (
    GroundednessJudge,
    RelevancyJudge,
)

MODEL = "gpt-5.4-mini"
SCORE_COLS = [
    "groundedness_socre",
    "relevance_score",
    "informativeness_score",
    "clarity_score",
]
CACHE_PATH = PROJECT_ROOT / "data/evaluate_annotated_cache.json"

# Set to True to read results from cache instead of calling the LLM.
USE_CACHE = False


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with CACHE_PATH.open() as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(hypothesis: str, run: int) -> str:
    return json.dumps({"hypothesis": hypothesis, "run": run}, sort_keys=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    annotated = pd.read_csv(PROJECT_ROOT / "data/annotated_25.csv")
    human_friendly = pd.read_csv(
        PROJECT_ROOT / "data/interesting_rows_human_friendly.csv"
    )
    paper_cols = [c for c in human_friendly.columns if c.startswith("paper_")]
    human_friendly["evidence"] = human_friendly[paper_cols].apply(
        lambda row: "\n\n---\n\n".join(v for v in row if pd.notna(v)), axis=1
    )
    merged = annotated.merge(
        human_friendly[["user_query", "evidence"]],
        left_on="query",
        right_on="user_query",
        how="left",
    )
    return merged.dropna(subset=SCORE_COLS, how="all").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Row evaluation
# ---------------------------------------------------------------------------


def evaluate_row(row, g_judge, r_judge, c_judge, i_judge, run: int, cache: dict):
    hypothesis = row["hypothesis"]
    key = _cache_key(hypothesis, run)

    if USE_CACHE and key in cache:
        return cache[key]

    evidence = row.get("evidence") or ""
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_g = (
            ex.submit(g_judge.judge, hypothesis=hypothesis, evidence=evidence)
            if evidence.strip()
            else None
        )
        f_r = ex.submit(r_judge.judge, hypothesis=hypothesis, query=row["query"])
        f_c = ex.submit(c_judge.judge, hypothesis=hypothesis) if c_judge else None
        f_i = ex.submit(i_judge.judge, hypothesis=hypothesis) if i_judge else None
    g_result = f_g.result() if f_g else None
    r_result = f_r.result()
    c_result = f_c.result() if f_c else None
    i_result = f_i.result() if f_i else None

    rec = {
        "run": run,
        "llm_groundedness": g_result.score if g_result else None,
        "llm_groundedness_reasoning": g_result.reasoning if g_result else None,
        "human_groundedness": row.get("groundedness_socre"),
        "llm_relevancy": r_result.score,
        "llm_relevancy_reasoning": r_result.reasoning,
        "human_relevancy": row.get("relevance_score"),
        "llm_clarity": c_result.score if c_result else None,
        "llm_clarity_reasoning": c_result.reasoning if c_result else None,
        "human_clarity": row.get("clarity_score"),
        "llm_informativeness": i_result.score if i_result else None,
        "llm_informativeness_reasoning": i_result.reasoning if i_result else None,
        "human_informativeness": row.get("informativeness_score"),
        "hypothesis": hypothesis,
    }

    cache[key] = rec
    return rec


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def save_llm_aligned_human_different(
    records1: list, records2: list, metrics: list
) -> None:
    llm1_by_hyp = {m: {rec["hypothesis"]: rec for rec in records1} for m in metrics}
    llm2_by_hyp = {m: {rec["hypothesis"]: rec for rec in records2} for m in metrics}

    for metric in metrics:
        rows = []
        for hyp, rec1 in llm1_by_hyp[metric].items():
            rec2 = llm2_by_hyp[metric].get(hyp)
            if rec2 is None:
                continue
            s1 = rec1[f"llm_{metric}"]
            s2 = rec2[f"llm_{metric}"]
            human = rec1[f"human_{metric}"]
            if pd.isna(s1) or pd.isna(s2) or pd.isna(human):
                continue
            s1, s2, human = int(s1), int(s2), int(human)
            if s1 == s2 and s1 != human:
                rows.append(
                    {
                        "hypothesis": hyp,
                        f"llm1_{metric}": s1,
                        f"llm2_{metric}": s2,
                        f"human_{metric}": human,
                        "llm1_reasoning": rec1[f"llm_{metric}_reasoning"],
                        "llm2_reasoning": rec2[f"llm_{metric}_reasoning"],
                    }
                )
        out_path = PROJECT_ROOT / f"data/llm_aligned_human_different_{metric}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Saved {len(rows)} disagreement cases to {out_path}")


def print_summary(records1: list, records2: list) -> None:
    metrics = ["groundedness", "relevancy", "clarity", "informativeness"]

    print("\n--- Summary ---")
    for metric in metrics:
        # human-llm1
        paired1 = [
            (int(rec[f"llm_{metric}"]), int(rec[f"human_{metric}"]))
            for rec in records1
            if pd.notna(rec[f"llm_{metric}"]) and pd.notna(rec[f"human_{metric}"])
        ]
        # human-llm2
        paired2 = [
            (int(rec[f"llm_{metric}"]), int(rec[f"human_{metric}"]))
            for rec in records2
            if pd.notna(rec[f"llm_{metric}"]) and pd.notna(rec[f"human_{metric}"])
        ]
        # llm1-llm2 (align by hypothesis)
        llm1_by_hyp = {rec["hypothesis"]: rec[f"llm_{metric}"] for rec in records1}
        llm2_by_hyp = {rec["hypothesis"]: rec[f"llm_{metric}"] for rec in records2}
        paired_llm = [
            (int(llm1_by_hyp[h]), int(llm2_by_hyp[h]))
            for h in llm1_by_hyp
            if h in llm2_by_hyp
            and pd.notna(llm1_by_hyp[h])
            and pd.notna(llm2_by_hyp[h])
        ]

        if not paired1 and not paired2 and not paired_llm:
            print(f"{metric}: no paired scores")
            continue

        def kappa_str(pairs):
            if len(pairs) < 2:
                return "n/a"
            a, b = zip(*pairs)
            return f"{cohen_kappa_score(a, b, weights='quadratic'):.3f}"

        def mae_str(pairs):
            if not pairs:
                return "n/a"
            return f"{sum(abs(x - y) for x, y in pairs) / len(pairs):.2f}"

        print(
            f"{metric}: "
            f"n={len(paired1)}  "
            f"kappa(human-llm1)={kappa_str(paired1)}  mae(human-llm1)={mae_str(paired1)}  "
            f"kappa(human-llm2)={kappa_str(paired2)}  mae(human-llm2)={mae_str(paired2)}  "
            f"kappa(llm1-llm2)={kappa_str(paired_llm)}  mae(llm1-llm2)={mae_str(paired_llm)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    client = OpenAI()
    g_judge = GroundednessJudge(client, MODEL, reasoning={"effort": "low"})
    r_judge = RelevancyJudge(client, MODEL, reasoning={"effort": "low"})
    # c_judge = ClarityJudge(client, MODEL, reasoning={"effort": "low"})
    # i_judge = InformativenessJudge(client, MODEL, reasoning={"effort": "low"})

    df = load_data()
    cache = _load_cache()
    records1: list = []
    records2: list = []

    if USE_CACHE:
        print(f"Using cache from {CACHE_PATH}")

    # Submit both runs for every row in parallel
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {}
        for idx, row in df.iterrows():
            for run in (1, 2):
                f = ex.submit(
                    evaluate_row, row, g_judge, r_judge, None, None, run, cache
                )
                futures[f] = run

        for future in as_completed(futures):
            rec = future.result()
            run = rec["run"]
            target = records1 if run == 1 else records2
            target.append(rec)
            print(
                f"[run{run}] g={rec['llm_groundedness']} r={rec['llm_relevancy']} "
                f"c={rec['llm_clarity']} i={rec['llm_informativeness']}  |  "
                f"human: g={rec['human_groundedness']} r={rec['human_relevancy']} "
                f"c={rec['human_clarity']} i={rec['human_informativeness']}"
            )

    if not USE_CACHE:
        _save_cache(cache)
        print(f"Cache saved to {CACHE_PATH}")

    print_summary(records1, records2)
    save_llm_aligned_human_different(records1, records2, ["groundedness", "relevancy"])


if __name__ == "__main__":
    main()
