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
    ClarityJudge,
    GroundednessJudge,
    InformativenessJudge,
    RelevancyJudge,
)

MODEL = "gpt-5.4"
SCORE_COLS = [
    "groundedness_socre",
    "relevance_score",
    "informativeness_score",
    "clarity_score",
]


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


def evaluate_row(row, g_judge, r_judge, c_judge, i_judge):
    hypothesis = row["hypothesis"]
    evidence = row.get("evidence") or ""
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_g = (
            ex.submit(g_judge.judge, hypothesis=hypothesis, evidence=evidence)
            if evidence.strip()
            else None
        )
        f_r = ex.submit(r_judge.judge, hypothesis=hypothesis, query=row["query"])
        f_c = ex.submit(c_judge.judge, hypothesis=hypothesis)
        f_i = ex.submit(i_judge.judge, hypothesis=hypothesis)
    return {
        "llm_groundedness": f_g.result().score if f_g else None,
        "human_groundedness": row.get("groundedness_socre"),
        "llm_relevancy": f_r.result().score,
        "human_relevancy": row.get("relevance_score"),
        "llm_clarity": f_c.result().score,
        "human_clarity": row.get("clarity_score"),
        "llm_informativeness": f_i.result().score,
        "human_informativeness": row.get("informativeness_score"),
    }


def main() -> None:
    client = OpenAI()
    g_judge = GroundednessJudge(client, MODEL)
    r_judge = RelevancyJudge(client, MODEL)
    c_judge = ClarityJudge(client, MODEL)
    i_judge = InformativenessJudge(client, MODEL)

    df = load_data()
    records = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(evaluate_row, row, g_judge, r_judge, c_judge, i_judge): idx
            for idx, row in df.iterrows()
        }
        for future in as_completed(futures):
            rec = future.result()
            records.append(rec)
            print(
                f"g={rec['llm_groundedness']} r={rec['llm_relevancy']} c={rec['llm_clarity']} i={rec['llm_informativeness']}  |  human: g={rec['human_groundedness']} r={rec['human_relevancy']} c={rec['human_clarity']} i={rec['human_informativeness']}"
            )

    print("\n--- Summary ---")
    for metric in ["groundedness", "relevancy", "clarity", "informativeness"]:
        paired = [
            (int(rec[f"llm_{metric}"]), int(rec[f"human_{metric}"]))
            for rec in records
            if pd.notna(rec[f"llm_{metric}"]) and pd.notna(rec[f"human_{metric}"])
        ]
        if not paired:
            print(f"{metric}: no paired scores")
            continue
        llm_vals, human_vals = zip(*paired)
        mae = sum(abs(l - h) for l, h in paired) / len(paired)
        kappa = cohen_kappa_score(llm_vals, human_vals, weights="quadratic")
        print(f"{metric}: n={len(paired)}  mae={mae:.2f}  kappa={kappa:.3f}")


if __name__ == "__main__":
    main()
