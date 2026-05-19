"""
Correlation analysis between human groundedness annotations and three
automated groundedness reward methods using Qwen3-Reranker-0.6B locally
via transformers.

Three methods evaluated:
  v1 (per-hypothesis): query=hypothesis, documents=papers
                       per-hyp score = mean reranker score across papers
  v2 (per-paper):      query=paper, documents=all hypotheses in group
                       per-hyp score = mean score across papers
  v3 (merged, current):query=merged papers, documents=all hypotheses in group
                       per-hyp score = individual reranker score

Usage:
    python scripts/grpo/groundedness_correlation.py \
        [--model Qwen/Qwen3-Reranker-0.6B] \
        [--device cuda] \
        [--batch-size 8]
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
ANN_CSV = REPO_ROOT / "data/annotation/merged_annotations.csv"
HF_CSV = REPO_ROOT / "data/annotation/merged_annotations_human_friendly.csv"

# ── reranker setup ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_LENGTH = 8192

GROUNDEDNESS_INSTRUCTION = (
    "Given a collection of scientific papers, retrieve hypotheses that are "
    "grounded in or directly supported by the papers' content"
)
GROUNDEDNESS_INSTRUCTION_V1 = (
    "Given a scientific hypothesis, retrieve scientific papers that provide "
    "evidence supporting or grounding the hypothesis"
)


def load_reranker(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    model.to(device).eval()
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    return model, tokenizer, token_true_id, token_false_id, prefix_tokens, suffix_tokens


def _format_pair(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


@torch.no_grad()
def score_pairs(
    pairs: List[Tuple[str, str]],
    instruction: str,
    model,
    tokenizer,
    token_true_id: int,
    token_false_id: int,
    prefix_tokens: List[int],
    suffix_tokens: List[int],
    batch_size: int,
    device: str,
) -> List[float]:
    """Score (query, document) pairs, returning probabilities in [0, 1]."""
    texts = [_format_pair(instruction, q, d) for q, d in pairs]
    all_scores: List[float] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
        )
        for j in range(len(encoded["input_ids"])):
            encoded["input_ids"][j] = (
                prefix_tokens + encoded["input_ids"][j] + suffix_tokens
            )
        inputs = tokenizer.pad(
            encoded, padding=True, return_tensors="pt", max_length=MAX_LENGTH
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs).logits[:, -1, :]
        true_vec = logits[:, token_true_id]
        false_vec = logits[:, token_false_id]
        stacked = torch.stack([false_vec, true_vec], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        probs = log_probs[:, 1].exp().tolist()
        all_scores.extend(probs)

    return all_scores


# ── method implementations ────────────────────────────────────────────────────


def score_v1(
    hypotheses: List[str],
    papers: List[str],
    **kwargs,
) -> List[float]:
    """v1: query=hypothesis, docs=papers. Per-hyp score = mean across papers."""
    per_hyp_scores = []
    for hyp in hypotheses:
        pairs = [(hyp, p) for p in papers]
        scores = score_pairs(pairs, GROUNDEDNESS_INSTRUCTION_V1, **kwargs)
        per_hyp_scores.append(sum(scores) / len(scores))
    return per_hyp_scores


def score_v2(
    hypotheses: List[str],
    papers: List[str],
    **kwargs,
) -> List[float]:
    """v2: query=paper, docs=all hypotheses. Per-hyp score = mean across papers."""
    # scores[paper_idx][hyp_idx]
    scores_by_paper = []
    for paper in papers:
        pairs = [(paper, h) for h in hypotheses]
        scores = score_pairs(pairs, GROUNDEDNESS_INSTRUCTION, **kwargs)
        scores_by_paper.append(scores)

    # transpose to per-hypothesis
    per_hyp_scores = [
        sum(scores_by_paper[p][h] for p in range(len(papers))) / len(papers)
        for h in range(len(hypotheses))
    ]
    return per_hyp_scores


def score_v3(
    hypotheses: List[str],
    papers: List[str],
    **kwargs,
) -> List[float]:
    """v3 (current): query=merged papers, docs=all hypotheses in one call."""
    merged = "\n\n".join(papers)
    pairs = [(merged, h) for h in hypotheses]
    return score_pairs(pairs, GROUNDEDNESS_INSTRUCTION, **kwargs)


# ── data loading ──────────────────────────────────────────────────────────────


def load_data():
    ann = pd.read_csv(ANN_CSV)
    hf = pd.read_csv(HF_CSV)

    paper_cols = [c for c in hf.columns if c.startswith("paper_")]

    # Build query → list of paper texts
    query_to_papers: dict = {}
    for _, row in hf.iterrows():
        query = row["user_query"].strip()
        papers = [
            str(row[c]).strip()
            for c in paper_cols
            if pd.notna(row[c]) and str(row[c]).strip()
        ]
        if papers:
            query_to_papers[query] = papers

    # Filter annotations to queries we have papers for
    ann = ann[ann["query"].str.strip().isin(query_to_papers)].copy()
    ann["query"] = ann["query"].str.strip()
    return ann, query_to_papers


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading reranker: {args.model} on {args.device}")
    model, tokenizer, token_true_id, token_false_id, prefix_tokens, suffix_tokens = (
        load_reranker(args.model, args.device)
    )
    reranker_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        token_true_id=token_true_id,
        token_false_id=token_false_id,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("Loading annotation data...")
    ann, query_to_papers = load_data()
    print(
        f"Loaded {len(ann)} annotated hypotheses across "
        f"{ann['query'].nunique()} queries / "
        f"{ann.groupby(['query', 'run_index']).ngroups} (query, run) groups"
    )

    # Collect per-hypothesis scores from all 3 methods
    results_v1: List[float] = []
    results_v2: List[float] = []
    results_v3: List[float] = []
    human_scores: List[float] = []

    groups = ann.groupby(["query", "run_index"], sort=False)
    total = len(groups)
    for idx, ((query, run_index), group) in enumerate(groups, 1):
        hypotheses = group["hypothesis"].tolist()
        papers = query_to_papers[query]
        print(
            f"[{idx}/{total}] query='{query[:60]}...' "
            f"run={run_index} | {len(hypotheses)} hyps, {len(papers)} papers"
        )

        v1 = score_v1(hypotheses, papers, **reranker_kwargs)
        v2 = score_v2(hypotheses, papers, **reranker_kwargs)
        v3 = score_v3(hypotheses, papers, **reranker_kwargs)

        results_v1.extend(v1)
        results_v2.extend(v2)
        results_v3.extend(v3)
        human_scores.extend(group["groundedness_socre"].tolist())

    # Compute correlations
    print("\n" + "=" * 60)
    print(f"Correlation analysis over {len(human_scores)} hypotheses")
    print("=" * 60)

    header = f"{'Method':<12} {'Pearson r':>10} {'Pearson p':>10} {'Spearman r':>11} {'Spearman p':>11}"
    print(header)
    print("-" * len(header))

    for label, scores in [
        ("v1", results_v1),
        ("v2", results_v2),
        ("v3 (current)", results_v3),
    ]:
        pr, pp = pearsonr(human_scores, scores)
        sr, sp = spearmanr(human_scores, scores)
        print(f"{label:<12} {pr:>10.4f} {pp:>10.4f} {sr:>11.4f} {sp:>11.4f}")

    print("=" * 60)

    # Optionally save detailed results
    out_path = REPO_ROOT / "data/annotation/groundedness_correlation_results.csv"
    result_df = ann[["query", "hypothesis", "run_index", "groundedness_socre"]].copy()
    result_df = result_df[result_df["query"].isin(query_to_papers)].copy()
    result_df["score_v1"] = results_v1
    result_df["score_v2"] = results_v2
    result_df["score_v3"] = results_v3
    result_df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
