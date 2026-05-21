"""
Semantics Evaluation v2 — Experiment 2
========================================
Replaces TopSim (which loses resolution on narrow scientific domains) with two
complementary measures of how the Retriever's emergent language evolves
structurally over training.

Measure A — Intra-window signal dispersion
-------------------------------------------
Tracks mean pairwise cosine *distance* between Retriever messages within each
training window.  A decrease signals convergence toward a shared vocabulary /
compressed protocol; an increase signals diverging, context-specific language.
Reported alongside standard deviation to distinguish systematic drift from
noise.

Measure B — Signal–prompt alignment (grounding index)
------------------------------------------------------
For each trajectory, computes cosine similarity between the Retriever's
message embedding and the embedding of the original user prompt.
High and rising alignment → the Retriever is grounding its synthesis in the
query rather than drifting into generic filler.

Both measures are computed for each training window and plotted over time.
When a --baseline_logs_dir is supplied, the same metrics are computed for the
baseline run and overlaid on the same axes for direct comparison.

Data contract
-------------
Reads traj_*.json written by scientific_workflow.py (debug=true).
Each file must contain:
  - "prompt"  : str
  - "turns"   : list of turn dicts where turn_type == "retriever_message"
                has "output_content" (or "extra.output_content") with the
                clean Retriever message text.

Output
------
  <output_dir>/
    ├── exp2v2_signal_structure.csv        windowed metrics (both runs)
    ├── exp2v2_dispersion_over_time.png    Measure A plot
    └── exp2v2_grounding_over_time.png     Measure B plot

Usage
-----
  EMBED_HOST=<node> EMBED_PORT=8000 \\
  python eval_semantics_v2.py \\
      --logs_dir          ./Agents/workflow_logs/<full_run> \\
      --baseline_logs_dir ./Agents/workflow_logs/<baseline_run> \\
      --output_dir        ./Agents/eval_results/<run>/experiment_2 \\
      --window_size       50
"""

import os
import json
import glob
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_HOST  = os.getenv("EMBED_HOST",  "localhost")
EMBED_PORT  = int(os.getenv("EMBED_PORT",  "8000"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
BATCH_SIZE  = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_field(turn: dict, field: str, default=None):
    """Look up a field in turn dict, checking 'extra' sub-dict first."""
    extra = turn.get("extra", {})
    if field in extra:
        return extra[field]
    return turn.get(field, default)


def _load_sorted_logs(logs_dir: str) -> list:
    """Return (timestamp, filepath) pairs sorted by timestamp."""
    files = glob.glob(os.path.join(logs_dir, "traj_*.json"))
    parsed = []
    for f in files:
        try:
            ts = int(os.path.basename(f).split("_")[1])
            parsed.append((ts, f))
        except (IndexError, ValueError):
            continue
    parsed.sort(key=lambda x: x[0])
    return parsed


def _use_train_if_exists(directory: str) -> str:
    train_dir = os.path.join(directory, "train")
    if os.path.isdir(train_dir):
        print(f"  Auto-detected 'train' subfolder: {train_dir}")
        return train_dir
    return directory


async def _get_embeddings(texts: list, host: str, port: int, model: str) -> list:
    """Fetch embeddings from vLLM in batches."""
    if not texts:
        return []
    all_embs = []
    url = f"http://{host}:{port}/v1/embeddings"
    async with aiohttp.ClientSession() as session:
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            payload = {"model": model, "input": batch}
            for attempt in range(3):
                try:
                    async with session.post(
                        url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                    ordered = sorted(data["data"], key=lambda x: x["index"])
                    all_embs.extend(item["embedding"] for item in ordered)
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise RuntimeError(f"Embedding server error: {exc}") from exc
                    await asyncio.sleep(2 ** attempt)
    return all_embs


def _mean_pairwise_cosine_distance(emb_matrix: np.ndarray) -> tuple:
    """Return (mean, std) pairwise cosine distance for a set of embeddings."""
    n = len(emb_matrix)
    if n < 2:
        return 0.0, 0.0
    dists = [
        cosine_dist(emb_matrix[i], emb_matrix[j])
        for i, j in combinations(range(n), 2)
    ]
    return float(np.mean(dists)), float(np.std(dists))


# ---------------------------------------------------------------------------
# Per-run windowed analysis
# ---------------------------------------------------------------------------

async def _analyse_run(
    logs_dir: str,
    window_size: int,
    run_label: str,
    embed_host: str,
    embed_port: int,
    embed_model: str,
) -> pd.DataFrame:
    """
    Compute windowed Measure A (dispersion) and Measure B (grounding) for a
    single run directory.  Returns a DataFrame with one row per complete window.
    """
    logs_dir = _use_train_if_exists(logs_dir)
    parsed_files = _load_sorted_logs(logs_dir)

    if not parsed_files:
        print(f"  [{run_label}] No traj_*.json files found in {logs_dir}")
        return pd.DataFrame()

    print(f"  [{run_label}] Found {len(parsed_files)} trajectories.")

    results = []

    for win_start in range(0, len(parsed_files), window_size):
        batch = parsed_files[win_start : win_start + window_size]
        if len(batch) < window_size:
            print(f"  [{run_label}] Window {win_start // window_size + 1}: "
                  f"incomplete ({len(batch)}/{window_size}) — skipped.")
            continue

        window_idx = win_start // window_size + 1
        prompts: list = []
        messages: list = []
        rewards: list = []

        for _, fpath in batch:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)

            prompt = data.get("prompt", "")
            reward = data.get("total_reward", 0.0)
            rewards.append(reward)

            for turn in data.get("turns", []):
                if _get_field(turn, "turn_type") == "retriever_message":
                    msg = _get_field(turn, "output_content", "") or ""
                    if msg.strip():
                        messages.append(msg.strip())
                        prompts.append(prompt)

        if len(messages) < 4:
            print(f"  [{run_label}] Window {window_idx}: "
                  f"too few messages ({len(messages)}) — skipped.")
            continue

        print(f"  [{run_label}] Window {window_idx}: "
              f"embedding {len(messages)} messages + {len(set(prompts))} unique prompts…")

        # Embed messages and prompts together in one call
        unique_prompts = list(set(prompts))
        all_texts = messages + unique_prompts
        all_embs  = await _get_embeddings(all_texts, embed_host, embed_port, embed_model)

        msg_embs    = np.array(all_embs[:len(messages)])
        prompt_embs_raw = np.array(all_embs[len(messages):])

        # Normalise for cosine arithmetic
        msg_embs_n    = normalize(msg_embs)
        prompt_embs_n = normalize(prompt_embs_raw)

        # Build prompt → embedding lookup
        prompt_to_emb = {p: prompt_embs_n[i] for i, p in enumerate(unique_prompts)}

        # ── Measure A: mean pairwise cosine distance between messages ─────────
        # Sample up to 200 messages to keep quadratic cost manageable
        sample_idx = (
            np.random.choice(len(msg_embs_n), 200, replace=False)
            if len(msg_embs_n) > 200
            else np.arange(len(msg_embs_n))
        )
        sampled = msg_embs_n[sample_idx]
        mean_dist, std_dist = _mean_pairwise_cosine_distance(sampled)

        # ── Measure B: cosine similarity between message and its prompt ───────
        sims = []
        for i, (msg_emb, ptext) in enumerate(zip(msg_embs_n, prompts)):
            p_emb = prompt_to_emb.get(ptext)
            if p_emb is not None:
                sim = float(np.dot(msg_emb, p_emb))   # vectors are L2-normalised
                sims.append(sim)

        mean_grounding = float(np.mean(sims)) if sims else float("nan")
        std_grounding  = float(np.std(sims))  if sims else float("nan")
        mean_reward    = float(np.mean(rewards))

        results.append({
            "Run":               run_label,
            "Window_Index":      window_idx,
            "N_Messages":        len(messages),
            "Mean_Dispersion":   round(mean_dist,       4),
            "Std_Dispersion":    round(std_dist,        4),
            "Mean_Grounding":    round(mean_grounding,  4),
            "Std_Grounding":     round(std_grounding,   4),
            "Mean_Reward":       round(mean_reward,     4),
        })
        print(f"    dispersion={mean_dist:.3f}  grounding={mean_grounding:.3f}  "
              f"reward={mean_reward:.3f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PLT_STYLE = {
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.6,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
}

_COLORS = {
    "full":     "#1a1a1a",
    "baseline": "#888888",
}


def _plot_metric(
    df: pd.DataFrame,
    y_col: str,
    y_err_col: str | None,
    title: str,
    ylabel: str,
    output_path: str,
) -> None:
    plt.rcParams.update(_PLT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 4))

    for run_label, group in df.groupby("Run"):
        color  = _COLORS.get(run_label, "#444444")
        ls     = "-" if run_label != "baseline" else "--"
        x      = group["Window_Index"]
        y      = group[y_col]

        ax.plot(x, y, color=color, lw=1.2, ls=ls, label=run_label)
        ax.fill_between(x, y, alpha=0.06, color=color)

        if y_err_col and y_err_col in group.columns:
            err = group[y_err_col]
            ax.fill_between(x, y - err, y + err, alpha=0.08, color=color)

    ax.set_title(title, pad=8)
    ax.set_xlabel("Training window")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.grid(axis="y", lw=0.4, color="#aaaaaa", ls=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path}")


def _plot_dual(
    df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Combined 2-panel plot: dispersion + grounding, both runs on same axes."""
    plt.rcParams.update(_PLT_STYLE)
    fig, (ax_d, ax_g) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    for run_label, group in df.groupby("Run"):
        color = _COLORS.get(run_label, "#444444")
        ls    = "-" if run_label != "baseline" else "--"
        x     = group["Window_Index"]

        ax_d.plot(x, group["Mean_Dispersion"], color=color, lw=1.2, ls=ls,
                  label=run_label)
        ax_d.fill_between(x, group["Mean_Dispersion"], alpha=0.06, color=color)

        ax_g.plot(x, group["Mean_Grounding"], color=color, lw=1.2, ls=ls,
                  label=run_label)
        ax_g.fill_between(x, group["Mean_Grounding"], alpha=0.06, color=color)

    ax_d.set_ylabel("Mean pairwise\ncosine distance")
    ax_d.yaxis.set_label_coords(-0.1, 0.5)
    ax_d.grid(axis="y", lw=0.4, color="#aaaaaa", ls=":")
    ax_d.legend(frameon=False)
    ax_d.set_title(
        "Experiment 2 — Signal structure evolution over training\n"
        "Upper: intra-window dispersion  ·  Lower: signal–prompt alignment",
        pad=8,
    )

    ax_g.set_ylabel("Mean cosine similarity\n(message ↔ prompt)")
    ax_g.yaxis.set_label_coords(-0.1, 0.5)
    ax_g.set_xlabel("Training window")
    ax_g.grid(axis="y", lw=0.4, color="#aaaaaa", ls=":")

    fig.tight_layout()
    out = os.path.join(output_dir, "exp2v2_signal_structure_combined.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_semantics_v2(
    logs_dir: str,
    output_dir: str,
    window_size: int = 50,
    baseline_logs_dir: str | None = None,
    embed_host: str = EMBED_HOST,
    embed_port: int = EMBED_PORT,
    embed_model: str = EMBED_MODEL,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Experiment 2v2 — Signal Structure Analysis ===")

    # Analyse full (experimental) run
    df_full = await _analyse_run(
        logs_dir, window_size, "full",
        embed_host, embed_port, embed_model,
    )

    all_dfs = [df_full] if not df_full.empty else []

    # Optionally analyse baseline run
    if baseline_logs_dir and os.path.isdir(baseline_logs_dir):
        print(f"\n  Baseline run: {baseline_logs_dir}")
        df_base = await _analyse_run(
            baseline_logs_dir, window_size, "baseline",
            embed_host, embed_port, embed_model,
        )
        if not df_base.empty:
            all_dfs.append(df_base)
    elif baseline_logs_dir:
        print(f"  WARNING: baseline_logs_dir not found: {baseline_logs_dir}")

    if not all_dfs:
        print("No data produced — check log directories.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    csv_path = os.path.join(output_dir, "exp2v2_signal_structure.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved → {csv_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    _plot_dual(df, output_dir)

    _plot_metric(
        df,
        y_col="Mean_Dispersion",
        y_err_col="Std_Dispersion",
        title=(
            "Experiment 2 — Intra-window signal dispersion over training\n"
            "(mean pairwise cosine distance between Retriever messages)"
        ),
        ylabel="Mean cosine distance",
        output_path=os.path.join(output_dir, "exp2v2_dispersion_over_time.png"),
    )

    _plot_metric(
        df,
        y_col="Mean_Grounding",
        y_err_col="Std_Grounding",
        title=(
            "Experiment 2 — Signal–prompt alignment (grounding index) over training\n"
            "(mean cosine similarity: Retriever message ↔ user prompt)"
        ),
        ylabel="Cosine similarity",
        output_path=os.path.join(output_dir, "exp2v2_grounding_over_time.png"),
    )

    # Print summary
    print("\n  === Summary (final window) ===")
    for run_label, group in df.groupby("Run"):
        last = group.iloc[-1]
        print(f"  [{run_label}]  dispersion={last['Mean_Dispersion']:.4f}  "
              f"grounding={last['Mean_Grounding']:.4f}  "
              f"reward={last['Mean_Reward']:.4f}")

    print(f"\nExperiment 2v2 complete. Results in: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 2v2 — Semantics: signal structure evolution"
    )
    parser.add_argument("--logs_dir",          default="./Agents/workflow_logs")
    parser.add_argument("--baseline_logs_dir", default=None,
                        help="Optional baseline run for comparison")
    parser.add_argument("--output_dir",        default="./Agents/eval_results/experiment_2")
    parser.add_argument("--window_size",       type=int, default=50)
    parser.add_argument("--embed_host",        default=EMBED_HOST)
    parser.add_argument("--embed_port",        type=int, default=EMBED_PORT)
    parser.add_argument("--embed_model",       default=EMBED_MODEL)
    args = parser.parse_args()

    asyncio.run(run_semantics_v2(
        logs_dir           = args.logs_dir,
        output_dir         = args.output_dir,
        window_size        = args.window_size,
        baseline_logs_dir  = args.baseline_logs_dir,
        embed_host         = args.embed_host,
        embed_port         = args.embed_port,
        embed_model        = args.embed_model,
    ))