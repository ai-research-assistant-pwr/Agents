"""
Pragmatics Evaluation v2 — Experiment 3
=========================================
Two complementary tests for *grounding*: whether the Retriever's language is
anchored to the input context (positive signaling) and whether the Generator
actually uses what the Retriever says (positive listening).

3a. Positive Signaling — t-SNE + k-means + ARI
------------------------------------------------
Hypothesis: if communication is grounded, messages describing *similar* input
contexts should cluster together in signal space.

Changes vs. v1:
  - k-means instead of DBSCAN for meaning-space clustering — eliminates the
    brittle eps hyper-parameter and always produces k non-empty clusters.
  - K is chosen automatically as the elbow of within-cluster inertia (range
    K=2..10), capped at min(10, N//5) to stay well below sample count.
  - ARI is computed between k-means clusters in meaning space and k-means
    clusters in signal space.
  - t-SNE visualisation coloured by meaning-space cluster label.
  - ARI tracked over training windows; final window scatter plotted.

3b. Positive Listening — two complementary probes
--------------------------------------------------
Probe 1 — Reward–length correlation (within-run, no extra inference):
  From ec_stats in each traj log we have mean_channel_tokens and total_reward.
  A positive Spearman ρ(reward, length) would mean the Generator rewards longer
  messages; after length-penalty training we expect the correlation to become
  negative (compressed messages suffice).  Tracked per training window.

Probe 2 — Cross-run generator output divergence (requires --baseline_logs_dir):
  Matches trajectories by prompt across the baseline run (no constraints) and
  the full run (with techniques).  For each matched pair we embed the Generator
  output from both runs and compute cosine distance.  A non-trivial mean
  distance confirms that the communication channel causally shifts the
  Generator's output — i.e., the Generator is *listening* to a different signal.

Output
------
  <output_dir>/
    ├── exp3v2_signaling_ari_over_time.csv
    ├── exp3v2_signaling_tsne_final.png
    ├── exp3v2_listening_reward_length_corr.csv   (Probe 1)
    ├── exp3v2_listening_reward_length_plot.png   (Probe 1)
    ├── exp3v2_listening_crossrun_distances.csv   (Probe 2, optional)
    ├── exp3v2_listening_crossrun_plot.png         (Probe 2, optional)
    └── exp3v2_summary.json

Usage
-----
  EMBED_HOST=<node> EMBED_PORT=8000 \\
  python eval_pragmatics_v2.py \\
      --logs_dir          ./Agents/workflow_logs/<full_run> \\
      --baseline_logs_dir ./Agents/workflow_logs/<baseline_run> \\
      --output_dir        ./Agents/eval_results/<run>/experiment_3 \\
      --window_size       50
"""

import argparse
import asyncio
import json
import glob
import os
from itertools import combinations

import aiohttp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_dist

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
BATCH_SIZE  = 32

ACCENT   = "#1a1a1a"
GREY_MID = "#666666"
GREY_REF = "#aaaaaa"

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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_field(turn: dict, field: str, default=None):
    extra = turn.get("extra", {})
    if field in extra:
        return extra[field]
    return turn.get(field, default)


def _load_sorted_logs(logs_dir: str) -> list:
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
                        timeout=aiohttp.ClientTimeout(total=120),
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


def _choose_k(embs: np.ndarray, k_max: int = 10) -> int:
    """Elbow method: pick K that maximises the second derivative of inertia."""
    k_max = max(2, min(k_max, len(embs) // 5, 10))
    if k_max < 2:
        return 2
    inertias = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        km.fit(embs)
        inertias.append(km.inertia_)
    if len(inertias) < 3:
        return 2
    diffs  = np.diff(inertias)
    diffs2 = np.diff(diffs)
    return int(np.argmax(diffs2) + 2) + 1   # +2 because range starts at 2, +1 for second diff offset


# ---------------------------------------------------------------------------
# 3a — Positive Signaling
# ---------------------------------------------------------------------------

async def run_positive_signaling(
    logs_dir: str,
    output_dir: str,
    embed_host: str,
    embed_port: int,
    window_size: int = 50,
) -> pd.DataFrame:
    """
    Track ARI between meaning-space and signal-space k-means clusters over
    training windows.  Save t-SNE plot of the final window.
    """
    print("\n─── 3a: Positive Signaling (t-SNE + k-means + ARI) ───────")
    logs_dir = _use_train_if_exists(logs_dir)
    parsed_files = _load_sorted_logs(logs_dir)
    if not parsed_files:
        print(f"  No logs found in {logs_dir}")
        return pd.DataFrame()

    print(f"  Found {len(parsed_files)} trajectories.")
    results = []
    last_window_data = None

    for win_start in range(0, len(parsed_files), window_size):
        batch = parsed_files[win_start : win_start + window_size]
        if len(batch) < window_size:
            continue

        window_idx = win_start // window_size + 1
        prompts:  list = []
        messages: list = []

        for _, fpath in batch:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            prompt = data.get("prompt", "")
            for turn in data.get("turns", []):
                if _get_field(turn, "turn_type") == "retriever_message":
                    msg = _get_field(turn, "output_content", "") or ""
                    if msg.strip():
                        messages.append(msg.strip())
                        prompts.append(prompt)

        if len(messages) < 10:
            print(f"  Window {window_idx}: too few messages ({len(messages)}) — skip.")
            continue

        # Cap sample size to keep t-SNE tractable
        if len(messages) > 300:
            idx = np.random.choice(len(messages), 300, replace=False)
            messages = [messages[i] for i in idx]
            prompts  = [prompts[i]  for i in idx]

        print(f"  Window {window_idx}: embedding {len(messages)} messages…")
        unique_prompts = list(set(prompts))
        all_texts      = messages + unique_prompts
        all_embs       = await _get_embeddings(all_texts, embed_host, embed_port, EMBED_MODEL)

        msg_embs_raw    = np.array(all_embs[:len(messages)])
        prompt_embs_raw = np.array(all_embs[len(messages):])

        msg_embs_n    = normalize(msg_embs_raw)
        prompt_embs_n = normalize(prompt_embs_raw)

        # Map each message to its prompt embedding
        prompt_to_emb = {p: prompt_embs_n[i] for i, p in enumerate(unique_prompts)}
        meaning_embs  = np.array([prompt_to_emb[p] for p in prompts])

        # Choose K via elbow on meaning space
        k = _choose_k(meaning_embs)

        km_meaning = KMeans(n_clusters=k, n_init=10, random_state=42).fit(meaning_embs)
        km_signal  = KMeans(n_clusters=k, n_init=10, random_state=42).fit(msg_embs_n)

        ari = adjusted_rand_score(km_meaning.labels_, km_signal.labels_)

        results.append({
            "Window_Index": window_idx,
            "N_Messages":   len(messages),
            "K_Clusters":   k,
            "ARI":          round(ari, 4),
        })
        print(f"    k={k}  ARI={ari:.4f}")

        last_window_data = {
            "meaning_embs": meaning_embs,
            "signal_embs":  msg_embs_n,
            "km_labels":    km_meaning.labels_,
            "window_idx":   window_idx,
            "k":            k,
            "ari":          ari,
        }

    if not results:
        print("  No results produced for signaling analysis.")
        return pd.DataFrame()

    df_sig = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp3v2_signaling_ari_over_time.csv")
    df_sig.to_csv(csv_path, index=False)
    print(f"  ARI CSV saved → {csv_path}")

    # ── ARI over-time plot ────────────────────────────────────────────────
    plt.rcParams.update(_PLT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_sig["Window_Index"], df_sig["ARI"], color=ACCENT, lw=1.2)
    ax.fill_between(df_sig["Window_Index"], df_sig["ARI"], alpha=0.06, color=ACCENT)
    ax.axhline(0.0, lw=0.6, color=GREY_REF, ls="-.")
    ax.set_title(
        "Experiment 3a — Positive signaling: ARI over training\n"
        "(meaning-space vs. signal-space k-means clustering)",
        pad=8,
    )
    ax.set_xlabel("Training window")
    ax.set_ylabel("Adjusted Rand Index")
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.grid(axis="y", lw=0.4, color=GREY_REF, ls=":")
    fig.tight_layout()
    ari_plot = os.path.join(output_dir, "exp3v2_signaling_ari_over_time.png")
    plt.savefig(ari_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ARI plot saved → {ari_plot}")

    # ── t-SNE of final window ─────────────────────────────────────────────
    if last_window_data is not None:
        _plot_tsne(last_window_data, output_dir)

    return df_sig


def _plot_tsne(data: dict, output_dir: str) -> None:
    """2-D t-SNE of signal embeddings, coloured by meaning-space cluster."""
    sig_embs   = data["signal_embs"]
    labels     = data["km_labels"]
    k          = data["k"]
    window_idx = data["window_idx"]
    ari        = data["ari"]

    perplexity = min(30, len(sig_embs) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(sig_embs)

    cmap    = plt.cm.get_cmap("tab10", k)
    colors  = [cmap(l) for l in labels]
    patches = [mpatches.Patch(color=cmap(i), label=f"Cluster {i}") for i in range(k)]

    plt.rcParams.update(_PLT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=12, alpha=0.7, linewidths=0)
    ax.set_title(
        f"Experiment 3a — t-SNE of Retriever messages (window {window_idx})\n"
        f"Coloured by meaning-space cluster  ·  ARI = {ari:.3f}",
        pad=8,
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(handles=patches, frameon=False, fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    tsne_path = os.path.join(output_dir, "exp3v2_signaling_tsne_final.png")
    plt.savefig(tsne_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  t-SNE plot saved → {tsne_path}")


# ---------------------------------------------------------------------------
# 3b — Positive Listening: Probe 1 (reward–length correlation)
# ---------------------------------------------------------------------------

def run_reward_length_correlation(
    logs_dir: str,
    output_dir: str,
    window_size: int = 50,
) -> pd.DataFrame:
    """
    Within-run analysis: Spearman ρ(reward, mean_channel_tokens) per window.

    A positive ρ in the baseline run would indicate the Generator rewards
    richer (longer) messages.  After length-penalty training we expect ρ to
    diminish or become negative because compressed messages become sufficient.
    """
    print("\n─── 3b Probe 1: Reward–length Spearman correlation ─────────")
    logs_dir = _use_train_if_exists(logs_dir)
    parsed_files = _load_sorted_logs(logs_dir)
    if not parsed_files:
        print(f"  No logs found in {logs_dir}")
        return pd.DataFrame()

    print(f"  Found {len(parsed_files)} trajectories.")
    results = []

    for win_start in range(0, len(parsed_files), window_size):
        batch = parsed_files[win_start : win_start + window_size]
        if len(batch) < window_size:
            continue

        window_idx = win_start // window_size + 1
        rewards: list = []
        lengths: list = []

        for _, fpath in batch:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)

            reward = data.get("total_reward", 0.0)
            ec     = data.get("ec_stats", {})

            # mean_channel_tokens from ec_stats (logged by workflow)
            mean_len = ec.get("mean_channel_tokens")
            if mean_len is None:
                # Fallback: compute from turns
                lens = []
                for turn in data.get("turns", []):
                    if _get_field(turn, "turn_type") == "retriever_message":
                        n = _get_field(turn, "n_channel_tokens") or 0
                        if n:
                            lens.append(n)
                mean_len = float(np.mean(lens)) if lens else None

            if mean_len is not None:
                rewards.append(reward)
                lengths.append(mean_len)

        if len(rewards) < 5:
            print(f"  Window {window_idx}: insufficient data — skip.")
            continue

        rho, pval = spearmanr(rewards, lengths)
        results.append({
            "Window_Index":  window_idx,
            "N_Trajectories": len(rewards),
            "Spearman_Rho":  round(rho,  4),
            "P_Value":       round(pval, 6),
            "Mean_Reward":   round(float(np.mean(rewards)), 4),
            "Mean_Length":   round(float(np.mean(lengths)), 2),
        })
        print(f"  Window {window_idx}: ρ={rho:.4f}  p={pval:.4f}  "
              f"mean_len={np.mean(lengths):.1f}  mean_reward={np.mean(rewards):.3f}")

    if not results:
        print("  No results.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp3v2_listening_reward_length_corr.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved → {csv_path}")

    # Plot
    plt.rcParams.update(_PLT_STYLE)
    fig, (ax_r, ax_l) = plt.subplots(
        2, 1, figsize=(7, 5), sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    sig_mask = df["P_Value"] < 0.05

    ax_r.plot(df["Window_Index"], df["Spearman_Rho"], color=ACCENT, lw=1.2)
    ax_r.scatter(
        df["Window_Index"][sig_mask], df["Spearman_Rho"][sig_mask],
        color=ACCENT, s=20, zorder=5, label="p < 0.05",
    )
    ax_r.axhline(0.0, lw=0.6, color=GREY_REF, ls="-.")
    ax_r.set_ylabel("Spearman ρ\n(reward vs. length)")
    ax_r.yaxis.set_label_coords(-0.12, 0.5)
    ax_r.grid(axis="y", lw=0.4, color=GREY_REF, ls=":")
    ax_r.legend(frameon=False, fontsize=8)
    ax_r.set_title(
        "Experiment 3b — Positive listening: reward–length correlation\n"
        "Upper: Spearman ρ  ·  Lower: mean message length",
        pad=8,
    )

    ax_l.plot(df["Window_Index"], df["Mean_Length"], color=ACCENT, lw=1.2)
    ax_l.fill_between(df["Window_Index"], df["Mean_Length"], alpha=0.06, color=ACCENT)
    ax_l.set_ylabel("Mean channel tokens")
    ax_l.yaxis.set_label_coords(-0.12, 0.5)
    ax_l.set_xlabel("Training window")
    ax_l.grid(axis="y", lw=0.4, color=GREY_REF, ls=":")

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "exp3v2_listening_reward_length_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {plot_path}")

    return df


# ---------------------------------------------------------------------------
# 3b — Positive Listening: Probe 2 (cross-run generator output divergence)
# ---------------------------------------------------------------------------

async def run_crossrun_generator_divergence(
    full_logs_dir: str,
    baseline_logs_dir: str,
    output_dir: str,
    embed_host: str,
    embed_port: int,
) -> pd.DataFrame:
    """
    Match trajectories by prompt between baseline and full run.
    Embed Generator outputs from both runs and compute cosine distance.
    A non-trivial mean distance confirms the Generator is causally responsive
    to the communication channel (positive listening).
    """
    print("\n─── 3b Probe 2: Cross-run generator output divergence ───────")

    def _load_prompt_map(directory: str) -> dict:
        """Return {prompt: generator_output} for every traj file."""
        directory = _use_train_if_exists(directory)
        mapping = {}
        for _, fpath in _load_sorted_logs(directory):
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            prompt = data.get("prompt", "")
            if not prompt:
                continue
            gen_out = None
            for turn in data.get("turns", []):
                if _get_field(turn, "turn_type") == "generator_generate":
                    gen_out = _get_field(turn, "output", "") or turn.get("output", "")
                    break
            if gen_out and prompt not in mapping:
                mapping[prompt] = gen_out
        return mapping

    print("  Loading full run…")
    full_map = _load_prompt_map(full_logs_dir)
    print("  Loading baseline run…")
    base_map = _load_prompt_map(baseline_logs_dir)

    common = list(set(full_map) & set(base_map))
    if not common:
        print("  No matching prompts found between runs — skipping Probe 2.")
        return pd.DataFrame()

    print(f"  Matched {len(common)} prompts across runs.")

    full_outputs = [full_map[p] for p in common]
    base_outputs = [base_map[p] for p in common]

    all_texts = full_outputs + base_outputs
    all_embs  = await _get_embeddings(all_texts, embed_host, embed_port, EMBED_MODEL)
    full_embs = np.array(all_embs[:len(common)])
    base_embs = np.array(all_embs[len(common):])

    dists = [cosine_dist(full_embs[i], base_embs[i]) for i in range(len(common))]
    mean_dist = float(np.mean(dists))
    std_dist  = float(np.std(dists))
    print(f"  Mean cosine distance (generator outputs): {mean_dist:.4f} ± {std_dist:.4f}")

    df = pd.DataFrame({
        "Prompt_Index":      range(len(common)),
        "Generator_Cos_Dist": [round(d, 4) for d in dists],
    })
    csv_path = os.path.join(output_dir, "exp3v2_listening_crossrun_distances.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved → {csv_path}")

    # Distribution plot
    plt.rcParams.update(_PLT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dists, bins=30, color=ACCENT, alpha=0.75, linewidth=0)
    ax.axvline(mean_dist, color=ACCENT, ls="--", lw=1.0,
               label=f"Mean = {mean_dist:.4f}")
    ax.set_title(
        "Experiment 3b — Positive listening: generator output divergence\n"
        "(cross-run cosine distance: full run vs. baseline)",
        pad=8,
    )
    ax.set_xlabel("Cosine distance (generator output: full vs. baseline)")
    ax.set_ylabel("Count")
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.legend(frameon=False)
    ax.grid(axis="y", lw=0.4, color=GREY_REF, ls=":")
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "exp3v2_listening_crossrun_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {plot_path}")

    return df


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _write_summary(
    output_dir: str,
    sig_df: pd.DataFrame,
    rl_df: pd.DataFrame,
    cr_df: pd.DataFrame,
) -> None:
    summary: dict = {}

    if not sig_df.empty:
        summary["positive_signaling"] = {
            "final_ari":  float(sig_df["ARI"].iloc[-1]),
            "max_ari":    float(sig_df["ARI"].max()),
            "mean_ari":   round(float(sig_df["ARI"].mean()), 4),
            "n_windows":  int(len(sig_df)),
        }

    if not rl_df.empty:
        summary["positive_listening_probe1"] = {
            "final_rho":        float(rl_df["Spearman_Rho"].iloc[-1]),
            "initial_rho":      float(rl_df["Spearman_Rho"].iloc[0]),
            "significant_windows": int((rl_df["P_Value"] < 0.05).sum()),
            "n_windows":        int(len(rl_df)),
        }

    if not cr_df.empty and "Generator_Cos_Dist" in cr_df.columns:
        summary["positive_listening_probe2"] = {
            "mean_crossrun_dist": round(float(cr_df["Generator_Cos_Dist"].mean()), 4),
            "std_crossrun_dist":  round(float(cr_df["Generator_Cos_Dist"].std()),  4),
            "n_pairs":            int(len(cr_df)),
        }

    path = os.path.join(output_dir, "exp3v2_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {path}")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_pragmatics_v2(
    logs_dir: str,
    output_dir: str,
    window_size: int = 50,
    embed_host: str = "localhost",
    embed_port: int = 8000,
    baseline_logs_dir: str | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Experiment 3v2 — Pragmatics / Grounding ===")

    # 3a
    sig_df = await run_positive_signaling(
        logs_dir, output_dir, embed_host, embed_port, window_size,
    )

    # 3b Probe 1 — no embeddings needed
    rl_df = run_reward_length_correlation(logs_dir, output_dir, window_size)

    # 3b Probe 2 — cross-run (optional)
    cr_df = pd.DataFrame()
    if baseline_logs_dir and os.path.isdir(baseline_logs_dir):
        cr_df = await run_crossrun_generator_divergence(
            full_logs_dir     = logs_dir,
            baseline_logs_dir = baseline_logs_dir,
            output_dir        = output_dir,
            embed_host        = embed_host,
            embed_port        = embed_port,
        )
    elif baseline_logs_dir:
        print(f"  WARNING: baseline_logs_dir not found: {baseline_logs_dir}")

    _write_summary(output_dir, sig_df, rl_df, cr_df)
    print(f"\nExperiment 3v2 complete. Results in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 3v2 — Pragmatics: Positive Signaling & Positive Listening"
    )
    parser.add_argument("--logs_dir",          default="./Agents/workflow_logs")
    parser.add_argument("--baseline_logs_dir", default=None,
                        help="Baseline run directory for cross-run CIC (Probe 2)")
    parser.add_argument("--output_dir",        default="./Agents/eval_results/experiment_3")
    parser.add_argument("--window_size",       type=int, default=50)
    parser.add_argument("--embed_host",        default=os.getenv("EMBED_HOST", "localhost"))
    parser.add_argument("--embed_port",        type=int,
                        default=int(os.getenv("EMBED_PORT", 8000)))
    args = parser.parse_args()

    asyncio.run(run_pragmatics_v2(
        logs_dir          = args.logs_dir,
        output_dir        = args.output_dir,
        window_size       = args.window_size,
        embed_host        = args.embed_host,
        embed_port        = args.embed_port,
        baseline_logs_dir = args.baseline_logs_dir,
    ))