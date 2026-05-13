"""
Pragmatics Evaluation — Experiment 3
======================================
Two complementary tests for *grounding*: the degree to which the Retriever's
emergent language is semantically anchored to the input context and causally
influences the Generator's output.

3a. Positive Signaling — t-SNE + DBSCAN + ARI
----------------------------------------------
Hypothesis: if the communication protocol is grounded, messages that describe
*similar* input contexts should cluster together in signal space.

Method:
  1. Embed the meaning space (prompt + initial and dynamically retrieved abstracts) 
     and the signal space (Retriever output_content from all 'retriever_message' turns) 
     for every trajectory.
  2. Cluster the meaning-space embeddings with DBSCAN → ground-truth clusters.
  3. Cluster the signal-space embeddings with DBSCAN → predicted clusters.
  4. Compute ARI (Adjusted Rand Index) between the two clusterings.
     ARI → 1.0  : signal clusters mirror meaning clusters  (grounded)
     ARI → 0.0  : random agreement                         (ungrounded)
     ARI < 0    : worse than chance
  5. Visualise with t-SNE (2-D), coloured by meaning-space cluster label.

3b. Positive Listening — Proxy CIC
-----------------------------------
Hypothesis: if the Generator *listens* to the Retriever, replacing the clean
signal with a noisy version should measurably shift the Generator's output.

Method (offline, no extra inference required):
  - We exploit runs where `apply_channel_noise=True` so that each trajectory
    debug log contains both `output_content` (clean) and `noisy_output_content`
    (noisy) for 'retriever_message' turns.
  - We embed Generator outputs (from 'generator_generate' turn) paired with 
    their respective (clean / noisy) inputs and compute the cosine distance 
    between the two Generator outputs. A higher distance indicates the Generator 
    is sensitive to signal perturbations, i.e. it *listens*.

  Proxy CIC = mean cosine distance(
                  embed(gen_out | clean_signal),
                  embed(gen_out | noisy_signal)
              )

  Because we do not have paired (clean / noisy) generator outputs in a single
  log file, we instead compare trajectories across two *matched* run
  directories:
    - `run_baseline_free/`  (no noise)
    - `run_exp_full/`        (with noise)
  and look at the distribution of Generator output embeddings.

  As a *within-run* fallback (single directory), we measure the cosine
  distance between the embedded clean signal and the embedded noisy signal
  for each turn (signal sensitivity, not output sensitivity), which still
  gives directional evidence.

Output
------
  eval_results/experiment_3/
    ├── exp3_signaling_ari_over_time.csv
    ├── exp3_signaling_tsne_final.png
    ├── exp3_listening_cic_over_time.csv
    ├── exp3_listening_cic_distribution.png
    └── exp3_summary.json

Usage
-----
  # Both analyses (requires noisy logs for CIC):
  python eval_pragmatics.py \
      --logs_dir        ./Agents/workflow_logs \
      --output_dir      ./Agents/eval_results/experiment_3 \
      --window_size     50 \
      --embed_host      localhost \
      --embed_port      8000

  # Cross-run CIC (optional, compares two directories):
  python eval_pragmatics.py \
      --logs_dir        ./Agents/workflow_logs/baseline_free \
      --noisy_logs_dir  ./Agents/workflow_logs/exp_full \
      --output_dir      ./Agents/eval_results/experiment_3 \
      --cross_run_cic
"""

import argparse
import asyncio
import json
import glob
import os
import warnings
from itertools import combinations

import aiohttp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine as cosine_dist

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")

# DBSCAN parameters — tweak if clusters are too coarse/fine
DBSCAN_EPS = 0.35       # cosine distance threshold (embeddings are L2-normalised → euclidean ≈ cosine)
DBSCAN_MIN_SAMPLES = 3  # minimum points per cluster

# Plot style — shared across all figures in this module
ACCENT   = '#1a1a1a'   # near-black for primary series / bars
GREY_MID = '#666666'   # secondary series
GREY_REF = '#aaaaaa'   # reference lines and grid


# ---------------------------------------------------------------------------
# Helper: safe access to turn fields (extra vs. top-level)
# ---------------------------------------------------------------------------

def _get_turn_field(turn: dict, field: str, default=None):
    """
    Retrieves a field from a turn dictionary, looking first in 'extra'
    (where debug entries store step-specific metadata) and then in the
    top-level turn dict.  This avoids brittle assumptions about data layout.
    """
    extra = turn.get("extra", {})
    if field in extra:
        return extra[field]
    return turn.get(field, default)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

async def _get_embeddings_batch(
    texts: list[str],
    host: str,
    port: int,
    model: str,
    batch_size: int = 48,
) -> list[list[float]]:
    """Fetch embeddings from vLLM in batches to avoid request-size limits."""
    all_embeddings: list[list[float]] = []
    url = f"http://{host}:{port}/v1/embeddings"

    async with aiohttp.ClientSession() as session:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {"model": model, "input": batch}

            for attempt in range(3):
                try:
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                    ordered = sorted(data["data"], key=lambda x: x["index"])
                    all_embeddings.extend(item["embedding"] for item in ordered)
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise RuntimeError(
                            f"Embedding server unreachable after 3 attempts: {exc}"
                        ) from exc
                    await asyncio.sleep(2 ** attempt)

    return all_embeddings


# ---------------------------------------------------------------------------
# Log loading helpers
# ---------------------------------------------------------------------------

def _load_sorted_logs(logs_dir: str) -> list[tuple[int, str]]:
    """Return (timestamp, filepath) pairs sorted by timestamp."""
    files = glob.glob(os.path.join(logs_dir, "traj_*.json"))
    parsed: list[tuple[int, str]] = []
    for f in files:
        try:
            ts = int(os.path.basename(f).split("_")[1])
            parsed.append((ts, f))
        except (IndexError, ValueError):
            continue
    parsed.sort(key=lambda x: x[0])
    return parsed


def _extract_meaning(data: dict) -> str:
    """Build meaning-space string: prompt + initial papers + dynamically retrieved papers."""
    prompt = data.get("prompt", "")
    papers = (data.get("metadata") or {}).get("papers", [])[:5]
    parts = [prompt]
    
    paper_contexts = []
    for p in papers:
        title   = p.get("title", "")
        summary = p.get("summary", p.get("abstract", ""))
        if title or summary:
            paper_contexts.append(f"{title}\n{summary}".strip())
            
    # Gather new context from Weaviate searches
    for turn in data.get("turns", []):
        if _get_turn_field(turn, "turn_type") == "retriever_search":
            search_result = _get_turn_field(turn, "search_result", [])
            for p in search_result[:5]:
                title   = p.get("title", "")
                summary = p.get("summary", p.get("abstract", ""))
                if title or summary:
                    paper_contexts.append(f"{title}\n{summary}".strip())
    
    seen = set()
    unique_papers = []
    for p in paper_contexts:
        if p not in seen:
            seen.add(p)
            unique_papers.append(p)
            
    parts.extend(unique_papers)
    return "\n\n".join(parts)[:12_000]


def _extract_signal(data: dict) -> str:
    """
    Signal-space string: concatenated clean Retriever outputs from message turns.
    We always use `output_content` (pre-noise) so the metric reflects the
    agent's *intended* signal, not channel noise artefacts.
    """
    turns = data.get("turns", [])
    parts = []
    for turn in turns:
        if _get_turn_field(turn, "turn_type") == "retriever_message":
            content = _get_turn_field(turn, "output_content", "")
            if content:
                parts.append(content)
    return " ".join(parts)[:12_000]


def _extract_noisy_signal(data: dict) -> str | None:
    """
    Noisy signal: concatenated `noisy_output_content` for retriever_message turns.
    Returns None if the log was produced without channel noise.
    """
    turns = data.get("turns", [])
    parts = []
    for turn in turns:
        if _get_turn_field(turn, "turn_type") == "retriever_message":
            noisy = _get_turn_field(turn, "noisy_output_content")
            if noisy:
                parts.append(noisy)
    if not parts:
        return None
    return " ".join(parts)[:12_000]


def _extract_generator_output(data: dict) -> str:
    """Extract raw output from the final hypothesis generation turn."""
    turns = data.get("turns", [])
    for turn in turns:
        if _get_turn_field(turn, "turn_type") == "generator_generate":
            # Raw model output is always stored in turn["output"] (or in extra)
            return _get_turn_field(turn, "output", "")
    return ""


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
    Windowed ARI analysis between meaning-space and signal-space clusters.
    Returns a DataFrame with one row per window.
    """
    print("\n[3a] Positive Signaling — t-SNE + DBSCAN + ARI")
    parsed_files = _load_sorted_logs(logs_dir)
    if not parsed_files:
        print(f"  No logs found in: {logs_dir}")
        return pd.DataFrame()

    print(f"  Found {len(parsed_files)} trajectories.")
    results = []
    last_window_embeddings = None  # kept for t-SNE final plot

    for win_start in range(0, len(parsed_files), window_size):
        batch = parsed_files[win_start : win_start + window_size]
        win_idx = win_start // window_size + 1

        if len(batch) < window_size:
            print(f"  Window {win_idx}: incomplete ({len(batch)}/{window_size}) — skipping.")
            continue

        meanings: list[str] = []
        signals:  list[str] = []

        for _, fpath in batch:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            m = _extract_meaning(data)
            s = _extract_signal(data)
            if m and s:
                meanings.append(m)
                signals.append(s)

        if len(meanings) < 10:
            print(f"  Window {win_idx}: too few samples ({len(meanings)}) — skipping.")
            continue

        print(f"  Window {win_idx}: embedding {len(meanings)} samples...")

        m_embs_raw = await _get_embeddings_batch(meanings, embed_host, embed_port, EMBED_MODEL)
        s_embs_raw = await _get_embeddings_batch(signals,  embed_host, embed_port, EMBED_MODEL)

        # L2-normalise so Euclidean distance ≈ cosine distance (needed by DBSCAN)
        m_embs = normalize(np.array(m_embs_raw))
        s_embs = normalize(np.array(s_embs_raw))

        # DBSCAN clustering
        m_labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="euclidean").fit_predict(m_embs)
        s_labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="euclidean").fit_predict(s_embs)

        n_meaning_clusters = len(set(m_labels)) - (1 if -1 in m_labels else 0)
        n_signal_clusters  = len(set(s_labels)) - (1 if -1 in s_labels else 0)
        noise_frac = np.mean(m_labels == -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ari = adjusted_rand_score(m_labels, s_labels)

        print(
            f"    ARI={ari:.4f} | "
            f"meaning_clusters={n_meaning_clusters} | "
            f"signal_clusters={n_signal_clusters} | "
            f"noise_frac={noise_frac:.2f}"
        )

        results.append({
            "Window_Index":        win_idx,
            "N_Samples":           len(meanings),
            "ARI":                 round(ari, 4),
            "N_Meaning_Clusters":  n_meaning_clusters,
            "N_Signal_Clusters":   n_signal_clusters,
            "Noise_Fraction":      round(float(noise_frac), 4),
        })

        last_window_embeddings = (s_embs, m_labels, ari, win_idx)

    if not results:
        print("  No complete windows produced results.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp3_signaling_ari_over_time.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved ARI data → {csv_path}")

    # ── Shared style ────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':         10,
        'axes.labelsize':    10,
        'axes.titlesize':    11,
        'legend.fontsize':    9,
        'xtick.labelsize':    9,
        'ytick.labelsize':    9,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
    })

    # ── Plot 1: ARI over time ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["Window_Index"], df["ARI"],
            color=ACCENT, linewidth=1.2)
    ax.fill_between(df["Window_Index"], df["ARI"], alpha=0.06, color=ACCENT)
    ax.axhline(0, linewidth=0.6, color=GREY_REF, linestyle=':',
               label='Chance (ARI = 0)')
    ax.set_title(
        'Experiment 3a — Positive signaling: ARI between meaning- and signal-space clusters',
        pad=8,
    )
    ax.set_xlabel('Training window')
    ax.set_ylabel('Adjusted Rand Index')
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False)
    ax.grid(axis='y', linewidth=0.4, color=GREY_REF, linestyle=':')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp3_signaling_ari_over_time.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot 2: t-SNE of final window, coloured by meaning cluster ──────────
    if last_window_embeddings is not None:
        s_embs_final, m_labels_final, ari_final, win_final = last_window_embeddings

        print(f"  Computing t-SNE for final window ({s_embs_final.shape[0]} points)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsne_coords = TSNE(
                n_components=2,
                perplexity=min(30, s_embs_final.shape[0] // 3),
                random_state=42,
                init="pca",
            ).fit_transform(s_embs_final)

        unique_labels = sorted(set(m_labels_final))
        n_clusters    = len(unique_labels)

        # Greyscale ramp for clusters; noise points get a light grey
        greys = plt.cm.get_cmap("Greys", n_clusters + 2)
        label_to_color = {
            lbl: ("#cccccc" if lbl == -1 else greys(0.3 + 0.6 * i / max(n_clusters - 1, 1)))
            for i, lbl in enumerate(l for l in unique_labels if l != -1)
        }
        if -1 in unique_labels:
            label_to_color[-1] = "#cccccc"

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        for lbl in unique_labels:
            mask      = m_labels_final == lbl
            is_noise  = lbl == -1
            ax2.scatter(
                tsne_coords[mask, 0], tsne_coords[mask, 1],
                c=[label_to_color[lbl]],
                s=10 if is_noise else 18,
                alpha=0.3 if is_noise else 0.65,
                linewidths=0,
                label="Noise" if is_noise else f"Cluster {lbl}",
                zorder=1 if is_noise else 2,
            )

        ax2.set_title(
            f"Experiment 3a — Signal space (t-SNE), final window\n"
            f"Colour = meaning-space cluster  |  ARI = {ari_final:.4f}",
            pad=8,
        )
        ax2.set_xlabel("t-SNE dim 1")
        ax2.set_ylabel("t-SNE dim 2")
        ax2.yaxis.set_label_coords(-0.1, 0.5)
        ax2.legend(
            handles=[
                mpatches.Patch(
                    color=label_to_color[l],
                    label="Noise" if l == -1 else f"Cluster {l}",
                )
                for l in unique_labels
            ],
            loc="best", fontsize=8, frameon=False,
        )
        ax2.grid(linewidth=0.4, color=GREY_REF, linestyle=":")
        fig2.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "exp3_signaling_tsne_final.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print("  t-SNE plot saved.")

    return df


# ---------------------------------------------------------------------------
# 3b — Positive Listening (Proxy CIC)
# ---------------------------------------------------------------------------

async def run_positive_listening(
    logs_dir: str,
    output_dir: str,
    embed_host: str,
    embed_port: int,
    window_size: int = 50,
    noisy_logs_dir: str | None = None,
) -> pd.DataFrame:
    """
    Two modes:

    Within-run (default):
        For each trajectory that has `noisy_output_content` in its log,
        compute cosine distance between the clean signal embedding and the
        noisy signal embedding.  A rising distance → less robust grounding.
        A stable, non-zero distance shows the Generator would see a different
        signal, making cross-run comparison meaningful.

    Cross-run (--noisy_logs_dir supplied):
        Compare Generator outputs embeddings between a baseline run
        (no noise) and a noisy run, matched by prompt.  Reports the mean
        cosine distance between matched pairs — a direct proxy for CIC.
    """
    print("\n[3b] Positive Listening — Proxy CIC")

    if noisy_logs_dir:
        return await _cross_run_cic(
            clean_dir=logs_dir,
            noisy_dir=noisy_logs_dir,
            output_dir=output_dir,
            embed_host=embed_host,
            embed_port=embed_port,
            window_size=window_size,
        )
    else:
        return await _within_run_signal_sensitivity(
            logs_dir=logs_dir,
            output_dir=output_dir,
            embed_host=embed_host,
            embed_port=embed_port,
            window_size=window_size,
        )


async def _within_run_signal_sensitivity(
    logs_dir: str,
    output_dir: str,
    embed_host: str,
    embed_port: int,
    window_size: int,
) -> pd.DataFrame:
    """
    Within-run mode: embed clean and noisy signals, compute cosine distance.
    """
    parsed_files = _load_sorted_logs(logs_dir)
    if not parsed_files:
        print(f"  No logs found in: {logs_dir}")
        return pd.DataFrame()

    # Filter to only files that have noisy_output_content
    noisy_files = []
    for ts, fpath in parsed_files:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        noisy = _extract_noisy_signal(data)
        if noisy:
            noisy_files.append((ts, fpath))

    if not noisy_files:
        print(
            "  No trajectories with `noisy_output_content` found.\n"
            "  Re-run training with APPLY_CHANNEL_NOISE=true, or supply\n"
            "  --noisy_logs_dir for cross-run CIC."
        )
        return pd.DataFrame()

    print(f"  Found {len(noisy_files)} noisy trajectories.")
    results = []

    for win_start in range(0, len(noisy_files), window_size):
        batch = noisy_files[win_start : win_start + window_size]
        win_idx = win_start // window_size + 1

        if len(batch) < window_size:
            print(f"  Window {win_idx}: incomplete — skipping.")
            continue

        clean_signals: list[str] = []
        noisy_signals: list[str] = []

        for _, fpath in batch:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            c = _extract_signal(data)
            n = _extract_noisy_signal(data)
            if c and n:
                clean_signals.append(c)
                noisy_signals.append(n)

        if len(clean_signals) < 5:
            print(f"  Window {win_idx}: too few paired samples — skipping.")
            continue

        print(f"  Window {win_idx}: embedding {len(clean_signals)} clean+noisy pairs...")

        all_texts = clean_signals + noisy_signals
        all_embs  = await _get_embeddings_batch(all_texts, embed_host, embed_port, EMBED_MODEL)
        c_embs = np.array(all_embs[: len(clean_signals)])
        n_embs = np.array(all_embs[len(clean_signals) :])

        # Pairwise cosine distances between clean and noisy signals
        dists = [
            cosine_dist(c_embs[i], n_embs[i])
            for i in range(len(c_embs))
        ]
        mean_dist = float(np.mean(dists))
        std_dist  = float(np.std(dists))

        print(f"    Signal sensitivity: mean_cos_dist={mean_dist:.4f} ± {std_dist:.4f}")

        results.append({
            "Window_Index":      win_idx,
            "N_Pairs":           len(dists),
            "Mean_Signal_Dist":  round(mean_dist, 4),
            "Std_Signal_Dist":   round(std_dist,  4),
            "Mode":              "within_run",
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp3_listening_cic_over_time.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved CIC data → {csv_path}")

    _plot_cic(df, output_dir, title_suffix="(Within-Run Signal Sensitivity)")
    return df


async def _cross_run_cic(
    clean_dir: str,
    noisy_dir: str,
    output_dir: str,
    embed_host: str,
    embed_port: int,
    window_size: int,
) -> pd.DataFrame:
    """
    Cross-run mode: match trajectories by prompt, compare Generator outputs.
    """
    print("  Cross-run CIC mode.")

    def _load_prompt_map(logs_dir: str) -> dict[str, str]:
        """Map prompt → generator output_content for each trajectory."""
        mapping: dict[str, str] = {}
        for _, fpath in _load_sorted_logs(logs_dir):
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            p = data.get("prompt", "")
            g = _extract_generator_output(data)
            if p and g and p not in mapping:
                mapping[p] = g
        return mapping

    print("  Loading clean run...")
    clean_map = _load_prompt_map(clean_dir)
    print("  Loading noisy run...")
    noisy_map = _load_prompt_map(noisy_dir)

    # Intersect on prompt
    common_prompts = list(set(clean_map) & set(noisy_map))
    if not common_prompts:
        print("  No matching prompts found between clean and noisy runs. Aborting CIC.")
        return pd.DataFrame()

    print(f"  Matched {len(common_prompts)} prompts across runs.")

    clean_outputs = [clean_map[p] for p in common_prompts]
    noisy_outputs = [noisy_map[p] for p in common_prompts]

    all_texts = clean_outputs + noisy_outputs
    all_embs  = await _get_embeddings_batch(all_texts, embed_host, embed_port, EMBED_MODEL)
    c_embs = np.array(all_embs[: len(common_prompts)])
    n_embs = np.array(all_embs[len(common_prompts) :])

    dists = [cosine_dist(c_embs[i], n_embs[i]) for i in range(len(c_embs))]

    df = pd.DataFrame({
        "Prompt_Index":       range(len(common_prompts)),
        "Generator_Cos_Dist": [round(d, 4) for d in dists],
    })
    csv_path = os.path.join(output_dir, "exp3_listening_cic_distribution.csv")
    df.to_csv(csv_path, index=False)

    mean_cic = float(np.mean(dists))
    print(f"  Proxy CIC = {mean_cic:.4f} (mean cosine distance of Generator outputs)")

    # ── Distribution plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dists, bins=30, color=ACCENT, alpha=0.75, linewidth=0)
    ax.axvline(mean_cic, color=ACCENT, linestyle='--', linewidth=1.0,
               label=f'Mean = {mean_cic:.4f}')
    ax.set_title(
        'Experiment 3b — Positive listening: generator output shift\n'
        '(cross-run proxy CIC)',
        pad=8,
    )
    ax.set_xlabel('Cosine distance (generator output: clean vs. noisy run)')
    ax.set_ylabel('Count')
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.legend(frameon=False)
    ax.grid(axis='y', linewidth=0.4, color=GREY_REF, linestyle=':')
    fig.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "exp3_listening_cic_distribution.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    print("  CIC distribution plot saved.")

    return df


def _plot_cic(df: pd.DataFrame, output_dir: str, title_suffix: str = "") -> None:
    """Line plot of mean signal distance over training windows."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["Window_Index"], df["Mean_Signal_Dist"],
            color=ACCENT, linewidth=1.2)
    ax.fill_between(
        df["Window_Index"],
        df["Mean_Signal_Dist"] - df["Std_Signal_Dist"],
        df["Mean_Signal_Dist"] + df["Std_Signal_Dist"],
        color=ACCENT, alpha=0.08, label="± 1 SD",
    )
    ax.set_title(
        f'Experiment 3b — Positive listening: signal sensitivity over training\n'
        f'{title_suffix}',
        pad=8,
    )
    ax.set_xlabel('Training window')
    ax.set_ylabel('Mean cosine distance\n(clean vs. noisy signal)')
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(axis='y', linewidth=0.4, color=GREY_REF, linestyle=':')
    fig.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "exp3_listening_cic_over_time.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _write_summary(
    output_dir: str,
    signaling_df: pd.DataFrame,
    listening_df: pd.DataFrame,
) -> None:
    summary: dict = {}

    if not signaling_df.empty:
        summary["positive_signaling"] = {
            "final_ari":     float(signaling_df["ARI"].iloc[-1]),
            "max_ari":       float(signaling_df["ARI"].max()),
            "mean_ari":      round(float(signaling_df["ARI"].mean()), 4),
            "n_windows":     int(len(signaling_df)),
        }

    if not listening_df.empty:
        if "Mean_Signal_Dist" in listening_df.columns:
            summary["positive_listening"] = {
                "mode":              "within_run",
                "final_mean_dist":   float(listening_df["Mean_Signal_Dist"].iloc[-1]),
                "mean_dist_overall": round(float(listening_df["Mean_Signal_Dist"].mean()), 4),
                "n_windows":         int(len(listening_df)),
            }
        elif "Generator_Cos_Dist" in listening_df.columns:
            summary["positive_listening"] = {
                "mode":          "cross_run",
                "proxy_cic":     round(float(listening_df["Generator_Cos_Dist"].mean()), 4),
                "n_pairs":       int(len(listening_df)),
            }

    path = os.path.join(output_dir, "exp3_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {path}")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3 — Pragmatics: Positive Signaling & Positive Listening"
    )
    parser.add_argument("--logs_dir",      default="./Agents/workflow_logs",
                        help="Directory with traj_*.json debug logs")
    parser.add_argument("--noisy_logs_dir", default=None,
                        help="(Optional) Separate log directory for cross-run CIC")
    parser.add_argument("--output_dir",    default="./Agents/eval_results/experiment_3")
    parser.add_argument("--window_size",   type=int, default=50)
    parser.add_argument("--embed_host",    default=os.getenv("EMBED_HOST", "localhost"))
    parser.add_argument("--embed_port",    type=int, default=int(os.getenv("EMBED_PORT", 8000)))
    parser.add_argument("--cross_run_cic", action="store_true",
                        help="Force cross-run CIC even without --noisy_logs_dir (will warn)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Auto-detect 'train' subfolder ───────────────────────────────────────
    def _use_train_if_exists(directory: str) -> str:
        train_dir = os.path.join(directory, "train")
        if os.path.isdir(train_dir):
            print(f"Auto-detected 'train' subfolder: {train_dir}")
            return train_dir
        return directory

    logs_dir = _use_train_if_exists(args.logs_dir)
    noisy_logs_dir = _use_train_if_exists(args.noisy_logs_dir) if args.noisy_logs_dir else None

    print("=" * 60)
    print("Experiment 3 — Pragmatics Evaluation")
    print(f"  logs_dir    : {logs_dir}")
    if noisy_logs_dir:
        print(f"  noisy_dir   : {noisy_logs_dir}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  embed_server: {args.embed_host}:{args.embed_port}")
    print("=" * 60)

    sig_df = await run_positive_signaling(
        logs_dir   = logs_dir,
        output_dir = args.output_dir,
        embed_host = args.embed_host,
        embed_port = args.embed_port,
        window_size= args.window_size,
    )

    lis_df = await run_positive_listening(
        logs_dir       = logs_dir,
        output_dir     = args.output_dir,
        embed_host     = args.embed_host,
        embed_port     = args.embed_port,
        window_size    = args.window_size,
        noisy_logs_dir = noisy_logs_dir,
    )

    _write_summary(args.output_dir, sig_df, lis_df)
    print("\nExperiment 3 completed.")


if __name__ == "__main__":
    asyncio.run(main())