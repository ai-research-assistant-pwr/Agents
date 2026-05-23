"""
Semantics Evaluation v2 — Experiment 2 & 3 (Merged Suite)
=========================================================
Comprehensive semantic analysis of emergent communication. 
Combines window-based structural analysis (Measure A & B) with global 
trajectory clustering (Measure C) using UMAP and HDBSCAN.

Measure A — Intra-window signal dispersion
Tracks mean pairwise cosine distance between Retriever messages within 
each training window. 

Measure B — Signal-prompt alignment (grounding index)
Computes cosine similarity between the Retriever's message and the user prompt.

Measure C — Global Message Space Evolution (UMAP + HDBSCAN)
Projects all messages across the entire training run onto a 2D UMAP space, 
colored by training window to show temporal drift. Performs HDBSCAN clustering 
on the ORIGINAL high-dimensional embeddings to identify true semantic clusters, 
extracting TF-IDF keywords for each.

Output
------
  <output_dir>/
    ├── exp2_signal_structure.csv           (Windowed metrics A & B)
    ├── exp2_clustering_metrics.csv         (Global clustering metrics)
    ├── exp2_cluster_keywords.json          (TF-IDF keywords per cluster)
    ├── exp2_dispersion_over_time.png       
    ├── exp2_grounding_over_time.png        
    ├── exp2_umap_by_time.png               (Colored by temporal drift)
    ├── exp2_umap_by_cluster.png            (Colored by HDBSCAN label)
    ├── exp2_umap_by_reward.png             (Colored by final trajectory reward)
    └── exp2_umap_multi_run.png             (If multiple runs are provided)
"""

import os
import json
import glob
import asyncio
import argparse
from collections import defaultdict

import aiohttp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, cosine as cosine_dist
from itertools import combinations

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    import umap
    import hdbscan
except ImportError:
    raise ImportError("Please install required packages: pip install umap-learn hdbscan scikit-learn")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_HOST  = os.getenv("EMBED_HOST", "localhost")
EMBED_PORT  = int(os.getenv("EMBED_PORT", "8000"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
BATCH_SIZE  = 32

STOP_WORDS_EN = {
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "she", "her",
    "it", "its", "they", "them", "what", "which", "who", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "a", "an", "the",
    "and", "but", "if", "or", "as", "of", "at", "by", "for", "with",
    "about", "into", "through", "to", "from", "in", "out", "on", "not",
    "no", "can", "will", "just", "than", "so", "also", "such", "more",
    "may", "would", "could", "should", "study", "research", "paper",
    "show", "shown", "found", "results", "using", "used", "based",
    "two", "one", "three", "four", "five", "however", "thus", "therefore",
    "between", "among", "provide", "provides", "suggest", "suggests",
    # Technical tokens from noise injection
    "mask", "[mask]"
}

# ---------------------------------------------------------------------------
# Helpers: Data Loading & Embeddings
# ---------------------------------------------------------------------------

def _get_field(turn: dict, field: str, default=None):
    """Look up a field in turn dict, checking 'extra' sub-dict first."""
    extra = turn.get("extra", {})
    if field in extra:
        return extra[field]
    return turn.get(field, default)

def _load_sorted_logs(logs_dir: str) -> list:
    """Return (timestamp, filepath) pairs sorted by timestamp."""
    train_dir = os.path.join(logs_dir, "train")
    if os.path.isdir(train_dir):
        logs_dir = train_dir
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

async def _get_embeddings(texts: list, host: str, port: int, model: str) -> list:
    """Fetch embeddings from vLLM asynchronously in batches."""
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
                        timeout=aiohttp.ClientTimeout(total=180)
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
    dists = [cosine_dist(emb_matrix[i], emb_matrix[j]) for i, j in combinations(range(n), 2)]
    return float(np.mean(dists)), float(np.std(dists))


# ---------------------------------------------------------------------------
# Core Analysis Functions
# ---------------------------------------------------------------------------

async def analyse_run_windowed(
    logs_dir: str, window_size: int, run_label: str, embed_host: str, embed_port: int, embed_model: str
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Computes windowed Measure A & B, and collects flat data for Measure C (Clustering).
    """
    parsed_files = _load_sorted_logs(logs_dir)
    if not parsed_files:
        print(f"  [{run_label}] No traj_*.json files found in {logs_dir}")
        return pd.DataFrame(), []

    print(f"  [{run_label}] Found {len(parsed_files)} trajectories. Analyzing windows...")
    results = []
    global_messages = []
    total_files = len(parsed_files)

    for win_start in range(0, total_files, window_size):
        batch = parsed_files[win_start : win_start + window_size]
        if len(batch) < window_size:
            continue

        window_idx = win_start // window_size + 1
        prompts, messages, rewards, traj_indices = [], [], [], []

        for offset, (_, fpath) in enumerate(batch):
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            prompt = data.get("prompt", "")
            reward = data.get("total_reward", 0.0)

            for turn in data.get("turns", []):
                if _get_field(turn, "turn_type") == "retriever_message":
                    msg = (_get_field(turn, "output_content", "") or "").strip()
                    if msg:
                        messages.append(msg)
                        prompts.append(prompt)
                        rewards.append(reward)
                        traj_indices.append(win_start + offset)

        if len(messages) < 4:
            continue

        # Embed messages and prompts
        unique_prompts = list(set(prompts))
        all_texts = messages + unique_prompts
        all_embs = await _get_embeddings(all_texts, embed_host, embed_port, embed_model)
        
        msg_embs = normalize(np.array(all_embs[:len(messages)]))
        prompt_embs = normalize(np.array(all_embs[len(messages):]))
        prompt_to_emb = {p: prompt_embs[i] for i, p in enumerate(unique_prompts)}

        # Measure A: Dispersion
        sample_idx = np.random.choice(len(msg_embs), min(200, len(msg_embs)), replace=False)
        mean_dist, std_dist = _mean_pairwise_cosine_distance(msg_embs[sample_idx])

        # Measure B: Grounding
        sims = [float(np.dot(m_emb, prompt_to_emb[p])) for m_emb, p in zip(msg_embs, prompts) if p in prompt_to_emb]
        
        results.append({
            "Run": run_label,
            "Window_Index": window_idx,
            "Mean_Dispersion": round(mean_dist, 4),
            "Std_Dispersion": round(std_dist, 4),
            "Mean_Grounding": round(np.mean(sims) if sims else 0, 4),
            "Std_Grounding": round(np.std(sims) if sims else 0, 4),
            "Mean_Reward": round(np.mean(rewards), 4),
        })

        # Collect for Measure C
        for i in range(len(messages)):
            quartile = int(traj_indices[i] / total_files * 4)
            global_messages.append({
                "text": messages[i],
                "embedding": msg_embs[i],
                "reward": rewards[i],
                "traj_idx": traj_indices[i],
                "window_quartile": quartile,
                "run_label": run_label
            })

    return pd.DataFrame(results), global_messages

def extract_cluster_keywords(records: list[dict], labels: np.ndarray, top_k: int = 8) -> dict:
    """Extracts top TF-IDF keywords for each HDBSCAN cluster."""
    cluster_docs = defaultdict(list)
    for rec, label in zip(records, labels):
        if label >= 0:
            cluster_docs[int(label)].append(rec["text"])

    if not cluster_docs:
        return {}

    all_cluster_ids = sorted(cluster_docs.keys())
    corpus = [" ".join(cluster_docs[c]) for c in all_cluster_ids]

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=list(STOP_WORDS_EN),
        ngram_range=(1, 2),
        min_df=2,
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        return {}

    keywords = {}
    for i, cluster_id in enumerate(all_cluster_ids):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-top_k:][::-1]
        keywords[cluster_id] = [{"term": feature_names[j], "score": round(float(row[j]), 4)} for j in top_indices]
    return keywords

def run_hdbscan_clustering(embs_orig: np.ndarray, records: list[dict], min_cluster_size: int) -> tuple[np.ndarray, dict]:
    """
    Runs HDBSCAN on the original L2-normalized embeddings.
    Since data is L2-normalized, Euclidean distance is monotonically related to Cosine distance.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_epsilon=0.1)
    labels = clusterer.fit_predict(embs_orig)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_pct = 100 * n_noise / len(labels)

    metrics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(noise_pct, 2),
    }

    if n_clusters >= 2:
        mask = labels >= 0
        if mask.sum() >= 2:
            sil = silhouette_score(embs_orig[mask], labels[mask], metric="cosine")
            db_idx = davies_bouldin_score(embs_orig[mask], labels[mask])
            metrics["silhouette"] = round(float(sil), 4)
            metrics["davies_bouldin"] = round(float(db_idx), 4)

    # Temporal drift: Centroid of Q0 vs Q3
    quartiles = np.array([r["window_quartile"] for r in records])
    q0_mask, q3_mask = quartiles == 0, quartiles == 3
    if q0_mask.sum() > 0 and q3_mask.sum() > 0:
        c0 = embs_orig[q0_mask].mean(axis=0, keepdims=True)
        c3 = embs_orig[q3_mask].mean(axis=0, keepdims=True)
        metrics["temporal_drift_cosine"] = round(float(cdist(c0, c3, metric="cosine")[0, 0]), 4)

    return labels, metrics


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

_STYLE = {
    "font.family": "serif", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
}

def plot_windowed_metrics(df: pd.DataFrame, output_dir: str):
    plt.rcParams.update(_STYLE)
    
    # 1. Dispersion
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in df.groupby("Run"):
        ax.plot(group["Window_Index"], group["Mean_Dispersion"], label=label, lw=1.5)
        ax.fill_between(group["Window_Index"], group["Mean_Dispersion"] - group["Std_Dispersion"], 
                        group["Mean_Dispersion"] + group["Std_Dispersion"], alpha=0.1)
    ax.set(title="Experiment 2 — Intra-window signal dispersion", xlabel="Training Window", ylabel="Mean Cosine Distance")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_dispersion_over_time.png"), dpi=300)
    plt.close()

    # 2. Grounding
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in df.groupby("Run"):
        ax.plot(group["Window_Index"], group["Mean_Grounding"], label=label, lw=1.5)
        ax.fill_between(group["Window_Index"], group["Mean_Grounding"] - group["Std_Grounding"], 
                        group["Mean_Grounding"] + group["Std_Grounding"], alpha=0.1)
    ax.set(title="Experiment 2 — Signal-prompt alignment (Grounding)", xlabel="Training Window", ylabel="Cosine Similarity")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_grounding_over_time.png"), dpi=300)
    plt.close()

def plot_umap_scatter(xy, c, cmap, title, cbar_label, output_path, vmin=None, vmax=None):
    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, s=8, alpha=0.5, linewidths=0, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
    plt.colorbar(sc, ax=ax, label=cbar_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_umap_clusters(xy, labels, output_path):
    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(9, 6))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))

    for lbl in unique_labels:
        mask = labels == lbl
        color = "lightgray" if lbl == -1 else cmap(unique_labels.index(lbl))
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[color], s=10 if lbl >= 0 else 4, 
                   alpha=0.6 if lbl >= 0 else 0.2, linewidths=0, label=f"Cluster {lbl}" if lbl >= 0 else "Noise")
    
    ax.set_xticks([]); ax.set_yticks([])
    ax.set(title="Global UMAP Projection: HDBSCAN Clusters", xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1), markerscale=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_umap_multi_run(xy_list, labels_list, output_path):
    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#1a1a1a", "#e6194b", "#3cb44b", "#4363d8", "#f58231"]
    
    for i, (xy, label) in enumerate(zip(xy_list, labels_list)):
        ax.scatter(xy[:, 0], xy[:, 1], c=colors[i % len(colors)], s=6, alpha=0.4, linewidths=0, label=label)
        
    ax.set_xticks([]); ax.set_yticks([])
    ax.set(title="UMAP Comparison: Multiple Training Runs", xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Experiment 2 & 3 — Semantics & Global Clustering Suite")
    parser.add_argument("--logs_dir", default="./Agents/workflow_logs")
    parser.add_argument("--baseline_logs_dir", default=None, help="Optional baseline run for comparison from run_eval.sh")
    parser.add_argument("--extra_logs_dir", nargs="*", default=[], help="Extra runs 'label:path'")
    parser.add_argument("--output_dir", default="./Agents/eval_results/merged_semantics")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--hdbscan_min_samples", type=int, default=15)
    parser.add_argument("--umap_neighbors", type=int, default=50)
    
    # Argumenty przekazywane przez run_eval.sh
    parser.add_argument("--embed_host", default=EMBED_HOST)
    parser.add_argument("--embed_port", type=int, default=EMBED_PORT)
    parser.add_argument("--embed_model", default=EMBED_MODEL)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== Experiment 2 & 3: Semantic Evolution & Global Clustering ===")
    
    # 1. Process Main Run (Windowed + Global Data)
    df_main, global_main = await analyse_run_windowed(
        args.logs_dir, args.window_size, "main", args.embed_host, args.embed_port, args.embed_model
    )
    
    if not global_main:
        print("No data extracted. Exiting.")
        return

    # 2. Process Extra Runs (w tym baseline z run_eval.sh)
    dfs = [df_main]
    global_extras = []
    
    # Kompatybilność z parametrem BASELINE_ARG ze skryptu run_eval.sh
    if args.baseline_logs_dir:
        args.extra_logs_dir.append(f"baseline:{args.baseline_logs_dir}")

    for spec in args.extra_logs_dir:
        if ":" not in spec: continue
        label, path = spec.split(":", 1)
        df_ext, glob_ext = await analyse_run_windowed(
            path, args.window_size, label, args.embed_host, args.embed_port, args.embed_model
        )
        if glob_ext:
            dfs.append(df_ext)
            global_extras.append((label, glob_ext))

    # Save Windowed Metrics
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(os.path.join(args.output_dir, "exp2_signal_structure.csv"), index=False)
    plot_windowed_metrics(df_all, args.output_dir)

    # 3. Global UMAP on Main Run
    print("\nFitting UMAP on main trajectory...")
    main_embs = np.array([r["embedding"] for r in global_main])
    reducer = umap.UMAP(n_neighbors=args.umap_neighbors, min_dist=0.01, n_components=2, metric="cosine", random_state=42)
    main_embs_2d = reducer.fit_transform(main_embs)

    # 4. HDBSCAN Clustering (On Original Embeddings)
    print(f"Running HDBSCAN (min_samples={args.hdbscan_min_samples})...")
    labels, metrics = run_hdbscan_clustering(main_embs, global_main, args.hdbscan_min_samples)
    
    print(f"  Detected Clusters: {metrics['n_clusters']} | Noise: {metrics['noise_pct']:.1f}%")
    if "temporal_drift_cosine" in metrics:
        print(f"  Temporal Drift (Cosine): {metrics['temporal_drift_cosine']:.4f}")

    # Save Clustering Metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(args.output_dir, "exp2_clustering_metrics.csv"), index=False)

    # 5. Extract TF-IDF Keywords
    print("Extracting TF-IDF Keywords per cluster...")
    kw = extract_cluster_keywords(global_main, labels)
    with open(os.path.join(args.output_dir, "exp2_cluster_keywords.json"), "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=2)

    # 6. Global Plotting
    print("Generating UMAP projections...")
    times = np.array([r["traj_idx"] for r in global_main])
    plot_umap_scatter(main_embs_2d, times, "plasma", "Temporal Drift (Trajectory Index)", "Chronological Index", 
                      os.path.join(args.output_dir, "exp2_umap_by_time.png"))
    
    rewards = np.array([r["reward"] for r in global_main])
    plot_umap_scatter(main_embs_2d, rewards, "RdYlGn", "UMAP colored by Trajectory Reward", "Total Reward", 
                      os.path.join(args.output_dir, "exp2_umap_by_reward.png"),
                      vmin=np.percentile(rewards, 5), vmax=np.percentile(rewards, 95))

    plot_umap_clusters(main_embs_2d, labels, os.path.join(args.output_dir, "exp2_umap_by_cluster.png"))

    # 7. Multi-run Plotting (Projection via fitted UMAP)
    if global_extras:
        print("Projecting additional runs onto shared UMAP space...")
        all_2d = [main_embs_2d]
        all_labels = ["main"]
        
        for run_label, run_data in global_extras:
            ext_embs = np.array([r["embedding"] for r in run_data])
            ext_2d = reducer.transform(ext_embs)  # Project into the same manifold
            all_2d.append(ext_2d)
            all_labels.append(run_label)
            
        plot_umap_multi_run(all_2d, all_labels, os.path.join(args.output_dir, "exp2_umap_multi_run.png"))

    print(f"\n=== Pipeline Complete. Results saved to {args.output_dir} ===")

if __name__ == "__main__":
    asyncio.run(main())