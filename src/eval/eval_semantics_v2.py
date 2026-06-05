"""
Semantics Evaluation v2 — Windowed Structural Analysis
======================================================
Semantic analysis of emergent communication focusing on structural 
changes over time.

Measure A — Intra-window signal dispersion
Tracks mean pairwise cosine distance between Retriever messages within 
each training window. 

Measure B — Signal-prompt alignment (grounding index)
Computes cosine similarity between the Retriever's message and the user prompt.

Output
------
  <output_dir>/
    ├── exp2_signal_structure.csv           (Windowed metrics A & B)
    ├── exp2_dispersion_over_time.png       
    └── exp2_grounding_over_time.png        
"""

import os
import json
import glob
import asyncio
import argparse

import aiohttp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_dist
from itertools import combinations

from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_HOST  = os.getenv("EMBED_HOST", "localhost")
EMBED_PORT  = int(os.getenv("EMBED_PORT", "8000"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
BATCH_SIZE  = 32

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
) -> pd.DataFrame:
    """
    Computes windowed Measure A (Dispersion) & B (Grounding).
    """
    parsed_files = _load_sorted_logs(logs_dir)
    if not parsed_files:
        print(f"  [{run_label}] No traj_*.json files found in {logs_dir}")
        return pd.DataFrame()

    print(f"  [{run_label}] Found {len(parsed_files)} trajectories. Analyzing windows...")
    results = []
    total_files = len(parsed_files)

    for win_start in range(0, total_files, window_size):
        batch = parsed_files[win_start : win_start + window_size]
        if len(batch) < window_size:
            continue

        window_idx = win_start // window_size + 1
        prompts, messages, rewards = [], [], []

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

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

_STYLE = {
    'font.family':      'serif',
    'font.size':        11,
    'axes.labelsize':   11,
    'axes.titlesize':   12,
    'legend.fontsize':   10,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'axes.linewidth':   0.8,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
}

def plot_windowed_metrics(df: pd.DataFrame, output_dir: str):
    plt.rcParams.update(_STYLE)
    
    GREY_REF = '#cccccc'
    
    # 1. Dispersion
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    for label, group in df.groupby("Run"):
        ax1.plot(group["Window_Index"], group["Mean_Dispersion"], label=label, lw=1.5)
        ax1.fill_between(group["Window_Index"], 
                         group["Mean_Dispersion"] - group["Std_Dispersion"], 
                         group["Mean_Dispersion"] + group["Std_Dispersion"], alpha=0.1)
    
    ax1.set_title("Dyspersja sygnału wewnątrz okna treningowego", pad=10)
    ax1.set_xlabel("Okno treningowe")
    ax1.set_ylabel("Średni dystans kosinusowy")
    ax1.grid(axis='y', lw=0.4, color=GREY_REF, ls='--')
    # Legenda wyciągnięta na zewnątrz po prawej stronie
    ax1.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_dispersion_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Grounding
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    for label, group in df.groupby("Run"):
        ax2.plot(group["Window_Index"], group["Mean_Grounding"], label=label, lw=1.5)
        ax2.fill_between(group["Window_Index"], 
                         group["Mean_Grounding"] - group["Std_Grounding"], 
                         group["Mean_Grounding"] + group["Std_Grounding"], alpha=0.1)
        
    ax2.set_title("Ugruntowanie (Podobieństwo sygnału do zapytania)", pad=10)
    ax2.set_xlabel("Okno treningowe")
    ax2.set_ylabel("Podobieństwo kosinusowe")
    ax2.grid(axis='y', lw=0.4, color=GREY_REF, ls='--')
    # Legenda wyciągnięta na zewnątrz po prawej stronie
    ax2.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_grounding_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Experiment 2 & 3 — Semantics Windowed Analysis")
    parser.add_argument("--logs_dir", default="./Agents/workflow_logs")
    parser.add_argument("--baseline_logs_dir", default=None, help="Optional baseline run for comparison from run_eval.sh")
    parser.add_argument("--extra_logs_dir", nargs="*", default=[], help="Extra runs 'label:path'")
    parser.add_argument("--output_dir", default="./Agents/eval_results/merged_semantics")
    parser.add_argument("--window_size", type=int, default=50)
    
    # Argumenty przekazywane przez run_eval.sh
    parser.add_argument("--embed_host", default=EMBED_HOST)
    parser.add_argument("--embed_port", type=int, default=EMBED_PORT)
    parser.add_argument("--embed_model", default=EMBED_MODEL)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== Experiment 2 & 3: Semantic Structural Evolution ===")
    
    dfs = []
    
    # 1. Process Main Run
    df_main = await analyse_run_windowed(
        args.logs_dir, args.window_size, "main", args.embed_host, args.embed_port, args.embed_model
    )
    
    if df_main.empty:
        print("No data extracted. Exiting.")
        return
        
    dfs.append(df_main)

    # 2. Process Extra Runs (w tym baseline z run_eval.sh)
    if args.baseline_logs_dir:
        args.extra_logs_dir.append(f"baseline:{args.baseline_logs_dir}")

    for spec in args.extra_logs_dir:
        if ":" not in spec: continue
        label, path = spec.split(":", 1)
        df_ext = await analyse_run_windowed(
            path, args.window_size, label, args.embed_host, args.embed_port, args.embed_model
        )
        if not df_ext.empty:
            dfs.append(df_ext)

    # Save and Plot Windowed Metrics
    df_all = pd.concat(dfs, ignore_index=True)
    csv_path = os.path.join(args.output_dir, "exp2_signal_structure.csv")
    df_all.to_csv(csv_path, index=False)
    
    print(f"Generating plots...")
    plot_windowed_metrics(df_all, args.output_dir)

    print(f"\n=== Pipeline Complete. Results saved to {args.output_dir} ===")

if __name__ == "__main__":
    asyncio.run(main())