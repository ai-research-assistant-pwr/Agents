"""
Semantics Evaluation — Experiment 2
======================================
Tests whether channel noise (random token masking during training) causes the
Retriever's emergent language to become more compositional — i.e. whether
similar input contexts systematically produce similar output signals.

Compositionality is measured with Topological Similarity (TopSim), the
Spearman rank correlation between pairwise distances in the *meaning space*
(inputs) and pairwise distances in the *signal space* (Retriever messages):

  TopSim = Spearman ρ ( dist_meaning(i,j),  dist_signal(i,j) )

  TopSim → +1 : perfectly compositional protocol
  TopSim →  0 : random / holistic protocol
  TopSim → −1 : anti-compositional (rarely observed)

Two signal-distance metrics are computed in parallel:

  Semantic TopSim (ρ_sem)
      Signal distances are cosine distances between vLLM embeddings of the
      Retriever outputs.  Captures high-level semantic similarity.

  Lexical TopSim (ρ_lex)
      Signal distances are normalised Levenshtein distances between the raw
      output strings.  Captures surface-form / syntactic similarity.

The analysis is performed both for the **intended** (clean) messages and,
if available, for the **noisy** messages (after channel noise injection).
Comparing the two shows whether the emergent protocol is robust to
transmission errors.

Meaning space
-------------
  prompt + concatenated titles & summaries of initial papers and any papers
  retrieved dynamically during 'retriever_search' (truncated to 12 000
  characters to stay within vLLM token limits).

Signal space
------------
  Concatenation of 'output_content' (clean intent) and separately
  concatenation of 'noisy_output_content' (noisy version) from all
  'retriever_message' turns.

Data contract
-------------
  Reads traj_*.json files written by scientific_workflow.py (DEBUG=True).
  Requires a running vLLM embedding server (Qwen/Qwen3-Embedding-4B by default).
  Server is configured via environment variables EMBED_HOST / EMBED_PORT /
  EMBED_MODEL, or via the argparse flags --embed_host / --embed_port.

Output
------
  eval_results/experiment_2/
    ├── exp2_topsim_evolution.csv          — ρ_sem, ρ_lex, ρ_noisy per window
    ├── exp2_topsim_evolution.png          — line plot over training steps
    └── exp2_topsim_scatter_final.png      — scatter of final-window distances

Usage
-----
  # Ensure the vLLM server is running, then:
  EMBED_HOST=<node> EMBED_PORT=8000 \
  python eval_semantics.py \
      --logs_dir    ./Agents/workflow_logs/<run_name> \
      --output_dir  ./Agents/eval_results/<run_name>/experiment_2 \
      --window_size 50
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
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist

try:
    import Levenshtein
except ImportError:
    raise ImportError("Please install the Levenshtein package: pip install python-Levenshtein")

# ==========================================
# vLLM Server Configuration
# ==========================================
EMBED_HOST = os.getenv("EMBED_HOST", "localhost")
EMBED_PORT = os.getenv("EMBED_PORT", "8000")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")


async def get_embeddings(texts: list, host: str, port: str, model: str) -> list:
    """Fetch embeddings from the vLLM server asynchronously."""
    if not texts:
        return []
    url     = f"http://{host}:{port}/v1/embeddings"
    payload = {"model": model, "input": texts}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_msg = await resp.text()
                print(f"\n[VLLM Error] Received status {resp.status}. Details: {error_msg}")
            resp.raise_for_status()
            data = await resp.json()

    # Sort by index to maintain original order
    ordered = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in ordered]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Compute the normalised Levenshtein distance between two strings."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return Levenshtein.distance(s1, s2) / max_len


def compute_lexical_distances(signals: list) -> np.ndarray:
    """
    Compute pairwise normalised-Levenshtein distances.
    """
    n = len(signals)
    distances = np.array([
        normalized_levenshtein(signals[i], signals[j])
        for i, j in combinations(range(n), 2)
    ])
    return distances


async def run_hybrid_topsim_analysis(
    logs_dir: str,
    output_dir: str,
    window_size: int = 50,
    max_samples_per_window: int = 150,
):
    """
    Analyzes the emergence of compositionality using Topological Similarity
    (TopSim). Computes both Lexical (Edit Distance) and Semantic (Embedding
    Distance) correlations between the meaning space and the signal space.

    Additionally, if noisy outputs are present, a second set of TopSim values
    is computed on the noisy signals to quantify robustness.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Auto-detect 'train' subfolder (Problem C) ─────────────────────────
    train_logs_dir = os.path.join(logs_dir, "train")
    if os.path.isdir(train_logs_dir):
        print(f"Auto-detected 'train' subfolder. Reading trajectories from: {train_logs_dir}")
        logs_dir = train_logs_dir

    # Load and sort log files by timestamp
    log_files = glob.glob(os.path.join(logs_dir, "traj_*.json"))
    parsed_files = []

    for f in log_files:
        try:
            ts = int(os.path.basename(f).split('_')[1])
            parsed_files.append((ts, f))
        except Exception:
            continue

    parsed_files.sort(key=lambda x: x[0])

    if not parsed_files:
        print(f"No log files found in: {logs_dir}")
        return

    print(f"Found {len(parsed_files)} trajectories. "
          f"Starting asynchronous windowed analysis...")

    results = []
    last_window_data = None  # stored for the final scatter plot

    for i in range(0, len(parsed_files), window_size):
        batch_files = parsed_files[i:i + window_size]

        # Subsample within the window to prevent O(N²) complexity explosion
        if len(batch_files) > max_samples_per_window:
            np.random.seed(42 + i)
            indices     = np.random.choice(len(batch_files), max_samples_per_window, replace=False)
            batch_files = [batch_files[idx] for idx in indices]

        meanings: list[str] = []      # Meaning Space (inputs)
        signals_clean: list[str] = [] # Clean Signal Space (intended)
        signals_noisy: list[str] = [] # Noisy Signal Space (after channel noise)

        for _, file_path in batch_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            try:
                prompt = data.get("prompt", "")
                paper_contexts: list[str] = []

                # Fallback for initial papers if stored in metadata
                for p in data.get("metadata", {}).get("papers", [])[:5]:
                    title   = p.get("title",   "")
                    summary = p.get("summary", p.get("abstract", ""))
                    if title or summary:
                        paper_contexts.append(f"{title}\n{summary}".strip())

                clean_parts: list[str] = []
                noisy_parts: list[str] = []

                # Extract dynamic contexts and communication signals from turns
                for turn in data.get("turns", []):
                    # Gather new context from Weaviate searches
                    if turn.get("turn_type") == "retriever_search":
                        search_results = turn.get("search_result") or turn.get("extra", {}).get("search_result", [])
                        for p in search_results[:5]:
                            title   = p.get("title",   "")
                            summary = p.get("summary", p.get("abstract", ""))
                            if title or summary:
                                paper_contexts.append(f"{title}\n{summary}".strip())

                    # Gather both clean and noisy Retriever outputs
                    elif turn.get("turn_type") == "retriever_message":
                        clean_text = turn.get("output_content") or turn.get("extra", {}).get("output_content", "")
                        noisy_text = turn.get("noisy_output_content") or turn.get("extra", {}).get("noisy_output_content", "")
                        
                        if clean_text:
                            clean_parts.append(clean_text)
                        if noisy_text:
                            noisy_parts.append(noisy_text)

                # Remove duplicates from paper_contexts to avoid over-weighting
                seen = set()
                unique_papers = []
                for p in paper_contexts:
                    if p not in seen:
                        seen.add(p)
                        unique_papers.append(p)

                meaning_space = prompt + "\n\n" + "\n\n".join(unique_papers)

                # Truncate to avoid vLLM token-length limits
                if meaning_space:
                    meaning_space = meaning_space[:12_000]

                # Signals: concatenated Retriever outputs from all message turns
                signal_clean_str = " ".join(clean_parts).strip()
                signal_noisy_str = " ".join(noisy_parts).strip()

                if signal_clean_str:
                    signal_clean_str = signal_clean_str[:12_000]
                if signal_noisy_str:
                    signal_noisy_str = signal_noisy_str[:12_000]

                # Only include trajectory if we have both meaning and at least clean signal
                if meaning_space and signal_clean_str:
                    meanings.append(meaning_space)
                    signals_clean.append(signal_clean_str)
                    # noisy signal might be missing if noise was off; use empty string
                    signals_noisy.append(signal_noisy_str if signal_noisy_str else signal_clean_str)

            except Exception as e:
                # Catch any unexpected parsing errors and skip the corrupted trajectory
                continue

        if len(meanings) < 10:
            print(f"  Window {i // window_size + 1}: Only {len(meanings)} samples — skipping.")
            continue

        print(f"  Window {i // window_size + 1}: "
              f"Fetching embeddings for {len(meanings)} samples...")

        # Asynchronous batch embedding for meanings and both signal types
        meaning_embeddings: list = []
        signal_clean_embs: list = []
        signal_noisy_embs: list = []
        batch_size = 50

        for b in range(0, len(meanings), batch_size):
            m_batch = meanings[b:b + batch_size]
            c_batch = signals_clean[b:b + batch_size]
            n_batch = signals_noisy[b:b + batch_size]

            m_embs = await get_embeddings(m_batch, EMBED_HOST, EMBED_PORT, EMBED_MODEL)
            c_embs = await get_embeddings(c_batch, EMBED_HOST, EMBED_PORT, EMBED_MODEL)
            n_embs = await get_embeddings(n_batch, EMBED_HOST, EMBED_PORT, EMBED_MODEL)

            meaning_embeddings.extend(m_embs)
            signal_clean_embs.extend(c_embs)
            signal_noisy_embs.extend(n_embs)

        # Pairwise distance matrices
        meaning_distances = pdist(meaning_embeddings, metric='cosine')
        signal_semantic_clean = pdist(signal_clean_embs, metric='cosine')
        signal_semantic_noisy = pdist(signal_noisy_embs, metric='cosine')
        signal_lexical_clean = compute_lexical_distances(signals_clean)
        signal_lexical_noisy = compute_lexical_distances(signals_noisy)

        # TopSim = Spearman ρ
        rho_sem_clean, _ = spearmanr(meaning_distances, signal_semantic_clean)
        rho_lex_clean, _ = spearmanr(meaning_distances, signal_lexical_clean)
        rho_sem_noisy, _ = spearmanr(meaning_distances, signal_semantic_noisy)
        rho_lex_noisy, _ = spearmanr(meaning_distances, signal_lexical_noisy)

        results.append({
            "Window_Index": i // window_size + 1,
            "Trajectories": len(meanings),
            "TopSim_Semantic_Rho_Clean": round(rho_sem_clean, 4),
            "TopSim_Lexical_Rho_Clean":   round(rho_lex_clean, 4),
            "TopSim_Semantic_Rho_Noisy": round(rho_sem_noisy, 4),
            "TopSim_Lexical_Rho_Noisy":   round(rho_lex_noisy, 4),
        })

        last_window_data = (
            meaning_distances,
            signal_semantic_clean,
            signal_semantic_noisy,
            signal_lexical_clean,
            signal_lexical_noisy,
            rho_sem_clean,
            rho_lex_clean,
            rho_sem_noisy,
            rho_lex_noisy,
        )

    if not results:
        print("No windows produced results. Check log files.")
        return

    # Save metrics to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp2_topsim_evolution.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved TopSim evolution data to: {csv_path}")

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

    ACCENT   = '#1a1a1a'
    GREY_MID = '#666666'
    GREY_REF = '#aaaaaa'

    # ------------------------------------------------------------------
    # Plot 1: TopSim evolution over time (clean + noisy)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df["Window_Index"], df["TopSim_Semantic_Rho_Clean"],
            color=ACCENT,   linewidth=1.2, label='Semantic TopSim (clean)')
    ax.plot(df["Window_Index"], df["TopSim_Lexical_Rho_Clean"],
            color=GREY_MID, linewidth=1.2, linestyle='--',
            label='Lexical TopSim (clean)')
    ax.plot(df["Window_Index"], df["TopSim_Semantic_Rho_Noisy"],
            color='#d62728', linewidth=1.0, linestyle=':',
            label='Semantic TopSim (noisy)')
    ax.plot(df["Window_Index"], df["TopSim_Lexical_Rho_Noisy"],
            color='#ff7f0e', linewidth=1.0, linestyle=':',
            label='Lexical TopSim (noisy)')
    ax.axhline(0, linewidth=0.6, color=GREY_REF, linestyle='-.')

    ax.set_title('Experiment 2 — Emergence of compositionality: TopSim over training',
                 pad=8)
    ax.set_xlabel('Training window')
    ax.set_ylabel('Spearman ρ')
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.grid(axis='y', linewidth=0.4, color=GREY_REF, linestyle=':')
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_topsim_evolution.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Scatter — final training state (semantic TopSim, clean)
    # ------------------------------------------------------------------
    if last_window_data:
        (m_dist, s_sem_clean, _, _, _, rho_sem_clean, *_) = last_window_data

        fig, ax = plt.subplots(figsize=(5, 5))

        plot_idx = np.random.choice(len(m_dist), min(5000, len(m_dist)), replace=False)
        x_pts = m_dist[plot_idx]
        y_pts = s_sem_clean[plot_idx]

        ax.scatter(x_pts, y_pts, s=6, alpha=0.12, color=ACCENT, linewidths=0)

        sort_order = np.argsort(x_pts)
        x_sorted   = x_pts[sort_order]
        m_coef, b_coef = np.polyfit(x_pts, y_pts, 1)
        ax.plot(x_sorted, m_coef * x_sorted + b_coef,
                color=ACCENT, linewidth=1.2)

        ax.set_title(
            f'Experiment 2 — Semantic TopSim (clean), final window\n'
            f'Spearman ρ = {rho_sem_clean:.3f}',
            pad=8,
        )
        ax.set_xlabel('Meaning-space distance (context)')
        ax.set_ylabel('Signal-space distance (messages)')
        ax.yaxis.set_label_coords(-0.12, 0.5)
        ax.grid(linewidth=0.4, color=GREY_REF, linestyle=':')

        fig.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "exp2_topsim_scatter_final.png"),
            dpi=300, bbox_inches='tight',
        )
        plt.close()

    print(f"Analysis completed successfully! Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 2 — Semantics: TopSim compositionality"
    )
    parser.add_argument("--logs_dir",    default="./Agents/workflow_logs",
                        help="Directory with traj_*.json debug logs")
    parser.add_argument("--output_dir",  default="./Agents/eval_results/experiment_2")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--embed_host",  default=EMBED_HOST,
                        help="Hostname of the vLLM embedding server")
    parser.add_argument("--embed_port",  type=int, default=EMBED_PORT,
                        help="Port of the vLLM embedding server")
    args = parser.parse_args()

    # Override global variables with command-line arguments
    EMBED_HOST = args.embed_host
    EMBED_PORT = args.embed_port

    asyncio.run(run_hybrid_topsim_analysis(
        args.logs_dir, args.output_dir, window_size=args.window_size
    ))