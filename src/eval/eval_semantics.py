"""
Semantics Evaluation — Experiment 2
==========================================
Tests whether the Retriever's emergent language is compositional — i.e. whether
similar input contexts systematically produce similar output signals.

Compositionality is measured with Topological Similarity (TopSim):
  TopSim = Spearman ρ ( dist_meaning(i,j),  dist_signal(i,j) )

Refactored Features:
1. Hybrid Meaning Space: Computes separate embeddings for the prompt and the retrieved 
   documents, combining them (0.7 * prompt + 0.3 * docs) to capture both user intent 
   and actual context without either dominating the vector.
2. Syntactic Lexical Distance: Replaced Jaccard (BoW) with Token Edit Distance 
   (Word-level Levenshtein) to accurately measure structural/compositional changes.
3. Corrected Subsampling: Filters for unique, valid trajectories *before* downsampling.

Usage
-----
  EMBED_HOST=<node> EMBED_PORT=8000 \\
  python eval_semantics.py \\
      --logs_dir    ./Agents/workflow_logs/<run_name> \\
      --output_dir  ./Agents/eval_results/<run_name>/experiment_2 \\
      --window_size 50
"""

import os
import json
import glob
import asyncio
import aiohttp
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize

# ==========================================
# vLLM Server Configuration
# ==========================================
EMBED_HOST  = os.getenv("EMBED_HOST",  "localhost")
EMBED_PORT  = os.getenv("EMBED_PORT",  "8000")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

async def get_embeddings(texts: list, host: str, port: str, model: str) -> list:
    """Fetch embeddings from the vLLM server asynchronously."""
    if not texts:
        return []
    url     = f"http://{host}:{port}/v1/embeddings"
    payload = {"model": model, "input": texts}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                print(f"\n[VLLM Error] Status {resp.status}: {await resp.text()}")
            resp.raise_for_status()
            data = await resp.json()

    ordered = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in ordered]


# ---------------------------------------------------------------------------
# Meaning extraction helpers
# ---------------------------------------------------------------------------

def _get_turn_field(turn: dict, field: str, default=None):
    extra = turn.get("extra", {})
    if field in extra:
        return extra[field]
    return turn.get(field, default)

def _extract_papers_text(data: dict) -> str:
    """Extract and concatenate document abstracts to form the context string."""
    papers = (data.get("metadata") or {}).get("papers", [])[:5]
    paper_contexts = []
    
    for p in papers:
        title   = p.get("title", "")
        summary = p.get("summary", p.get("abstract", ""))
        if title or summary:
            paper_contexts.append(f"{title}\n{summary}".strip())
            
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
            
    # Return context string, bounded to avoid excessive length
    return "\n\n".join(unique_papers)[:10_000]


# ---------------------------------------------------------------------------
# Token Edit Distance (Word-level Levenshtein)
# ---------------------------------------------------------------------------

def token_edit_distance(s1: str, s2: str) -> float:
    """
    Computes normalized Levenshtein distance at the word level.
    Unlike character-level edit distance, this captures structural compositionality.
    Returns 0.0 (identical) to 1.0 (completely disjoint/different).
    """
    tok1 = re.findall(r'\b\w+\b', s1.lower())
    tok2 = re.findall(r'\b\w+\b', s2.lower())
    
    if not tok1 and not tok2:
        return 0.0
    if not tok1 or not tok2:
        return 1.0

    m, n = len(tok1), len(tok2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if tok1[i-1] == tok2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,      # deletion
                           dp[i][j-1] + 1,      # insertion
                           dp[i-1][j-1] + cost) # substitution
                           
    max_len = max(m, n)
    return dp[m][n] / max_len if max_len > 0 else 0.0

def compute_lexical_distances(signals: list) -> np.ndarray:
    """Pairwise Token Edit Distances for a list of signal strings."""
    n = len(signals)
    return np.array([
        token_edit_distance(signals[i], signals[j])
        for i, j in combinations(range(n), 2)
    ])


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

async def run_hybrid_topsim_analysis(
    logs_dir: str,
    output_dir: str,
    window_size: int = 50,
    max_samples_per_window: int = 150,
    min_noisy_fraction: float = 0.5,
):
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect 'train' subfolder
    train_logs_dir = os.path.join(logs_dir, "train")
    if os.path.isdir(train_logs_dir):
        print(f"Auto-detected 'train' subfolder → {train_logs_dir}")
        logs_dir = train_logs_dir

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

    print(f"Found {len(parsed_files)} trajectories.")

    results       = []
    last_window_data = None

    for i in range(0, len(parsed_files), window_size):
        batch_files = parsed_files[i:i + window_size]
        window_idx  = i // window_size + 1

        if len(batch_files) < window_size:
            print(f"  Window {window_idx}: Skipping incomplete window "
                  f"({len(batch_files)}/{window_size}).")
            continue

        # Extract all valid data first, *before* subsampling
        valid_trajectories = []
        seen_prompts = set()

        for _, file_path in batch_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            try:
                prompt = data.get("prompt", "")
                if not prompt or prompt in seen_prompts:
                    continue
                seen_prompts.add(prompt)

                papers_text = _extract_papers_text(data)

                clean_parts: list[str] = []
                noisy_parts: list[str] = []

                for turn in data.get("turns", []):
                    if _get_turn_field(turn, "turn_type") == "retriever_message":
                        clean_text = _get_turn_field(turn, "output_content", "")
                        noisy_text = _get_turn_field(turn, "noisy_output_content", "")
                        
                        if clean_text: clean_parts.append(clean_text)
                        if noisy_text: noisy_parts.append(noisy_text)

                signal_clean_str = " ".join(clean_parts).strip()[:12_000]
                signal_noisy_str = " ".join(noisy_parts).strip()

                if not signal_clean_str:
                    continue
                
                # Use "EMPTY_DOCS" if no docs found to avoid empty embedding queries
                if not papers_text.strip():
                    papers_text = "EMPTY_DOCS"

                valid_trajectories.append({
                    "prompt": prompt[:2_000],
                    "papers": papers_text,
                    "signal_clean": signal_clean_str,
                    "signal_noisy": signal_noisy_str
                })

            except Exception:
                continue

        # Subsample *after* filtering
        if len(valid_trajectories) > max_samples_per_window:
            np.random.seed(42 + i)
            idx = np.random.choice(len(valid_trajectories), max_samples_per_window, replace=False)
            valid_trajectories = [valid_trajectories[k] for k in idx]

        if len(valid_trajectories) < 10:
            print(f"  Window {window_idx}: Only {len(valid_trajectories)} valid samples — skipping.")
            continue

        noisy_count = sum(1 for t in valid_trajectories if t["signal_noisy"])
        print(f"  Window {window_idx}: {len(valid_trajectories)} samples "
              f"({noisy_count} with noisy signal). Fetching embeddings…")

        meanings_prompt = [t["prompt"] for t in valid_trajectories]
        meanings_papers = [t["papers"] for t in valid_trajectories]
        signals_clean   = [t["signal_clean"] for t in valid_trajectories]
        signals_noisy   = [t["signal_noisy"] for t in valid_trajectories]

        # Embed meanings (both prompt and papers separately) and clean signals
        all_prompt_embs: list = []
        all_papers_embs: list = []
        all_clean_embs:  list = []
        BATCH = 50

        for b in range(0, len(valid_trajectories), BATCH):
            p_embs = await get_embeddings(meanings_prompt[b:b+BATCH], EMBED_HOST, EMBED_PORT, EMBED_MODEL)
            d_embs = await get_embeddings(meanings_papers[b:b+BATCH], EMBED_HOST, EMBED_PORT, EMBED_MODEL)
            c_embs = await get_embeddings(signals_clean[b:b+BATCH],   EMBED_HOST, EMBED_PORT, EMBED_MODEL)
            all_prompt_embs.extend(p_embs)
            all_papers_embs.extend(d_embs)
            all_clean_embs.extend(c_embs)

        # Normalize components before addition
        p_embs_norm = normalize(np.array(all_prompt_embs))
        d_embs_norm = normalize(np.array(all_papers_embs))
        
        # Build Hybrid Meaning Space: 70% User Intent, 30% Context Docs
        hybrid_meaning_embs = 0.7 * p_embs_norm + 0.3 * d_embs_norm
        hybrid_meaning_embs = normalize(hybrid_meaning_embs) # Re-normalize the hybrid vector

        meaning_distances      = pdist(hybrid_meaning_embs,  metric='cosine')
        signal_semantic_clean  = pdist(all_clean_embs,       metric='cosine')
        signal_lexical_clean   = compute_lexical_distances(signals_clean)

        rho_sem_clean, p_sem_clean = spearmanr(meaning_distances, signal_semantic_clean)
        rho_lex_clean, p_lex_clean = spearmanr(meaning_distances, signal_lexical_clean)

        row = {
            "Window_Index":              window_idx,
            "Trajectories":              len(valid_trajectories),
            "Noisy_Trajectories":        noisy_count,
            "TopSim_Semantic_Rho_Clean": round(rho_sem_clean, 4),
            "TopSim_Semantic_P_Clean":   round(p_sem_clean,   6),
            "TopSim_Lexical_Rho_Clean":  round(rho_lex_clean, 4),
            "TopSim_Lexical_P_Clean":    round(p_lex_clean,   6),
            "TopSim_Semantic_Rho_Noisy": float("nan"),
            "TopSim_Lexical_Rho_Noisy":  float("nan"),
        }

        # Only compute noisy metrics when enough real noisy signals exist.
        noisy_fraction = noisy_count / len(valid_trajectories)
        if noisy_fraction >= min_noisy_fraction:
            # Filter to trajectories that actually have noisy signals
            valid_idx = [k for k, s in enumerate(signals_noisy) if s]
            m_valid   = [hybrid_meaning_embs[k] for k in valid_idx]
            n_valid   = [signals_noisy[k]       for k in valid_idx]

            all_noisy_embs: list = []
            for b in range(0, len(n_valid), BATCH):
                n_embs = await get_embeddings(n_valid[b:b+BATCH], EMBED_HOST, EMBED_PORT, EMBED_MODEL)
                all_noisy_embs.extend(n_embs)

            m_dist_noisy   = pdist(m_valid,         metric='cosine')
            s_sem_noisy    = pdist(all_noisy_embs,  metric='cosine')
            s_lex_noisy    = compute_lexical_distances(n_valid)

            rho_sem_noisy, _ = spearmanr(m_dist_noisy, s_sem_noisy)
            rho_lex_noisy, _ = spearmanr(m_dist_noisy, s_lex_noisy)
            row["TopSim_Semantic_Rho_Noisy"] = round(rho_sem_noisy, 4)
            row["TopSim_Lexical_Rho_Noisy"]  = round(rho_lex_noisy, 4)
        else:
            print(f"    → Only {noisy_fraction:.0%} noisy signals in window — "
                  f"noisy TopSim skipped (set APPLY_CHANNEL_NOISE=true to enable).")

        results.append(row)

        last_window_data = {
            "meaning_distances":     meaning_distances,
            "signal_semantic_clean": signal_semantic_clean,
            "signal_lexical_clean":  signal_lexical_clean,
            "rho_sem_clean":         rho_sem_clean,
            "rho_lex_clean":         rho_lex_clean,
        }

    if not results:
        print("No windows produced results.")
        return

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp2_topsim_evolution.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved evolution data → {csv_path}")

    if last_window_data:
        md = last_window_data["meaning_distances"]
        print(f"\n[Diagnostic] Final-window hybrid meaning-space distances:")
        print(f"  min={md.min():.3f}  median={np.median(md):.3f}  "
              f"max={md.max():.3f}  std={md.std():.3f}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
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
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
    })

    ACCENT   = '#1a1a1a'
    GREY_MID = '#666666'
    GREY_REF = '#aaaaaa'

    x = df["Window_Index"]
    has_noisy = df["TopSim_Semantic_Rho_Noisy"].notna().any()

    # Plot 1: TopSim evolution
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, df["TopSim_Semantic_Rho_Clean"], color=ACCENT,   lw=1.2,
            label='Semantic TopSim (clean)')
    ax.plot(x, df["TopSim_Lexical_Rho_Clean"],  color=GREY_MID, lw=1.2, ls='--',
            label='Lexical TopSim / Token Edit (clean)')

    if has_noisy:
        ax.plot(x, df["TopSim_Semantic_Rho_Noisy"], color='#d62728', lw=1.0, ls=':',
                label='Semantic TopSim (noisy)')
        ax.plot(x, df["TopSim_Lexical_Rho_Noisy"],  color='#ff7f0e', lw=1.0, ls=':',
                label='Lexical TopSim / Token Edit (noisy)')
    else:
        ax.text(0.5, 0.04,
                "Noisy series not shown — run with APPLY_CHANNEL_NOISE=true",
                transform=ax.transAxes, ha='center', fontsize=8, color=GREY_MID,
                style='italic')

    ax.axhline(0, lw=0.6, color=GREY_REF, ls='-.')
    ax.set_title('Experiment 2 — Emergence of compositionality: TopSim over training\n'
                 '(meaning space = hybrid 70/30; lexical = Token Edit Distance)', pad=8)
    ax.set_xlabel('Training window')
    ax.set_ylabel('Spearman ρ')
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.grid(axis='y', lw=0.4, color=GREY_REF, ls=':')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_topsim_evolution.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved exp2_topsim_evolution.png")

    # Plot 2: Scatter (final window, semantic clean) + distance histogram
    if last_window_data:
        m_dist = last_window_data["meaning_distances"]
        s_dist = last_window_data["signal_semantic_clean"]
        rho    = last_window_data["rho_sem_clean"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Left: scatter
        ax_s = axes[0]
        idx  = np.random.choice(len(m_dist), min(5000, len(m_dist)), replace=False)
        ax_s.scatter(m_dist[idx], s_dist[idx],
                     s=6, alpha=0.12, color=ACCENT, linewidths=0)
        x_sorted = np.sort(m_dist[idx])
        m_c, b_c = np.polyfit(m_dist[idx], s_dist[idx], 1)
        ax_s.plot(x_sorted, m_c * x_sorted + b_c, color=ACCENT, lw=1.2)
        ax_s.set_title(f'Semantic TopSim (clean) — final window\nSpearman ρ = {rho:.3f}',
                       pad=8)
        ax_s.set_xlabel('Meaning-space distance (hybrid cosine)')
        ax_s.set_ylabel('Signal-space distance (message cosine)')
        ax_s.grid(lw=0.4, color=GREY_REF, ls=':')

        # Right: meaning-distance histogram
        ax_h = axes[1]
        ax_h.hist(m_dist, bins=60, color=ACCENT, alpha=0.7, linewidth=0)
        ax_h.set_title('Hybrid Meaning-space distance distribution', pad=8)
        ax_h.set_xlabel('Pairwise cosine distance')
        ax_h.set_ylabel('Pair count')
        ax_h.grid(axis='y', lw=0.4, color=GREY_REF, ls=':')
        ax_h.text(0.97, 0.95,
                  f"std={m_dist.std():.3f}\nmedian={np.median(m_dist):.3f}",
                  transform=ax_h.transAxes, ha='right', va='top',
                  fontsize=8, color=GREY_MID)

        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, "exp2_topsim_scatter_final.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved exp2_topsim_scatter_final.png")

    print(f"\nDone. All results in: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 2 — Semantics: TopSim compositionality (v3)"
    )
    parser.add_argument("--logs_dir",    default="./Agents/workflow_logs")
    parser.add_argument("--output_dir",  default="./Agents/eval_results/experiment_2")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--embed_host",  default=EMBED_HOST)
    parser.add_argument("--embed_port",  type=int, default=int(EMBED_PORT))
    parser.add_argument("--min_noisy_fraction", type=float, default=0.5,
                        help="Min fraction of trajectories with noisy signal to compute ρ_noisy")
    args = parser.parse_args()

    EMBED_HOST = args.embed_host
    EMBED_PORT = str(args.embed_port)

    asyncio.run(run_hybrid_topsim_analysis(
        logs_dir              = args.logs_dir,
        output_dir            = args.output_dir,
        window_size           = args.window_size,
        min_noisy_fraction    = args.min_noisy_fraction,
    ))