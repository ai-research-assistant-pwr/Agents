"""
Semantics Evaluation — Experiment 2
==========================================
Tests whether the Retriever's emergent language is compositional — i.e. whether
similar input contexts systematically produce similar output signals.

Compositionality is measured with Topological Similarity (TopSim):
  TopSim = Spearman ρ ( dist_meaning(i,j),  dist_signal(i,j) )

Changes vs v1
-------------
[FIX 1] Meaning space: prompt-only embedding (no papers)
    Previously: meaning_space = prompt + all_paper_summaries (up to 12 000 chars)
    Problem:    Two trajectories retrieving the same papers got near-zero meaning
                distance regardless of what the user actually asked.  Papers also
                dominate the string, so the embedding reflects paper content, not
                user intent.  This led to an artificially bimodal distance
                distribution (same-paper clusters vs. different-paper clusters)
                that inflated ρ.
    Fix:        meaning_space = prompt only.
                If you want a richer meaning signal, consider a weighted average:
                    0.7 * embed(prompt) + 0.3 * mean(embed(papers))
                but keep them separate so papers don't swamp intent.

[FIX 2] Noisy-signal fallback removed; noisy window skipped when noise is absent
    Previously: signals_noisy.append(signal_noisy_str if signal_noisy_str else signal_clean_str)
    Problem:    When APPLY_CHANNEL_NOISE=false the fallback fills noisy with clean,
                so ρ_noisy ≡ ρ_clean and the plot shows four indistinguishable lines.
    Fix:        Track per-trajectory whether a noisy signal exists.  Only compute
                noisy TopSim if ≥ 10 trajectories in the window actually have a
                noisy_output_content field.  Otherwise report NaN and skip plotting
                those series entirely — a flat line at ρ=0.83 is more misleading
                than a missing series.

[FIX 3] Lexical distance: Jaccard on word sets instead of character-level Levenshtein
    Previously: Levenshtein.distance(s1, s2) / max(len(s1), len(s2))
    Problem:    Character-level edit distance on 400–900-token LLM outputs is
                effectively meaningless — a single rephrasing of a sentence changes
                hundreds of characters even if semantics are identical.  This makes
                lexical TopSim measure surface noise, not structural similarity.
    Fix:        Jaccard distance on unigram bag-of-words:
                    J(A,B) = 1 − |A∩B| / |A∪B|
                This captures lexical overlap directly, is order-insensitive (which
                is appropriate for messages that may reorganise content), and is
                O(|vocab|) not O(|string|²).
                If you want to preserve some word-order sensitivity, replace with
                soft cosine on tf-idf vectors (see commented alternative below).

[FIX 4] Incomplete window handling consistent with eval_morphology.py
    Previously: incomplete last window was processed normally.
    Problem:    A window with 20 trajectories instead of 50 has a smaller sample
                from the joint distance distribution, which can spuriously raise or
                lower ρ.  eval_morphology already skips incomplete windows.
    Fix:        Skip windows where len(batch) < window_size (same as morphology).

Usage
-----
  EMBED_HOST=<node> EMBED_PORT=8000 \\
  python eval_semantics.py \\
      --logs_dir    ./Agents/workflow_logs/<run_name>/train \\
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
# Jaccard lexical distance
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Word-level tokenisation — returns a set (bag-of-words)."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def jaccard_distance(s1: str, s2: str) -> float:
    """
    Jaccard distance on unigram word sets: 1 − |A∩B|/|A∪B|.
    Returns 1.0 for completely disjoint vocabularies, 0.0 for identical.
    """
    a, b = _tokenize(s1), _tokenize(s2)
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def compute_lexical_distances(signals: list) -> np.ndarray:
    """Pairwise Jaccard distances for a list of signal strings."""
    n = len(signals)
    return np.array([
        jaccard_distance(signals[i], signals[j])
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

        # Skip incomplete windows — keeps sample size uniform across windows
        if len(batch_files) < window_size:
            print(f"  Window {window_idx}: Skipping incomplete window "
                  f"({len(batch_files)}/{window_size}).")
            continue

        # Subsample within the window to prevent O(N²) complexity explosion
        if len(batch_files) > max_samples_per_window:
            np.random.seed(42 + i)
            idx         = np.random.choice(len(batch_files), max_samples_per_window, replace=False)
            batch_files = [batch_files[k] for k in idx]

        meanings:       list[str] = []
        signals_clean:  list[str] = []
        signals_noisy:  list[str] = []   # only appended when noisy actually exists
        noisy_count:    int       = 0    # trajectories with real noisy signal
        seen_prompts:   set       = set() # <--- DODANE: Zbiór do śledzenia unikalnych promptów

        for _, file_path in batch_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            try:
                prompt = data.get("prompt", "")
                if not prompt or prompt in seen_prompts:
                    continue
                seen_prompts.add(prompt)

                meaning_space = prompt[:2_000]

                clean_parts: list[str] = []
                noisy_parts: list[str] = []

                for turn in data.get("turns", []):
                    if turn.get("turn_type") == "retriever_message":
                        clean_text = (
                            turn.get("output_content")
                            or turn.get("extra", {}).get("output_content", "")
                        )
                        noisy_text = (
                            turn.get("noisy_output_content")
                            or turn.get("extra", {}).get("noisy_output_content", "")
                        )
                        if clean_text:
                            clean_parts.append(clean_text)
                        if noisy_text:
                            noisy_parts.append(noisy_text)

                signal_clean_str = " ".join(clean_parts).strip()[:12_000]
                signal_noisy_str = " ".join(noisy_parts).strip()

                if not signal_clean_str:
                    continue

                meanings.append(meaning_space)
                signals_clean.append(signal_clean_str)

                if signal_noisy_str:
                    signals_noisy.append(signal_noisy_str[:12_000])
                    noisy_count += 1
                else:
                    signals_noisy.append("")   # sentinel — excluded below

            except Exception:
                continue

        if len(meanings) < 10:
            print(f"  Window {window_idx}: Only {len(meanings)} valid samples — skipping.")
            continue

        print(f"  Window {window_idx}: {len(meanings)} samples "
              f"({noisy_count} with noisy signal). Fetching embeddings…")

        # Embed meanings and clean signals
        all_mean_embs:  list = []
        all_clean_embs: list = []
        BATCH = 50

        for b in range(0, len(meanings), BATCH):
            m_embs = await get_embeddings(meanings[b:b+BATCH],       EMBED_HOST, EMBED_PORT, EMBED_MODEL)
            c_embs = await get_embeddings(signals_clean[b:b+BATCH],  EMBED_HOST, EMBED_PORT, EMBED_MODEL)
            all_mean_embs.extend(m_embs)
            all_clean_embs.extend(c_embs)

        meaning_distances      = pdist(all_mean_embs,  metric='cosine')
        signal_semantic_clean  = pdist(all_clean_embs, metric='cosine')
        signal_lexical_clean   = compute_lexical_distances(signals_clean)

        rho_sem_clean, p_sem_clean = spearmanr(meaning_distances, signal_semantic_clean)
        rho_lex_clean, p_lex_clean = spearmanr(meaning_distances, signal_lexical_clean)

        row = {
            "Window_Index":              window_idx,
            "Trajectories":              len(meanings),
            "Noisy_Trajectories":        noisy_count,
            "TopSim_Semantic_Rho_Clean": round(rho_sem_clean, 4),
            "TopSim_Semantic_P_Clean":   round(p_sem_clean,   6),
            "TopSim_Lexical_Rho_Clean":  round(rho_lex_clean, 4),
            "TopSim_Lexical_P_Clean":    round(p_lex_clean,   6),
            "TopSim_Semantic_Rho_Noisy": float("nan"),
            "TopSim_Lexical_Rho_Noisy":  float("nan"),
        }

        # Only compute noisy metrics when enough real noisy signals exist.
        noisy_fraction = noisy_count / len(meanings)
        if noisy_fraction >= min_noisy_fraction:
            # Filter to trajectories that actually have noisy signals
            valid_idx = [k for k, s in enumerate(signals_noisy) if s]
            m_valid   = [all_mean_embs[k]  for k in valid_idx]
            n_valid   = [signals_noisy[k]  for k in valid_idx]

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

    # -----------------------------------------------------------------------
    # Diagnostic: print distance distribution stats to flag bimodality
    # -----------------------------------------------------------------------
    if last_window_data:
        md = last_window_data["meaning_distances"]
        print(f"\n[Diagnostic] Final-window meaning-space distances:")
        print(f"  min={md.min():.3f}  median={np.median(md):.3f}  "
              f"max={md.max():.3f}  std={md.std():.3f}")
        q25, q75 = np.percentile(md, [25, 75])
        print(f"  25th pct={q25:.3f}  75th pct={q75:.3f}")
        if md.std() > 0.28 and q25 < 0.05:
            print("  ⚠  Distribution looks bimodal (large std, low 25th pct).")
            print("     Consider filtering to diverse prompt pairs only, or stratifying.")

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
            label='Lexical TopSim / Jaccard (clean)')

    if has_noisy:
        ax.plot(x, df["TopSim_Semantic_Rho_Noisy"], color='#d62728', lw=1.0, ls=':',
                label='Semantic TopSim (noisy)')
        ax.plot(x, df["TopSim_Lexical_Rho_Noisy"],  color='#ff7f0e', lw=1.0, ls=':',
                label='Lexical TopSim / Jaccard (noisy)')
    else:
        ax.text(0.5, 0.04,
                "Noisy series not shown — run with APPLY_CHANNEL_NOISE=true",
                transform=ax.transAxes, ha='center', fontsize=8, color=GREY_MID,
                style='italic')

    ax.axhline(0, lw=0.6, color=GREY_REF, ls='-.')
    ax.set_title('Experiment 2 — Emergence of compositionality: TopSim over training\n'
                 '(meaning space = prompt only; lexical = Jaccard)', pad=8)
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
        ax_s.set_xlabel('Meaning-space distance (prompt cosine)')
        ax_s.set_ylabel('Signal-space distance (message cosine)')
        ax_s.grid(lw=0.4, color=GREY_REF, ls=':')

        # Right: meaning-distance histogram — reveals bimodality if present
        ax_h = axes[1]
        ax_h.hist(m_dist, bins=60, color=ACCENT, alpha=0.7, linewidth=0)
        ax_h.set_title('Meaning-space distance distribution\n(prompt only)', pad=8)
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
        description="Experiment 2 — Semantics: TopSim compositionality (v2)"
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