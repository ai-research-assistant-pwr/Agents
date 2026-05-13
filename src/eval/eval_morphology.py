"""
Morphology Evaluation — Experiment 1
======================================
Tests the Information Bottleneck hypothesis: does adding a per-token length
penalty (R = R_task − λ · length(message)) cause the Retriever to compress
its messages over training, and does that compression correlate with task
performance?

Three windowed metrics are tracked over the course of training:

  Average Message Length
      Mean number of word tokens in Retriever outputs (turn_type == "retriever_message").
      A downward trend indicates the Retriever is learning to compress.

  Active Vocabulary Size
      Number of unique word types seen across all Retriever messages within
      a window.  Shrinking vocabulary suggests pruning of low-information
      tokens in favour of a more focused lexicon.

  Unigram Entropy (raw + normalised)
      Shannon entropy H over the unigram token distribution within a window.
        H_norm = H / log₂(V)   (V = vocabulary size, window-independent)
      A decrease in H_norm signals that the distribution is becoming more
      peaked — the Retriever is reusing a smaller set of high-value tokens.

Data contract
-------------
  Reads traj_*.json files written by scientific_workflow.py (DEBUG=True).
  Each file must contain a "turns" list; Retriever communication turns are 
  identified by "turn_type" == "retriever_message".

Output
------
  eval_results/experiment_1/
    ├── exp1_morphology_metrics.csv
    ├── exp1_compression_vs_reward.png   — reward + length over time
    └── exp1_entropy_and_vocab.png       — entropy (raw + norm) + vocab size

Usage
-----
  python eval_morphology.py \
      --logs_dir    ./Agents/workflow_logs/<run_name> \
      --output_dir  ./Agents/eval_results/<run_name>/experiment_1 \
      --window_size 50
"""

import os
import glob
import json
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def tokenize(text: str) -> list:
    """Simple tokenization on word level using regex."""
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())

def run_morphology_analysis(logs_dir: str, output_dir: str, window_size: int = 50):
    os.makedirs(output_dir, exist_ok=True)

    # Load and sort log files by timestamp
    log_files = glob.glob(os.path.join(logs_dir, "traj_*.json"))

    parsed_files = []
    for f in log_files:
        basename = os.path.basename(f)
        try:
            # Extract timestamp from format: traj_{ts_ms}_{uid}.json
            ts = int(basename.split('_')[1])
            parsed_files.append((ts, f))
        except (IndexError, ValueError):
            continue

    parsed_files.sort(key=lambda x: x[0])

    if not parsed_files:
        print(f"No logs found in directory: {logs_dir}")
        return

    print(f"Found {len(parsed_files)} trajectories. Analyzing in windows of {window_size}...")

    # --- Diagnostic: check the first file for correct turn types ---
    with open(parsed_files[0][1], 'r', encoding='utf-8') as f:
        _sample_data = json.load(f)
    _sample_turns = _sample_data.get("turns", [])
    
    retriever_msgs_in_sample = [t for t in _sample_turns if t.get("turn_type") == "retriever_message"]
    print(f"[Diagnostic] First file has {len(_sample_turns)} turns total, "
          f"{len(retriever_msgs_in_sample)} are retriever_message turns.")
    if not retriever_msgs_in_sample:
        print("[WARNING] No retriever_message turns found! Ensure logs are from the new environment.")

    results = []

    # Process logs in windows to compute metrics over time
    for i in range(0, len(parsed_files), window_size):
        batch_files = parsed_files[i:i + window_size]

        # Skip incomplete windows to prevent entropy from dropping at the end of training due to smaller sample size, not compression.
        if len(batch_files) < window_size:
            print(f"  Window {i // window_size + 1}: Skipping incomplete window "
                  f"({len(batch_files)}/{window_size} trajectories).")
            continue

        window_rewards = []
        window_message_lengths = []
        window_vocab = Counter()

        for _, file_path in batch_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            window_rewards.append(data.get("total_reward", 0.0))
            traj_lengths = []

            for turn in data.get("turns", []):
                # In the new environment, we strictly analyze the communication channel
                # (retriever_message), ignoring Weaviate search queries (retriever_search).
                if turn.get("turn_type") == "retriever_message":
                    text = turn.get("output_content", "")
                    tokens = tokenize(text)
                    traj_lengths.append(len(tokens))
                    window_vocab.update(tokens)

            # Average retriever message length for this specific trajectory
            if traj_lengths:
                window_message_lengths.append(sum(traj_lengths) / len(traj_lengths))

        if not window_message_lengths:
            print(f"  Window {i // window_size + 1}: No retriever messages found — skipping.")
            continue

        # Calculate window metrics
        avg_reward = np.mean(window_rewards)
        avg_length = np.mean(window_message_lengths)
        vocab_size = len(window_vocab)

        # Calculate Token Distribution Entropy for the window.
        counts = list(window_vocab.values())
        if counts:
            probs = np.array(counts) / sum(counts)
            h = entropy(probs, base=2)
            # Normalised entropy: 1.0 = perfectly uniform, 0.0 = single token
            h_norm = h / np.log2(vocab_size) if vocab_size > 1 else 0.0
        else:
            h = 0.0
            h_norm = 0.0

        results.append({
            "Window_Index": i // window_size + 1,
            "Trajectories_Count": len(batch_files),
            "Avg_Reward": round(avg_reward, 4),
            "Avg_Message_Length": round(avg_length, 2),
            "Active_Vocab_Size": vocab_size,
            "Unigram_Entropy": round(h, 4),
            "Unigram_Entropy_Norm": round(h_norm, 4),
        })

    if not results:
        print("No complete windows produced results. Check log files and turn types.")
        return

    # Save metrics to CSV for further analysis
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp1_morphology_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw windowed data to: {csv_path}")

    # ==========================================
    # Plot Generation
    # ==========================================

    plt.rcParams.update({
        'font.family':      'serif',
        'font.size':        10,
        'axes.labelsize':   10,
        'axes.titlesize':   11,
        'legend.fontsize':   9,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'axes.spines.top':  False,
        'axes.spines.right': False,
        'axes.linewidth':   0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'figure.facecolor': 'white',
        'axes.facecolor':   'white',
    })

    # Single accent colour; all other series in neutral greys
    ACCENT   = '#1a1a1a'   # near-black for the primary series
    GREY_MID = '#666666'   # secondary series
    GREY_REF = '#aaaaaa'   # reference / zero lines
    FILL_A   = '#1a1a1a'   # fill matching accent, low alpha below

    x_axis = df["Window_Index"]

    # ------------------------------------------------------------------
    # Plot 1: Compression vs. Task Reward  (2 rows, shared x-axis)
    # ------------------------------------------------------------------
    fig1, (ax_r, ax_l) = plt.subplots(
        2, 1, figsize=(7, 5), sharex=True,
        gridspec_kw={'hspace': 0.12},
    )

    ax_r.plot(x_axis, df["Avg_Reward"], color=ACCENT, linewidth=1.2)
    ax_r.fill_between(x_axis, df["Avg_Reward"], alpha=0.06, color=ACCENT)
    ax_r.set_ylabel('Task reward')
    ax_r.yaxis.set_label_coords(-0.08, 0.5)
    ax_r.grid(axis='y', linewidth=0.4, color=GREY_REF, linestyle=':')
    ax_r.set_title('Experiment 1 — Information Bottleneck: compression vs. task reward',
                   pad=8, fontsize=11)

    ax_l.plot(x_axis, df["Avg_Message_Length"], color=ACCENT, linewidth=1.2)
    ax_l.fill_between(x_axis, df["Avg_Message_Length"], alpha=0.06, color=ACCENT)
    ax_l.set_ylabel('Retriever tokens')
    ax_l.yaxis.set_label_coords(-0.08, 0.5)
    ax_l.set_xlabel('Training window')
    ax_l.grid(axis='y', linewidth=0.4, color=GREY_REF, linestyle=':')

    fig1.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp1_compression_vs_reward.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Entropy (raw + normalised) and Vocabulary Size  (3 rows)
    # ------------------------------------------------------------------
    fig2, (ax_e, ax_en, ax_v) = plt.subplots(
        3, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={'hspace': 0.12},
    )

    ax_e.plot(x_axis, df["Unigram_Entropy"], color=ACCENT, linewidth=1.2)
    ax_e.fill_between(x_axis, df["Unigram_Entropy"], alpha=0.06, color=ACCENT)
    ax_e.set_ylabel('Entropy (bits)')
    ax_e.yaxis.set_label_coords(-0.08, 0.5)
    ax_e.grid(axis='y', linewidth=0.4, color=GREY_REF, linestyle=':')
    ax_e.set_title(
        f'Experiment 1 — Token distribution entropy and vocabulary size\n'
        f'(window = {window_size} trajectories)',
        pad=8, fontsize=11,
    )

    ax_en.plot(x_axis, df["Unigram_Entropy_Norm"], color=ACCENT, linewidth=1.2)
    ax_en.fill_between(x_axis, df["Unigram_Entropy_Norm"], alpha=0.06, color=ACCENT)
    ax_en.set_ylabel('H / log₂(V)')
    ax_en.yaxis.set_label_coords(-0.08, 0.5)
    ax_en.set_ylim(0, 1.05)
    ax_en.axhline(1.0, linewidth=0.5, color=GREY_REF, linestyle=':')
    ax_en.grid(axis='y', linewidth=0.4, color=GREY_REF, linestyle=':')

    ax_v.plot(x_axis, df["Active_Vocab_Size"], color=ACCENT, linewidth=1.2)
    ax_v.fill_between(x_axis, df["Active_Vocab_Size"], alpha=0.06, color=ACCENT)
    ax_v.set_ylabel('Vocabulary size')
    ax_v.yaxis.set_label_coords(-0.08, 0.5)
    ax_v.set_xlabel('Training window')
    ax_v.grid(axis='y', linewidth=0.4, color=GREY_REF, linestyle=':')

    fig2.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp1_entropy_and_vocab.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Completed analysis. High-resolution plots saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 1 — Morphology: compression vs. usefulness"
    )
    parser.add_argument("--logs_dir",    default="./Agents/workflow_logs",
                        help="Directory with traj_*.json debug logs")
    parser.add_argument("--output_dir",  default="./Agents/eval_results/experiment_1")
    parser.add_argument("--window_size", type=int, default=50)
    args = parser.parse_args()

    run_morphology_analysis(args.logs_dir, args.output_dir, window_size=args.window_size)