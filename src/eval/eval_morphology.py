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

def _detect_role_key(turns: list) -> str:
    """
    Detects whether the role key in the turn records is 'agent_role' or 'role'.
    """
    if not turns:
        return "role"
    sample = turns[0]
    if "agent_role" in sample:
        return "agent_role"
    if "role" in sample:
        return "role"
    raise KeyError(
        f"Cannot find role key in turn record. Available keys: {list(sample.keys())}"
    )


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

    # --- Diagnostic: detect role key from the first file ---
    with open(parsed_files[0][1], 'r', encoding='utf-8') as f:
        _sample_data = json.load(f)
    _sample_turns = _sample_data.get("turns", [])
    role_key = _detect_role_key(_sample_turns)
    print(f"[Diagnostic] Using role key: '{role_key}'")

    # --- Diagnostic: show how many retriever turns are found in first file ---
    retriever_turns_in_sample = [t for t in _sample_turns if t.get(role_key) == "retriever"]
    print(f"[Diagnostic] First file has {len(_sample_turns)} turns total, "
          f"{len(retriever_turns_in_sample)} are retriever turns.")
    if not retriever_turns_in_sample:
        print(f"[WARNING] No retriever turns found! Check that '{role_key}' == 'retriever' "
              f"matches actual values: {[t.get(role_key) for t in _sample_turns]}")

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
                if turn.get(role_key) == "retriever":
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
        print("No complete windows produced results. Check log files and role key.")
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
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
    })

    c_reward  = '#1f77b4'  # Steel Blue
    c_length  = '#d62728'  # Brick Red
    c_entropy = '#2ca02c'  # Forest Green
    c_enorm   = '#ff7f0e'  # Orange (normalised entropy)
    c_vocab   = '#9467bd'  # Deep Purple

    x_axis = df["Window_Index"]

    # ------------------------------------------------------------------
    # Plot 1: Trade-off — Compression vs. Usefulness
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(x_axis, df["Avg_Reward"],
                     color=c_reward, marker='o', linestyle='-',
                     linewidth=2, markersize=6, label='Average Reward (Task)')
    line2 = ax2.plot(x_axis, df["Avg_Message_Length"],
                     color=c_length, marker='s', linestyle='-',
                     linewidth=2, markersize=6, label='Avg Message Length (Retriever)')

    ax1.set_xlabel('Training Steps (Windows)')
    ax1.set_ylabel('Task Reward', color=c_reward, fontweight='bold')
    ax2.set_ylabel('Number of Tokens', color=c_length, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=c_reward)
    ax2.tick_params(axis='y', labelcolor=c_length)

    lines  = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=False, edgecolor='black')

    plt.title('Information Bottleneck: Compression vs. Task Usefulness', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.5, color='#b0b0b0')

    fig1_path = os.path.join(output_dir, "exp1_compression_vs_reward.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Evolution of Entropy (raw + normalised) and Vocabulary Size
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    ax_ent, ax_enorm, ax_voc = axes

    # Raw entropy subplot
    ax_ent.plot(x_axis, df["Unigram_Entropy"],
                color=c_entropy, marker='^', linestyle='-', linewidth=2, markersize=7)
    ax_ent.set_title(f'Raw Entropy of Token Distribution (windows of {window_size} traj.)')
    ax_ent.set_ylabel('Entropy (bits)')
    ax_ent.grid(True, linestyle='--', alpha=0.5, color='#b0b0b0')

    # FIX #3: Normalised entropy subplot — comparable across window sizes
    ax_enorm.plot(x_axis, df["Unigram_Entropy_Norm"],
                  color=c_enorm, marker='^', linestyle='-', linewidth=2, markersize=7)
    ax_enorm.set_title('Normalised Entropy  H / log₂(V)  — window-size independent')
    ax_enorm.set_ylabel('Normalised Entropy [0, 1]')
    ax_enorm.set_ylim(0, 1.05)
    ax_enorm.grid(True, linestyle='--', alpha=0.5, color='#b0b0b0')

    # Vocabulary size subplot
    ax_voc.plot(x_axis, df["Active_Vocab_Size"],
                color=c_vocab, marker='D', linestyle='-', linewidth=2, markersize=5)
    ax_voc.set_title('Active Vocabulary Size Over Time')
    ax_voc.set_xlabel('Training Steps (Windows)')
    ax_voc.set_ylabel('Unique Words')
    ax_voc.grid(True, linestyle='--', alpha=0.5, color='#b0b0b0')

    fig2.tight_layout(pad=2.0)

    fig2_path = os.path.join(output_dir, "exp1_entropy_and_vocab.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Completed analysis. High-resolution plots saved to: {output_dir}")


if __name__ == "__main__":
    LOG_DIR = "./Agents/workflow_logs"
    OUT_DIR = "./Agents/eval_results/experiment_1"

    run_morphology_analysis(LOG_DIR, OUT_DIR, window_size=20)