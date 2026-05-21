"""
Morphology Evaluation — Experiment 1 (Full Suite with Lexical Evolution)
======================================
Tests the Information Bottleneck hypothesis: does adding a per-token length
penalty (R = R_task − λ · length(message)) cause the Retriever to compress
its messages over training, and does that compression correlate with task
performance?

Metrics tracked over the course of training:
  - Average Message Length
  - Active Vocabulary Size
  - Unigram Entropy (raw + normalised)
  - Reward Dynamics (Total vs Task vs Penalty)
  - Reward Components (Similarity, Diversity, Groundedness, Relevancy)
  - Lexical Evolution (NEW: Relative frequency of top scientific terms)

Output
------
  eval_results/experiment_1/
    ├── exp1_morphology_metrics.csv
    ├── exp1_top5_evolution.json
    ├── exp1_compression_vs_reward.png
    ├── exp1_entropy_and_vocab.png
    ├── exp1_rewards_evolution.png
    ├── exp1_reward_components.png
    └── exp1_lexical_evolution.png
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

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "can", "will", "just", "don", "should", "now", "end"
}

def tokenize(text: str) -> list:
    """Tokenization with stop-word and length filtering."""
    if not text:
        return []
    # Filtracja: tylko litery, minimum 3 znaki
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return [w for w in words if w not in STOP_WORDS]

def run_morphology_analysis(logs_dir: str, output_dir: str, window_size: int = 50):
    os.makedirs(output_dir, exist_ok=True)

    train_logs_dir = os.path.join(logs_dir, "train")
    if os.path.isdir(train_logs_dir):
        print(f"Auto-detected 'train' subfolder. Reading trajectories from: {train_logs_dir}")
        logs_dir = train_logs_dir

    log_files = glob.glob(os.path.join(logs_dir, "traj_*.json"))
    parsed_files = []
    for f in log_files:
        basename = os.path.basename(f)
        try:
            ts = int(basename.split('_')[1])
            parsed_files.append((ts, f))
        except (IndexError, ValueError):
            continue

    parsed_files.sort(key=lambda x: x[0])

    if not parsed_files:
        print(f"No logs found in directory: {logs_dir}")
        return

    print(f"Found {len(parsed_files)} trajectories. Analyzing in windows of {window_size}...")

    results = []
    window_vocabs_raw = [] # Track full raw counters for post-processing lexical trends

    for i in range(0, len(parsed_files), window_size):
        batch_files = parsed_files[i:i + window_size]

        if len(batch_files) < window_size:
            print(f"  Window {i // window_size + 1}: Skipping incomplete window "
                  f"({len(batch_files)}/{window_size} trajectories).")
            continue

        window_total_rewards = []
        window_task_rewards = []
        window_length_penalties = []
        window_similarity = []
        window_diversity = []
        window_groundedness = []
        window_relevancy = []
        
        window_message_lengths = []
        window_vocab = Counter()

        for _, file_path in batch_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            total_r = data.get("total_reward", 0.0)
            ec_stats = data.get("ec_stats", {})
            task_r = ec_stats.get("task_reward_before_penalty", total_r)
            len_p = ec_stats.get("total_length_penalty", 0.0)

            window_total_rewards.append(total_r)
            window_task_rewards.append(task_r)
            window_length_penalties.append(len_p)

            traj_lengths = []
            sim_val, div_val, grd_val, rel_val = 0.0, 0.0, 0.0, 0.0

            for turn in data.get("turns", []):
                if turn.get("turn_type") == "retriever_message":
                    text = turn.get("output_content", "")
                    tokens = tokenize(text) 
                    n_tokens = turn.get("n_channel_tokens") or turn.get("extra", {}).get("n_channel_tokens", 0)
                    if n_tokens == 0 and text:
                        n_tokens = len(tokens)
                    traj_lengths.append(n_tokens)
                    window_vocab.update(tokens)
                
                if turn.get("turn_type") == "generator_generate":
                    sim_val = turn.get("similarity_score", 0.0)
                    div_val = turn.get("diversity_score", 0.0)
                    grd_val = turn.get("groundedness_score", 0.0)
                    rel_val = turn.get("relevancy_score", 0.0)

            window_similarity.append(sim_val)
            window_diversity.append(div_val)
            window_groundedness.append(grd_val)
            window_relevancy.append(rel_val)

            if traj_lengths:
                window_message_lengths.append(sum(traj_lengths) / len(traj_lengths))

        if not window_message_lengths:
            print(f"  Window {i // window_size + 1}: No retriever messages found — skipping.")
            continue

        vocab_size = len(window_vocab)
        counts = list(window_vocab.values())
        if counts:
            probs = np.array(counts) / sum(counts)
            h = entropy(probs, base=2)
            h_norm = h / np.log2(vocab_size) if vocab_size > 1 else 0.0
        else:
            h = 0.0
            h_norm = 0.0

        window_idx = i // window_size + 1
        window_vocabs_raw.append((window_idx, window_vocab))

        results.append({
            "Window_Index": window_idx,
            "Trajectories_Count": len(batch_files),
            "Avg_Total_Reward": round(np.mean(window_total_rewards), 4),
            "Avg_Task_Reward": round(np.mean(window_task_rewards), 4),
            "Avg_Length_Penalty": round(np.mean(window_length_penalties), 4),
            "Avg_Similarity": round(np.mean(window_similarity), 4),
            "Avg_Diversity": round(np.mean(window_diversity), 4),
            "Avg_Groundedness": round(np.mean(window_groundedness), 4),
            "Avg_Relevancy": round(np.mean(window_relevancy), 4),
            "Avg_Message_Length": round(np.mean(window_message_lengths), 2),
            "Active_Vocab_Size": vocab_size,
            "Unigram_Entropy": round(h, 4),
            "Unigram_Entropy_Norm": round(h_norm, 4),
            "Top_5_Words": dict(window_vocab.most_common(5)) 
        })

    if not results:
        print("No complete windows produced results. Check log files.")
        return

    # Save metrics and JSON
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp1_morphology_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw windowed data to: {csv_path}")

    top_words_evolution = {f"Window_{r['Window_Index']:03d}": r["Top_5_Words"] for r in results}
    json_path = os.path.join(output_dir, "exp1_top5_evolution.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(top_words_evolution, f, indent=4, ensure_ascii=False)
    print(f"Saved Top 5 words evolution to: {json_path}")

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
        'figure.facecolor': 'white',
        'axes.facecolor':   'white',
    })

    ACCENT   = '#1a1a1a'
    GREY_MID = '#666666'
    GREY_REF = '#aaaaaa'
    RED_ACCENT = '#d62728'
    x_axis = df["Window_Index"]

    # --- Plot 1: Compression vs Total Reward ---
    fig1, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={'hspace': 0.12})
    ax_r.plot(x_axis, df["Avg_Total_Reward"], color=ACCENT, lw=1.2)
    ax_r.fill_between(x_axis, df["Avg_Total_Reward"], alpha=0.06, color=ACCENT)
    ax_r.set_ylabel('Total reward')
    ax_r.grid(axis='y', lw=0.4, color=GREY_REF, ls=':')
    ax_r.set_title('Experiment 1 — Information Bottleneck: compression vs. total reward', pad=8)
    ax_l.plot(x_axis, df["Avg_Message_Length"], color=ACCENT, lw=1.2)
    ax_l.fill_between(x_axis, df["Avg_Message_Length"], alpha=0.06, color=ACCENT)
    ax_l.set_ylabel('Message length (tokens)')
    ax_l.set_xlabel('Training window')
    ax_l.grid(axis='y', lw=0.4, color=GREY_REF, ls=':')
    plt.savefig(os.path.join(output_dir, "exp1_compression_vs_reward.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Entropy and Vocab ---
    fig2, (ax_e, ax_en, ax_v) = plt.subplots(3, 1, figsize=(7, 7), sharex=True, gridspec_kw={'hspace': 0.12})
    ax_e.plot(x_axis, df["Unigram_Entropy"], color=ACCENT, lw=1.2)
    ax_e.set_ylabel('Entropy (bits)')
    ax_e.grid(axis='y', lw=0.4, color=GREY_REF, ls=':')
    ax_e.set_title(f'Experiment 1 — Token distribution entropy and vocabulary size\n(window = {window_size} trajectories)', pad=8)
    ax_en.plot(x_axis, df["Unigram_Entropy_Norm"], color=ACCENT, lw=1.2)
    ax_en.set_ylabel('H / log₂(V)')
    ax_en.set_ylim(0, 1.05)
    ax_en.grid(axis='y', lw=0.4, color=GREY_REF, ls=':')
    ax_v.plot(x_axis, df["Active_Vocab_Size"], color=ACCENT, lw=1.2)
    ax_v.set_ylabel('Vocabulary size')
    ax_v.set_xlabel('Training window')
    ax_v.grid(axis='y', lw=0.4, color=GREY_REF, ls=':')
    plt.savefig(os.path.join(output_dir, "exp1_entropy_and_vocab.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 3: Reward Evolution ---
    fig3, (ax_r_main, ax_pen) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={'hspace': 0.12})
    ax_r_main.plot(x_axis, df["Avg_Task_Reward"], color=GREY_MID, lw=1.2, ls='--', label='Task Reward')
    ax_r_main.plot(x_axis, df["Avg_Total_Reward"], color=ACCENT, lw=1.2, label='Total Reward')
    ax_r_main.fill_between(x_axis, df["Avg_Total_Reward"], df["Avg_Task_Reward"], color=RED_ACCENT, alpha=0.1, label='Penalty Gap')
    ax_r_main.set_ylabel('Reward Score')
    ax_r_main.grid(axis='y', lw=0.4, color=GREY_REF, ls=':')
    ax_r_main.set_title('Experiment 1 — Reward Dynamics: Task Quality vs. Length Penalty', pad=8)
    ax_r_main.legend(frameon=False, loc='lower left')
    ax_pen.plot(x_axis, df["Avg_Length_Penalty"], color=RED_ACCENT, lw=1.2)
    ax_pen.set_ylabel('Length Penalty')
    ax_pen.set_xlabel('Training window')
    ax_pen.grid(axis='y', lw=0.4, color=GREY_REF, ls=':')
    plt.savefig(os.path.join(output_dir, "exp1_rewards_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 4: Detailed Component Breakdown ---
    fig4, ax_comp = plt.subplots(figsize=(8, 4.5))
    ax_comp.plot(x_axis, df["Avg_Similarity"], color='#1f77b4', lw=1.2, label='Similarity')
    ax_comp.plot(x_axis, df["Avg_Diversity"], color='#ff7f0e', lw=1.2, label='Diversity')
    ax_comp.plot(x_axis, df["Avg_Groundedness"], color='#2ca02c', lw=1.2, label='Groundedness')
    ax_comp.plot(x_axis, df["Avg_Relevancy"], color='#9467bd', lw=1.2, label='Relevancy')
    ax_comp.plot(x_axis, df["Avg_Length_Penalty"], color=RED_ACCENT, lw=1.5, ls='--', label='Length Penalty')
    ax_comp.set_ylabel('Component Score / Penalty')
    ax_comp.set_xlabel('Training window')
    ax_comp.grid(axis='y', lw=0.4, color=GREY_REF, ls=':')
    ax_comp.set_title('Experiment 1 — Evolution of Individual Reward Components', pad=8)
    ax_comp.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.savefig(os.path.join(output_dir, "exp1_reward_components.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- NOWY Wykres 5: Kaskadowa Mapa Ciepła Ewolucji Leksykalnej ---
    global_top_keywords = set()
    for r in results:
        global_top_keywords.update(r["Top_5_Words"].keys())
        
    lexical_trends = {word: [] for word in global_top_keywords}
    for w_idx, vocab in window_vocabs_raw:
        total_words_in_window = sum(vocab.values())
        for word in global_top_keywords:
            freq = (vocab[word] / total_words_in_window) if total_words_in_window > 0 else 0.0
            lexical_trends[word].append(freq)
            
    # Wybieramy TOP 15 najważniejszych słów w całym treningu
    # SORTOWANIE KASKADOWE: Sortujemy według okna, w którym słowo osiągnęło SZCZYT popularności
    sorted_keywords = sorted(
        global_top_keywords, 
        key=lambda w: (np.argmax(lexical_trends[w]), -max(lexical_trends[w]))
    )[:15]
    
    # Budujemy macierz 2D (Słowa x Okna)
    matrix_data = np.array([lexical_trends[word] for word in sorted_keywords])
    
    fig5, ax_lex = plt.subplots(figsize=(9, 6))
    
    # Rysujemy mapę ciepła. Cmap 'Purples', 'Blues' lub 'Greys' wyglądają bardzo profesjonalnie
    im = ax_lex.imshow(matrix_data, cmap='Purples', aspect='auto', interpolation='nearest')
    
    # Konfiguracja osi Y (Słowa)
    ax_lex.set_yticks(np.arange(len(sorted_keywords)))
    ax_lex.set_yticklabels([f"'{word}'" for word in sorted_keywords], fontsize=9.5)
    
    # Konfiguracja osi X (Okna - pokazujemy co 10 okno, żeby opis nie był ściśnięty)
    x_indices = np.arange(len(x_axis))
    ax_lex.set_xticks(x_indices[::10])
    ax_lex.set_xticklabels(x_axis[::10])
    
    ax_lex.set_xlabel('Training window')
    ax_lex.set_ylabel('Emergent Vocabulary Tokens')
    ax_lex.set_title('Experiment 1 — Lexical Cascade: Vocabulary Stabilization Over Training', pad=12)
    
    # Dodajemy elegancki pasek boczny z legendą kolorów (colorbar)
    cbar = fig5.colorbar(im, ax_lex, fraction=0.03, pad=0.04)
    cbar.set_label('Relative Token Frequency (Probability Within Window)')
    cbar.outline.set_visible(False) # usuwamy ramkę wokół colorbaru dla minimalizmu
    
    # Usuwamy zbędne ramki wokół samej mapy ciepła
    for spine in ax_lex.spines.values():
        spine.set_visible(False)
        
    fig5.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp1_lexical_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Completed analysis. All 5 high-resolution plots saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 1 — Morphology Suite")
    parser.add_argument("--logs_dir",    default="./Agents/workflow_logs")
    parser.add_argument("--output_dir",  default="./Agents/eval_results/experiment_1")
    parser.add_argument("--window_size", type=int, default=50)
    args = parser.parse_args()

    run_morphology_analysis(args.logs_dir, args.output_dir, window_size=args.window_size)