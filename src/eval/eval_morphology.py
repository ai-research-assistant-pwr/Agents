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
    
    # load and sort log files by timestamp
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

    results = []
    
    # process logs in windows to compute metrics over time
    for i in range(0, len(parsed_files), window_size):
        batch_files = parsed_files[i:i+window_size]
        
        window_rewards = []
        window_message_lengths = []
        window_vocab = Counter()
        
        for _, file_path in batch_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            window_rewards.append(data.get("total_reward", 0.0))
            
            # Extract content from Retriever turns (Turn 0 and Turn 2)
            # We look at 'output_content' which is the raw intent before channel noise
            traj_lengths = []
            for turn in data.get("turns", []):
                if turn.get("role") == "retriever":
                    text = turn.get("output_content", "")
                    tokens = tokenize(text)
                    
                    traj_lengths.append(len(tokens))
                    window_vocab.update(tokens)
            
            # Average retriever message length for this specific trajectory
            if traj_lengths:
                window_message_lengths.append(sum(traj_lengths) / len(traj_lengths))
        
        if not window_message_lengths:
            continue
            
        # Calculate window metrics
        avg_reward = np.mean(window_rewards)
        avg_length = np.mean(window_message_lengths)
        vocab_size = len(window_vocab)
        
        # Calculate Token Distribution Entropy for the window
        counts = list(window_vocab.values())
        if counts:
            probs = np.array(counts) / sum(counts)
            h = entropy(probs, base=2)
        else:
            h = 0.0
            
        results.append({
            "Window_Index": i // window_size + 1,
            "Trajectories_Count": len(batch_files),
            "Avg_Reward": round(avg_reward, 4),
            "Avg_Message_Length": round(avg_length, 2),
            "Active_Vocab_Size": vocab_size,
            "Unigram_Entropy": round(h, 4)
        })

    # Save metrics to CSV for further analysis
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp1_morphology_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw windowed data to: {csv_path}")

    # Generate Plots
    
    # Plot 1: Trade-off - Compression vs. Usefulness (Reward vs Length)
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    x_axis = df["Window_Index"]
    
    # Plotting lines
    line1 = ax1.plot(x_axis, df["Avg_Reward"], 'b-o', label='Average Reward (Task)')
    line2 = ax2.plot(x_axis, df["Avg_Message_Length"], 'r-s', label='Avg Message Length (Retriever)')
    
    ax1.set_xlabel('Training Steps (Windows)')
    ax1.set_ylabel('Reward', color='b')
    ax2.set_ylabel('Number of Tokens', color='r')
    
    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Information Bottleneck Trade-off: Compression vs. Usefulness')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    fig1_path = os.path.join(output_dir, "exp1_compression_vs_reward.png")
    plt.savefig(fig1_path)
    plt.close()

    # Plot 2: Evolution of Entropy and Vocabulary Size
    fig, (ax_ent, ax_voc) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax_ent.plot(x_axis, df["Unigram_Entropy"], 'g-^')
    ax_ent.set_title(f'Entropy of Token Distribution (Windows of {window_size})')
    ax_ent.set_ylabel('Entropy (bits)')
    ax_ent.grid(True, linestyle='--', alpha=0.7)
    
    ax_voc.plot(x_axis, df["Active_Vocab_Size"], 'm-D')
    ax_voc.set_title('Active Vocabulary Size Over Time')
    ax_voc.set_xlabel('Training Steps (Windows)')
    ax_voc.set_ylabel('Unique Words')
    ax_voc.grid(True, linestyle='--', alpha=0.7)
    
    fig2_path = os.path.join(output_dir, "exp1_entropy_and_vocab.png")
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()
    
    print(f"Completed analysis. Plots saved to: {output_dir}")

if __name__ == "__main__":
    # Execute from 'disk' directory
    LOG_DIR = "./Agents/workflow_logs" 
    OUT_DIR = "./Agents/eval_results/experiment_1"
    
    run_morphology_analysis(LOG_DIR, OUT_DIR, window_size=20)