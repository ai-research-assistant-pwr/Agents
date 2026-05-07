import os
import json
import glob
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import re

def tokenize(text):
    """Simple tokenizing on word level, for offline log analysis."""
    return re.findall(r'\b\w+\b', text.lower())

def run_morphology_analysis(logs_dir: str, output_dir: str, window_size: int = 50):
    os.makedirs(output_dir, exist_ok=True)
    
    # Loading and sorting logs (file names contain timestamp ts_ms)
    log_files = glob.glob(os.path.join(logs_dir, "*.json"))
    log_files.sort(key=lambda f: int(f.split("_")[2])) # sorting by ts_ms
    
    if not log_files:
        print(f"No logs found in directory: {logs_dir}")
        return

    rewards = []
    lengths = []
    penalties = []
    vocab_over_time = []
    entropy_over_time = []
    
    global_vocab = Counter()
    
    for idx, fpath in enumerate(log_files):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        rewards.append(data.get("total_reward", 0.0))
        
        # Analyzing only retriever turns (turn0 i turn2)
        turn0 = data["turns"][0]
        turn2 = data["turns"][2]
        
        # Calculating average length and penalty for retriever turns
        avg_len = (turn0.get("n_output_tokens", 0) + turn2.get("n_output_tokens", 0)) / 2.0
        lengths.append(avg_len)
        
        avg_penalty = (turn0.get("length_penalty", 0) + turn2.get("length_penalty", 0)) / 2.0
        penalties.append(avg_penalty)
        
        # Vocabulary analysis
        text_content = turn0.get("output_content", "") + " " + turn2.get("output_content", "")
        tokens = tokenize(text_content)
        global_vocab.update(tokens)
        
        # Calculating local entropy and vocab size every `window_size` trajectories
        if (idx + 1) % window_size == 0 or (idx + 1) == len(log_files):
            # Token distribution entropy
            counts = list(global_vocab.values())
            probs = np.array(counts) / sum(counts)
            h = entropy(probs, base=2)
            entropy_over_time.append(h)
            vocab_over_time.append(len(global_vocab))
            global_vocab.clear() # Cleaning for local window to capture temporal changes

    # --- Generating plots ---
    
    # Plot 1: Length vs Reward (Moving Average)
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ma_rewards = moving_average(rewards, window_size)
    ma_lengths = moving_average(lengths, window_size)
    
    x_axis = range(len(ma_rewards))
    ax1.plot(x_axis, ma_rewards, 'b-', label='Average Reward (Task)')
    ax2.plot(x_axis, ma_lengths, 'r-', label='Message Length (Retriever)')
    
    ax1.set_xlabel('Training Steps (Windows)')
    ax1.set_ylabel('Reward', color='b')
    ax2.set_ylabel('Number of Tokens', color='r')
    plt.title('Trade-off: Compression vs. Usefulness')
    
    fig1_path = os.path.join(output_dir, "exp1_length_vs_reward.png")
    plt.savefig(fig1_path)
    plt.close()

    # Plot 2: Evolution of Entropy and Vocabulary Size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(entropy_over_time, 'g-o')
    ax1.set_title(f'Entropy of Token Distribution (windows of {window_size} trajectories)')
    ax1.set_ylabel('Entropia (bity)')
    
    ax2.plot(vocab_over_time, 'm-o')
    ax2.set_title('Active Vocabulary Size Over Time')
    ax2.set_xlabel('Training Steps (Windows)')
    ax2.set_ylabel('Number of Unique Words')
    
    fig2_path = os.path.join(output_dir, "exp1_entropy_vocab.png")
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()
    
    print(f"Completed analysis. Plots saved to: {output_dir}")

if __name__ == "__main__":
    LOG_DIR = "./workflow_logs" 
    OUT_DIR = "./eval_results/exp1_morphology"
    run_morphology_analysis(LOG_DIR, OUT_DIR)