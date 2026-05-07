import os
import json
import glob
import asyncio
import aiohttp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, cosine
import Levenshtein

# vLLM server parameters
EMBED_HOST = "localhost"
EMBED_PORT = 8000
EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"

async def get_embeddings(texts: list, host: str, port: int, model: str) -> list:
    """Fetch embeddings from the vLLM server (batched)."""
    url = f"http://{host}:{port}/v1/embeddings"
    payload = {"model": model, "input": texts}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
    ordered = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in ordered]

def normalized_levenshtein(s1: str, s2: str) -> float:
    """Return the normalized edit distance in the range [0, 1]."""
    if not s1 and not s2:
        return 0.0
    dist = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return dist / max_len

async def run_topsim_analysis(logs_dir: str, output_dir: str, sample_size: int = 200):
    os.makedirs(output_dir, exist_ok=True)
    
    log_files = glob.glob(os.path.join(logs_dir, "*.json"))
    if not log_files:
        print(f"No logs found in directory: {logs_dir}")
        return

    # Sample a subset of trajectories to avoid computing N^2 distances for e.g. 10,000 logs
    # (For 200 logs, we have ~20,000 pairs, which is statistically ideal)
    np.random.shuffle(log_files)
    selected_files = log_files[:sample_size]

    meaning_spaces = [] # Inputs (Prompt + Papers)
    signal_spaces = []  # Outputs (Retriever messages)

    for fpath in selected_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        turn0 = data["turns"][0]
        turn2 = data["turns"][2]
        
        # Meaning space: the exact prompt received by the Retriever
        meaning_spaces.append(turn0["input"])
        
        # Signal space: combined messages from both turns
        signal = turn0.get("output_content", "") + " " + turn2.get("output_content", "")
        signal_spaces.append(signal)

    print(f"Fetching embeddings for {len(meaning_spaces)} inputs...")
    
    # Fetch embeddings in batches to avoid overwhelming the server
    embeddings = []
    batch_size = 50
    for i in range(0, len(meaning_spaces), batch_size):
        batch = meaning_spaces[i:i+batch_size]
        embs = await get_embeddings(batch, EMBED_HOST, EMBED_PORT, EMBED_MODEL)
        embeddings.extend(embs)
        
    print("Computing distance matrices...")
    
    # Compute pairwise distance vectors
    # 1. Semantic distances (cosine distance)
    meaning_distances = pdist(embeddings, metric='cosine')
    
    # 2. Signal distances (normalized Levenshtein distance)
    n = len(signal_spaces)
    signal_distances = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = normalized_levenshtein(signal_spaces[i], signal_spaces[j])
            signal_distances.append(dist)
            
    signal_distances = np.array(signal_distances)

    # Compute the TopSim metric (Spearman correlation)
    rho, p_value = spearmanr(meaning_distances, signal_distances)
    print(f"\nTopographical Similarity (TopSim) results:")
    print(f"Spearman correlation (rho): {rho:.4f}")
    print(f"P-value: {p_value:.4e}")

    # Generate the scatter plot
    plt.figure(figsize=(8, 8))
    # Select at most 5,000 random points to keep the plot readable
    plot_indices = np.random.choice(len(meaning_distances), min(5000, len(meaning_distances)), replace=False)
    
    plt.scatter(meaning_distances[plot_indices], signal_distances[plot_indices], alpha=0.1, s=10, c='blue')
    
    # Trend line
    m, b = np.polyfit(meaning_distances[plot_indices], signal_distances[plot_indices], 1)
    plt.plot(meaning_distances[plot_indices], m * meaning_distances[plot_indices] + b, color='red', linewidth=2)

    plt.title(f'Topographical Similarity (TopSim)\nSpearman $\\rho$ = {rho:.3f} (p={p_value:.2e})')
    plt.xlabel('Semantic distance of inputs (Cosine Distance)')
    plt.ylabel('Message distance (Normalized Levenshtein)')
    plt.grid(True, alpha=0.3)
    
    out_file = os.path.join(output_dir, "exp2_topsim_scatter.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved plot to: {out_file}")

if __name__ == "__main__":
    LOGS = "./workflow_logs"
    OUT = "./eval_results/exp2_semantics"
    
    # Because we use aiohttp, we need to start the async event loop
    asyncio.run(run_topsim_analysis(LOGS, OUT, sample_size=200))