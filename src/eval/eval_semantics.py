import os
import json
import glob
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    """Fetch embeddings from the vLLM server asynchronously in batches."""
    if not texts: return []
    url = f"http://{host}:{port}/v1/embeddings"
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
    """Compute the normalized Levenshtein distance between two strings."""
    max_len = max(len(s1), len(s2))
    if max_len == 0: return 0.0
    return Levenshtein.distance(s1, s2) / max_len

async def run_hybrid_topsim_analysis(logs_dir: str, output_dir: str, window_size: int = 50, max_samples_per_window: int = 150):
    """
    Analyzes the emergence of compositionality using Topological Similarity (TopSim).
    Computes both Lexical (Edit Distance) and Semantic (Embedding Distance) correlations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and sort log files by timestamp
    log_files = glob.glob(os.path.join(logs_dir, "traj_*.json"))
    parsed_files = []
    
    for f in log_files:
        try:
            ts = int(os.path.basename(f).split('_')[1])
            parsed_files.append((ts, f))
        except: continue
            
    parsed_files.sort(key=lambda x: x[0])
    
    if not parsed_files:
        print(f"No log files found in: {logs_dir}")
        return

    print(f"Found {len(parsed_files)} trajectories. Starting asynchronous windowed analysis...")

    results = []
    last_window_data = None # Store for final scatter plot generation

    # Process logs in sliding windows to observe evolution over time
    for i in range(0, len(parsed_files), window_size):
        batch_files = parsed_files[i:i+window_size]
        
        # Subsample within the window to prevent O(N^2) complexity from exploding
        if len(batch_files) > max_samples_per_window:
            np.random.seed(42 + i)
            indices = np.random.choice(len(batch_files), max_samples_per_window, replace=False)
            batch_files = [batch_files[idx] for idx in indices]
            
        meanings = [] # Meaning Space (Inputs)
        signals = []  # Signal Space (Emergent Language)
        
        for _, file_path in batch_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            try:
                turn0 = data["turns"][0]
                turn2 = data["turns"][2]
                
                # Meaning: Prompt + Paper Summaries (Turn 0 input)
                meaning_space = turn0.get("input", data.get("prompt", ""))
                # Cutting to 12k characters to avoid vLLM limits and reduce noise
                if meaning_space:
                    meaning_space = meaning_space[:12000]
                
                # Signal: Concatenated Retriever messages from Turn 0 and Turn 2
                signal_space = turn0.get("output_content", "") + " " + turn2.get("output_content", "")
                if signal_space:
                    signal_space = signal_space[:12000]
                
                if meaning_space and signal_space:
                    meanings.append(meaning_space)
                    signals.append(signal_space)
            except (KeyError, IndexError):
                continue
                
        if len(meanings) < 10: continue
            
        print(f"  Window {i // window_size + 1}: Fetching embeddings for {len(meanings)} samples...")
        
        # Asynchronous batch processing
        meaning_embeddings = []
        signal_embeddings = []
        batch_size = 50
        
        for b in range(0, len(meanings), batch_size):
            m_batch = meanings[b:b+batch_size]
            s_batch = signals[b:b+batch_size]
            
            m_embs = await get_embeddings(m_batch, EMBED_HOST, EMBED_PORT, EMBED_MODEL)
            s_embs = await get_embeddings(s_batch, EMBED_HOST, EMBED_PORT, EMBED_MODEL)
            
            meaning_embeddings.extend(m_embs)
            signal_embeddings.extend(s_embs)

        # Compute Pairwise Distance Matrices
        meaning_distances = pdist(meaning_embeddings, metric='cosine')
        signal_semantic_distances = pdist(signal_embeddings, metric='cosine')
        signal_lexical_distances = pdist(np.array(signals).reshape(-1, 1), lambda u, v: normalized_levenshtein(u[0], v[0]))
        
        # Calculate Spearman Correlation (TopSim)
        rho_sem, _ = spearmanr(meaning_distances, signal_semantic_distances)
        rho_lex, _ = spearmanr(meaning_distances, signal_lexical_distances)
        
        results.append({
            "Window_Index": i // window_size + 1,
            "Trajectories": len(meanings),
            "TopSim_Semantic_Rho": round(rho_sem, 4),
            "TopSim_Lexical_Rho": round(rho_lex, 4)
        })
        
        # Save the latest window data for the final scatter plot
        last_window_data = (meaning_distances, signal_semantic_distances, signal_lexical_distances, rho_sem, rho_lex)

    # Save metrics to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "exp2_topsim_evolution.csv")
    df.to_csv(csv_path, index=False)

    # === PLOT 1: TopSim Evolution over Time ===
    plt.figure(figsize=(10, 6))
    plt.plot(df["Window_Index"], df["TopSim_Semantic_Rho"], marker='o', label='Semantic TopSim (\u03C1 cosine)', color='#d62728')
    plt.plot(df["Window_Index"], df["TopSim_Lexical_Rho"], marker='s', label='Lexical TopSim (\u03C1 Levenshtein)', color='#1f77b4')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Emergence of Compositionality: TopSim Over Time')
    plt.xlabel('Training Steps (Windows)')
    plt.ylabel('TopSim (Spearman \u03C1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "exp2_topsim_evolution.png"), dpi=300)
    plt.close()

    # === PLOT 2: Scatter Plot (Final State) ===
    if last_window_data:
        m_dist, s_sem_dist, s_lex_dist, rho_sem, _ = last_window_data
        
        plt.figure(figsize=(8, 8))
        # Plot up to 5,000 random pairs to keep the plot readable
        plot_idx = np.random.choice(len(m_dist), min(5000, len(m_dist)), replace=False)
        
        plt.scatter(m_dist[plot_idx], s_sem_dist[plot_idx], alpha=0.15, s=15, c='#d62728')
        m, b = np.polyfit(m_dist[plot_idx], s_sem_dist[plot_idx], 1)
        plt.plot(m_dist[plot_idx], m * m_dist[plot_idx] + b, color='black', linewidth=2)
        
        plt.title(f'Final Training State (Semantic TopSim)\nSpearman \u03C1 = {rho_sem:.3f}')
        plt.xlabel('Input Semantic Distance (Context)')
        plt.ylabel('Signal Semantic Distance (Messages)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "exp2_topsim_scatter_final.png"), dpi=300)
        plt.close()

    print(f"Analysis completed successfully! Results saved to {output_dir}")

if __name__ == "__main__":
    LOGS = "./Agents/workflow_logs"
    OUT = "./Agents/eval_results/experiment_2"
    asyncio.run(run_hybrid_topsim_analysis(LOGS, OUT))