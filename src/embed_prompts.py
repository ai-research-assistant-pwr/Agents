import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.dirname(SCRIPT_DIR)
if AGENTS_DIR not in sys.path:
    sys.path.append(AGENTS_DIR)

import gc
import numpy as np
import pandas as pd
from vllm import LLM
from vllm.config.compilation import CompilationConfig
from utils.config import CONFIG

RAW_BASE_PATH = CONFIG['paths']['base_path']
ABS_BASE_PATH = os.path.expanduser(RAW_BASE_PATH)

RAW_AGENTS_PATH = CONFIG['paths']['base_path_agents']
ABS_AGENTS_PATH = os.path.expanduser(RAW_AGENTS_PATH)

PROMPTS_DIR = os.path.join(ABS_AGENTS_PATH, CONFIG['paths']['prompts_dir_agents'])
QUERIES_FILE = os.path.join(PROMPTS_DIR, CONFIG['files']['prompts'])
EMBEDDINGS_DIR = os.path.join(ABS_BASE_PATH, CONFIG['paths']['embeddings_dir'])
EMBEDDINGS_OUTPUT_PATH = os.path.join(EMBEDDINGS_DIR, CONFIG['files']['prompt_embeddings'])

DISK_DIR = os.path.dirname(ABS_BASE_PATH)
MODELS_DIR = os.path.join(DISK_DIR, 'models')

# EMBED_MODEL = CONFIG['models']['embedding_model']
EMBED_MODEL = "/home/tymrom7227/disk/models/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"

COMPILE_CACHE_PATH = os.path.join(ABS_BASE_PATH, CONFIG['paths']['cache_dir'], "vllm")
os.makedirs(COMPILE_CACHE_PATH, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

NUM_GPUS = CONFIG['hardware']['num_gpus']
GPU_MEMORY_UTILIZATION = CONFIG['hardware']['gpu_memory_utilization']
BATCH_SIZE_ENCODE = CONFIG['inference']['batch_size_encode']

if __name__ == "__main__":
    print(f"Loading queries from {QUERIES_FILE}...")
    queries = pd.read_csv(QUERIES_FILE)
    
    queries['prompt_id'] = queries.index

    print(f"Initializing vLLM for model: {EMBED_MODEL}...")
    llm = LLM(
        model=EMBED_MODEL, 
        task="embed", 
        trust_remote_code=True,
        enforce_eager=True, 
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=NUM_GPUS,
        compilation_config=CompilationConfig(cache_dir=COMPILE_CACHE_PATH, local_cache_dir=COMPILE_CACHE_PATH)
    )

    print("Inferring embedding dimension...")
    sample_out = llm.embed([queries['generated_prompt'].iloc[0]]) 
    dim = len(sample_out[0].outputs.embedding)
    print(f"   Detected Dimension: {dim}")
    
    total_queries = len(queries)
    all_embeddings = np.zeros((total_queries, dim), dtype=np.float32)
    all_queries_ids = []
    all_topic_ids = []
    all_texts = []

    print(f"Encoding {total_queries} chunks using vLLM...")
    print(f"Starting batched encoding (Batch Size: {BATCH_SIZE_ENCODE})...")
    
    for i in range(0, total_queries, BATCH_SIZE_ENCODE):
        end_idx = min(i + BATCH_SIZE_ENCODE, total_queries)
        batch_queries = queries[i:end_idx]
        
        batch_texts, queries_ids, topic_ids = zip(
            *[(q.generated_prompt, q.prompt_id, q.topic_id) 
              for q in batch_queries.itertuples()]
        )

        print(f"   Processing {i} to {end_idx}...")
        outputs = llm.embed(batch_texts)
        
        for j, output in enumerate(outputs):
            all_embeddings[i + j] = output.outputs.embedding
        
        all_queries_ids.extend(queries_ids)
        all_topic_ids.extend(topic_ids)
        all_texts.extend(batch_texts)
        
        del outputs
        gc.collect()

    print(f"Encoding Complete. Matrix Shape: {all_embeddings.shape}")
    size_in_gb = all_embeddings.nbytes / (1024 ** 3)
    print(f"Embedding Matrix Size: {size_in_gb:.2f} GB")
    
    queries_data = pd.DataFrame({
        "prompt_id": all_queries_ids,
        "topic_id": all_topic_ids,
        "embedding": all_embeddings.tolist(),
        "text": all_texts
    })

    queries_data.to_pickle(EMBEDDINGS_OUTPUT_PATH)
    print(f"Queries embedded successfully. Saved to: {EMBEDDINGS_OUTPUT_PATH}")