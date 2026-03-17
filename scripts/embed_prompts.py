import os
import gc
import numpy as np
import pandas as pd
from vllm import LLM
from vllm.config.compilation import CompilationConfig
from utils.config import CONFIG

RAW_BASE_PATH = CONFIG['paths']['base_path']
ABS_BASE_PATH = os.path.expanduser(RAW_BASE_PATH)

ARTICLES_DIR = os.path.join(ABS_BASE_PATH, CONFIG['paths']['articles_dir'])
CHUNKS_FILE = os.path.join(ARTICLES_DIR, CONFIG['files']['chunks']) 
INDEX_OUTPUT_PATH =  os.path.join(ARTICLES_DIR, CONFIG['files']['index'])

PROMPTS_DIR = os.path.join(ABS_BASE_PATH, CONFIG['paths']['prompts_dir'])
QUERIES_FILE = os.path.join(PROMPTS_DIR, CONFIG['files']['prompts'])
EMBEDDINGS_OUTPUT_PATH = os.path.join(PROMPTS_DIR, CONFIG['files']['prompt_embeddings'])

EMBED_MODEL = CONFIG['models']['embedding_model']
MODEL_DOWNLOAD_DIR = os.path.join(ABS_BASE_PATH, CONFIG['paths']['models_dir'])

COMPILE_CACHE_PATH = os.path.join(ABS_BASE_PATH, CONFIG['paths']['compile_cache_dir'])

NUM_GPUS = CONFIG['hardware']['num_gpus']
GPU_MEMORY_UTILIZATION = CONFIG['hardware']['gpu_memory_utilization']
BATCH_SIZE_ENCODE = CONFIG['inference']['batch_size_encode']

if __name__ == "__main__":
    print(f"Loading queries from {QUERIES_FILE}...")
    queries = pd.read_csv(QUERIES_FILE)

    print(f"Initializing vLLM for model: {EMBED_MODEL}...")
    llm = LLM(
        model=EMBED_MODEL, 
        task="embed", 
        trust_remote_code=True,
        enforce_eager=True, # Often needed for embedding models in vLLM to avoid graph capture issues
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        download_dir=MODEL_DOWNLOAD_DIR,
        tensor_parallel_size=NUM_GPUS,
        compilation_config=CompilationConfig(cache_dir=COMPILE_CACHE_PATH, local_cache_dir=COMPILE_CACHE_PATH)

    )

    print("Inferring embedding dimension...")
    sample_out = llm.embed([queries['generated_prompt'].iloc[0]]) # Use .embed() instead of .encode()
    dim = len(sample_out[0].outputs.embedding)
    print(f"   Detected Dimension: {dim}")
    
    total_queries = len(queries)
    all_embeddings = np.zeros((total_queries, dim), dtype=np.float32)
    all_queries_ids = []
    all_user_ids = []
    all_texts = []

    print(f"Encoding {total_queries} chunks using vLLM...")
    print(f"Starting batched encoding (Batch Size: {BATCH_SIZE_ENCODE})...")
    
    for i in range(0, total_queries, BATCH_SIZE_ENCODE):
        end_idx = min(i + BATCH_SIZE_ENCODE, total_queries)
        batch_queries = queries[i:end_idx]
        batch_texts, queries_ids, user_ids = zip(
            *[(q.generated_prompt, q.prompt_id, q.user_id) 
              for q in batch_queries.itertuples()]
        )

        print(f"   Processing {i} to {end_idx}...")
        outputs = llm.embed(batch_texts)
        
        # Extract embeddings directly into the pre-allocated array
        # This allows Python to garbage collect the heavy 'outputs' objects immediately
        for j, output in enumerate(outputs):
            all_embeddings[i + j] = output.outputs.embedding
        
        all_queries_ids.extend(queries_ids)
        all_user_ids.extend(user_ids)
        all_texts.extend(batch_texts)
        # Clean up intermediate objects to free RAM
        del outputs
        # del batch_texts
        gc.collect()

    print(f"Encoding Complete. Matrix Shape: {all_embeddings.shape}")
    size_in_gb = all_embeddings.nbytes / (1024 ** 3)
    print(f"Embedding Matrix Size: {size_in_gb:.2f} GB")
    print(type(all_embeddings))
    
    queries_data = pd.DataFrame({
        "prompt_id": queries_ids,
        "user_id": user_ids,
        "embedding": all_embeddings.tolist(),
        "text": all_texts
    })

    queries_data.to_pickle(EMBEDDINGS_OUTPUT_PATH)

    print("Queries embedded successfully.")
