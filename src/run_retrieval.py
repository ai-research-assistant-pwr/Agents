import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.dirname(SCRIPT_DIR)
DISK_DIR = os.path.dirname(AGENTS_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils.config import CONFIG
from retrieval.index import DenseInMemoryIndex
from retrieval.embedding_retriever import EmbeddingRetriever

ARTICLES_DIR = os.path.join(DISK_DIR, "articles")
INDEX_PATH = os.path.join(ARTICLES_DIR, CONFIG['files']['index']) 

EMBEDDINGS_DIR = os.path.join(DISK_DIR, "embeddings")
PROMPT_EMBEDDINGS = os.path.join(EMBEDDINGS_DIR, CONFIG['files']['prompt_embeddings'])
RETRIEVAL_DIR = os.path.join(AGENTS_DIR, "data", "retrieval")
OUTPUT_PATH = os.path.join(RETRIEVAL_DIR, CONFIG['files']['retrieved_contexts'])

RETRIEVAL_K = CONFIG['retrieval']['top_k']

class EncoderStub():
    def __init__(self, device='cpu'):
        self.device = device

    def encode(self, query):
        if isinstance(query, np.ndarray):
            if query.dtype != np.float32:
                 query = query.astype(np.float32)
            tensor_emb = torch.from_numpy(query)
        elif isinstance(query, torch.Tensor):
            tensor_emb = query
        else:
            tensor_emb = torch.tensor(query)
        
        return tensor_emb.to(self.device)

if __name__ == "__main__":
    print(f"Loading prompt embeddings from {PROMPT_EMBEDDINGS}...")
    try:
        queries_data = pd.read_pickle(PROMPT_EMBEDDINGS)
    except Exception as e:
        print(f"Error loading prompt embeddings: {e}. Upewnij się, że embed_prompts.py zakończył się sukcesem.")
        sys.exit(-1)

    print(f"Loading index from {INDEX_PATH}...")
    try:
        encoder = EncoderStub(device='cuda:0')
        index = DenseInMemoryIndex(device='cuda:0')
        index.load(INDEX_PATH)
        retriever = EmbeddingRetriever(encoder, index)
    except Exception as e:
        print(f"Error loading index: {e}. Sprawdź pliki w {ARTICLES_DIR}")
        sys.exit(-1)

    print(f"Starting dense retrieval for {len(queries_data)} prompts (Top-K: {RETRIEVAL_K})...")
    
    results_list = []
    
    for _, row in tqdm(queries_data.iterrows(), total=len(queries_data)):
        user_query_emb = row.embedding 
        
        try:
            results = retriever.retrieve(user_query_emb, top_k=RETRIEVAL_K)
            
            context_str = "\n\n".join([r.chunk.content for r in results])
            context_meta = [{'chunk_id': r.chunk.chunk_id, 'doc_id': r.chunk.metadata.get('doc_id', 'unknown')} for r in results]
            
            results_list.append({
                "prompt_id": row.prompt_id,
                "topic_id": row.get("topic_id", None),
                "retrieved_context": context_str,
                "context_meta": context_meta
            })
            
        except Exception as e:
            print(f"Retrieval error for prompt_id {row.prompt_id}: {e}")

    print(f"Retrieval complete. Saving contexts to {OUTPUT_PATH}...")
    results_df = pd.DataFrame(results_list)
    results_df.to_pickle(OUTPUT_PATH)
    print("Done!")