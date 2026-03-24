import os
import sys
import requests
from dataclasses import dataclass
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.dirname(SCRIPT_DIR)
DISK_DIR = os.path.dirname(AGENTS_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

from src.sft.utils.config import CONFIG

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sft.utils.config import CONFIG
from retrieval.index import DenseInMemoryIndex
from retrieval.embedding_retriever import EmbeddingRetriever

ARTICLES_DIR = os.path.join(DISK_DIR, "articles")
INDEX_PATH = os.path.join(ARTICLES_DIR, CONFIG["files"]["index"])

EMBEDDINGS_DIR = os.path.join(DISK_DIR, "embeddings")
PROMPT_EMBEDDINGS = os.path.join(EMBEDDINGS_DIR, CONFIG["files"]["prompt_embeddings"])

RETRIEVAL_DIR = os.path.join(AGENTS_DIR, "data", "retrieval")
os.makedirs(RETRIEVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RETRIEVAL_DIR, CONFIG["files"]["retrieved_contexts"])

RETRIEVAL_K = CONFIG["retrieval"]["top_k"]
RERANK_TOP_K = CONFIG["retrieval"].get("rerank_top_k", 5)

RERANKER_MODEL = CONFIG['models']['reranker_model']
VLLM_API_URL = CONFIG['models']['vllm_api_url']
VLLM_API_KEY = CONFIG['models']['vllm_api_key']
_RERANKER_PORT = CONFIG['models']['reranker_port']
RERANKER_API_URL = f"{VLLM_API_URL}:{_RERANKER_PORT}/v1"

@dataclass
class RankedChunk:
    chunk: object
    score: float = 0.0

class VLLMRerankerSync:
    def __init__(self, model: str, api_url: str, api_key: str):
        self.model = model
        base = api_url.rstrip('/')
        self.rerank_url = base if base.endswith('/v1/rerank') else f"{base}/v1/rerank"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def rerank(self, query: str, candidates: List[RankedChunk], top_n: int) -> List[RankedChunk]:
        documents = [rc.chunk.content for rc in candidates]
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        
        response = requests.post(self.rerank_url, headers=self._headers, json=payload)
        response.raise_for_status()
        results = response.json().get("results", [])

        for item in results:
            candidates[item["index"]].score = item["relevance_score"]

        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        return reranked[:top_n]

class EncoderStub:
    def __init__(self, device="cpu"):
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
        print(
            f"Error loading prompt embeddings: {e}. Make sure embed_prompts.py completed successfully."
        )
        sys.exit(-1)

    print(f"Loading index from {INDEX_PATH}...")
    try:
        encoder = EncoderStub(device="cuda:0")
        index = DenseInMemoryIndex(device="cuda:0")
        index.load(INDEX_PATH)
        retriever = EmbeddingRetriever(encoder, index)
    except Exception as e:
        print(f"Error loading index: {e}. Check files in {ARTICLES_DIR}")
        sys.exit(-1)

    print("Initialising VLLM Reranker...")
    reranker = VLLMRerankerSync(
        model=RERANKER_MODEL,
        api_url=RERANKER_API_URL,
        api_key=VLLM_API_KEY
    )

    print(f"Starting dense retrieval + reranking for {len(queries_data)} prompts...")
    print(f"Retrieving Top-{RETRIEVAL_K} -> Reranking to Top-{RERANK_TOP_K}")

    results_list = []

    for _, row in tqdm(queries_data.iterrows(), total=len(queries_data)):
        user_query_emb = row.embedding
        
        prompt_text = row.get("generated_prompt", row.get("prompt", ""))

        try:
            # 1. Dense Retrieval
            initial_results = retriever.retrieve(user_query_emb, top_k=RETRIEVAL_K)
            
            # 2. Reranking
            if prompt_text:
                candidates = [RankedChunk(chunk=r.chunk) for r in initial_results]
                final_results = reranker.rerank(query=prompt_text, candidates=candidates, top_n=RERANK_TOP_K)
            else:
                print(f"Warning: Brak tekstu promptu dla prompt_id {row.prompt_id}. Pomijam reranking.")
                final_results = [RankedChunk(chunk=r.chunk) for r in initial_results[:RERANK_TOP_K]]

            # 3. Context Formatting
            context_str = "\n\n".join([rc.chunk.content for rc in final_results])
            context_meta = [
                {
                    "chunk_id": rc.chunk.chunk_id,
                    "doc_id": rc.chunk.metadata.get("doc_id", "unknown"),
                    "rerank_score": round(rc.score, 4) if hasattr(rc, 'score') else None
                }
                for rc in final_results
            ]

            results_list.append(
                {
                    "prompt_id": row.prompt_id,
                    "topic_id": row.get("topic_id", None),
                    "retrieved_context": context_str,
                    "context_meta": context_meta,
                }
            )

        except Exception as e:
            print(f"Retrieval/Reranking error for prompt_id {row.prompt_id}: {e}")

    print(f"Retrieval and Reranking complete. Saving contexts to {OUTPUT_PATH}...")
    results_df = pd.DataFrame(results_list)
    results_df.to_pickle(OUTPUT_PATH)
    print("Done!")
