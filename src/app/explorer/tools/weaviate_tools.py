import os
from typing import Any

import weaviate
from dotenv import load_dotenv
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from weaviate.classes.init import AdditionalConfig, Auth, Timeout
from weaviate.collections.classes.filters import Filter

load_dotenv()

VPS_IP = os.getenv("VPS_IP")
API_KEY = os.getenv("WEAVIATE_API_KEY")

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"

llm: LLM | None = None
reranker_llm: LLM | None = None
reranker_tokenizer: Any = None
client: weaviate.WeaviateClient | None = None


def _load_embedding_model() -> LLM:
    global llm
    if llm is None:
        print(f"Loading embedding model {MODEL_NAME}...")
        llm = LLM(
            model=MODEL_NAME,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        )
        print("Embedding model loaded.")
    return llm


def reload_embedding_model() -> LLM:
    global llm
    if llm is not None:
        del llm
        llm = None
    return _load_embedding_model()


def _load_reranker_model() -> tuple[LLM, Any]:
    global reranker_llm, reranker_tokenizer
    if reranker_llm is None:
        print(f"Loading reranker model {RERANKER_MODEL_NAME}...")
        reranker_tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL_NAME, padding_side="left"
        )
        reranker_tokenizer.pad_token = reranker_tokenizer.eos_token

        reranker_llm = LLM(
            model=RERANKER_MODEL_NAME,
            max_model_len=8192,
            enable_prefix_caching=True,
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.3,
        )
        print("Reranker model loaded.")
    return reranker_llm, reranker_tokenizer


def reload_reranker_model() -> tuple[LLM, Any]:
    global reranker_llm, reranker_tokenizer
    if reranker_llm is not None:
        del reranker_llm
        reranker_llm = None
        reranker_tokenizer = None
    return _load_reranker_model()


def _get_weaviate_client() -> weaviate.WeaviateClient:
    global client
    if client is None:
        client = weaviate.connect_to_custom(
            http_host=VPS_IP,
            http_port=8080,
            http_secure=False,
            grpc_host=VPS_IP,
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials=Auth.api_key(API_KEY),
            additional_config=AdditionalConfig(
                timeout=Timeout(init=60, query=300, insert=300)
            ),
        )
    return client


def get_n_sim_records(N: int, user_query: str) -> list[dict[str, Any]]:
    model = _load_embedding_model()
    weaviate_client = _get_weaviate_client()

    text = user_query.strip()
    output = model.embed([text])
    query_embedding = output[0].outputs.embedding

    collection = weaviate_client.collections.get("ResearchPapers")

    type_filter = Filter.by_property("type").equal("SUMMARY") | Filter.by_property(
        "type"
    ).equal("ABSTRACT")

    response = collection.query.near_vector(
        near_vector=query_embedding,
        filters=type_filter,
        limit=N,
        return_properties=["paperId", "title", "content", "type"],
    )

    results = []
    for obj in response.objects:
        props = obj.properties
        results.append(
            {
                "id": props.get("paperId"),
                "content": props.get("content"),
                "type": props.get("type"),
                "title": props.get("title"),
            }
        )

    return results


def get_chunks_from_article(
    paper_ids: list[str],
    query: str,
    n_max: int,
) -> list[dict[str, Any]]:
    """
    Fetch most relevant CHUNK records from specific papers.

    Args:
        paper_ids: List of paper IDs to search within
        query: Semantic search query
        n_max: Maximum number of chunks to return

    Returns:
        List of dicts with 'id', 'content', 'title', 'chunkIndex'
    """
    if not paper_ids:
        return []

    model = _load_embedding_model()
    weaviate_client = _get_weaviate_client()

    query_embedding = model.embed([query.strip()])[0].outputs.embedding
    query_embedding = (
        query_embedding.tolist()
        if hasattr(query_embedding, "tolist")
        else query_embedding
    )

    collection = weaviate_client.collections.get("ResearchPapers")

    type_filter = Filter.by_property("type").equal("CHUNK")

    paper_filter = Filter.by_property("paperId").contains_any(paper_ids)

    combined_filter = type_filter & paper_filter

    response = collection.query.near_vector(
        near_vector=query_embedding,
        filters=combined_filter,
        limit=n_max,
        return_properties=["paperId", "title", "content", "chunkIndex"],
    )

    results = []
    for obj in response.objects:
        props = obj.properties
        results.append(
            {
                "id": props.get("paperId"),
                "content": props.get("content"),
                "title": props.get("title"),
                "chunkIndex": props.get("chunkIndex"),
            }
        )

    return results


def rerank_and_limit(
    records: list[dict],
    user_query: str,
    top_k: int,
    only_id: bool = False,
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
) -> list[dict[str, Any]] | list[str]:
    """
    Rerank initial records using Qwen/Qwen3-Reranker-4B and return top_k results.

    Args:
        records: Initial results from get_n_sim_records
        user_query: The original user query
        top_k: Number of results to return after reranking
        only_id: If True, return only list of paperIds
        instruction: Custom instruction for reranking task

    Returns:
        Top-k reranked records (dict or list of paperIds)
    """
    if not records:
        return []

    if top_k is None or top_k > len(records):
        top_k = len(records)

    model, tokenizer = _load_reranker_model()

    query = user_query.strip()
    documents = [r["content"] for r in records]

    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    max_length = 8192

    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]

    def format_messages(instruction: str, query: str, doc: str):
        return [
            {
                "role": "system",
                "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".',
            },
            {
                "role": "user",
                "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
            },
        ]

    pairs = list(zip([query] * len(documents), documents))
    messages_list = [format_messages(instruction, q, d) for q, d in pairs]

    processed_messages = tokenizer.apply_chat_template(
        messages_list, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    processed_messages = [
        ele[:max_length] + suffix_tokens for ele in processed_messages
    ]
    inputs = [TokensPrompt(prompt_token_ids=ele) for ele in processed_messages]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )

    outputs = model.generate(inputs, sampling_params, use_tqdm=False)

    scores = []
    for output in outputs:
        final_logits = output.outputs[0].logprobs[-1]
        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob
        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob
        true_score = _safe_exp(true_logit)
        false_score = _safe_exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)

    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    top_indices = [idx for idx, _ in indexed_scores[:top_k]]

    if only_id:
        return [records[i]["id"] for i in top_indices]

    return [records[i] for i in top_indices]


def _safe_exp(x: float) -> float:
    import math

    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def close_connections():
    global client
    if client is not None:
        client.close()
        client = None
