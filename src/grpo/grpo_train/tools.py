"""
Weaviate search tools for the scientific workflow.
===================================================

The tool-call schema machinery from the previous version has been removed.
The pipeline is now step-driven and does not require LLMs to emit tool calls.

Public API
----------
  search_weaviate(query, n, weaviate_url, embed_host, embed_port)
      Full Weaviate semantic search; returns a list of paper dicts.

  search_papers_tool(query, weaviate_url, embed_host, embed_port)
      Convenience wrapper used by execute_turn for retriever_search steps.
      Fetches exactly 1 paper and returns a formatted string.
"""

from typing import Any, Dict, List

import requests


EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-4B"


def _get_query_vector(
    query: str,
    embed_host: str,
    embed_port: int,
) -> List[float]:
    """Embed *query* using the vLLM OpenAI-compatible /v1/embeddings endpoint."""
    url = f"http://{embed_host}:{embed_port}/v1/embeddings"
    payload = {"model": EMBEDDING_MODEL, "input": [query]}
    response = requests.post(
        url, json=payload, headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]


def search_weaviate(
    query: str,
    n: int,
    weaviate_url: str,
    embed_host: str,
    embed_port: int,
) -> List[Dict[str, str]]:
    """
    Search the Weaviate database for papers relevant to the given query.

    The query is embedded via the vLLM embedding server and the resulting
    vector is passed to Weaviate's ``nearVector`` GraphQL operator.

    Parameters
    ----------
    query        : Natural-language search query.
    n            : Maximum number of results to return.
    weaviate_url : Base URL of the Weaviate instance, e.g. "http://localhost:8080".
    embed_host   : Hostname of the vLLM embedding server.
    embed_port   : Port of the vLLM embedding server.

    Returns
    -------
    List of dicts with keys "id", "title", "summary".
    """
    vector = _get_query_vector(query, embed_host, embed_port)
    vector_str = ", ".join(str(v) for v in vector)
    graphql_query = {
        "query": (
            "{ Get { ResearchPapers("
            f"nearVector: {{vector: [{vector_str}]}} limit: {n}"
            ") { paperId title content } } }"
        )
    }
    response = requests.post(
        f"{weaviate_url}/v1/graphql",
        json=graphql_query,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    result = response.json()

    papers: List[Dict[str, str]] = []
    hits: List[Dict[str, Any]] = (
        result.get("data", {}).get("Get", {}).get("ResearchPapers", [])
    )
    for hit in hits:
        papers.append(
            {
                "id": hit.get("paperId", ""),
                "title": hit.get("title", ""),
                "summary": hit.get("content", ""),
            }
        )
    return papers


def search_papers_tool(
    query: str,
    weaviate_url: str,
    embed_host: str,
    embed_port: int,
) -> str:
    """
    Retrieve exactly 1 paper from Weaviate and format it as a string
    suitable for injection into the retriever's prompt context.

    Returns a plain string — either the formatted paper or an error note.
    """
    papers = search_weaviate(
        query,
        n=1,
        weaviate_url=weaviate_url,
        embed_host=embed_host,
        embed_port=embed_port,
    )
    if not papers:
        return "No results found for the given query."
    p = papers[0]
    title = p.get("title", "Unknown title")
    summary = p.get("summary", "No summary available.")
    return f"[Search result]\nTitle: {title}\n\n{summary.strip()}"
