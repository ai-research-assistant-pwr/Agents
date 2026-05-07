"""
Weaviate search tool for the scientific workflow.
"""
from typing import Any, Dict, List

import weaviate

def search_weaviate(query: str, n: int, weaviate_url: str) -> List[Dict[str, str]]:
    """
    Search the Weaviate database for papers relevant to the given query.

    The query is embedded using Weaviate's built-in text2vec vectoriser
    (the same model used when papers were indexed), so no separate embedding
    call is required here.

    Parameters
    ----------
    query       : Natural-language search query (e.g. the user research prompt).
    n           : Maximum number of results to return.
    weaviate_url: Base URL of the Weaviate instance, e.g. "http://localhost:8080".

    Returns
    -------
    List of dicts with keys "id", "title", "summary".
    """
    client = weaviate.Client(weaviate_url)

    result = (
        client.query.get("Paper", ["paper_id", "title", "summary"])
        .with_near_text({"concepts": [query]})
        .with_limit(n)
        .do()
    )

    papers: List[Dict[str, str]] = []
    hits: List[Dict[str, Any]] = result.get("data", {}).get("Get", {}).get("Paper", [])
    for hit in hits:
        papers.append(
            {
                "id": hit.get("paper_id", ""),
                "title": hit.get("title", ""),
                "summary": hit.get("summary", ""),
            }
        )
    return papers
