"""Weaviate vector-database search explorer using Qwen3 embedding and reranking."""

import logging

from app.explorer.base import BaseSearchExplorer
from app.explorer.tools.weaviate_tools import (
    get_n_sim_records,
    rerank_and_limit,
    close_connections,
)

logger = logging.getLogger(__name__)


class WeaviateSearchExplorer(BaseSearchExplorer):
    """Search explorer that queries Weaviate vector database.

    Embeds the user prompt with Qwen3-Embedding and performs a
    nearest-neighbour search, then reranks results.
    """

    def __init__(self, config: dict) -> None:
        weaviate_cfg = config.get("search", {}).get("weaviate", {})
        self.top_k = weaviate_cfg.get("top_k", 20)
        self.rerank_top_k = weaviate_cfg.get("rerank_top_k", 10)

        logger.info(
            "WeaviateSearchExplorer initialized: top_k=%d, rerank_top_k=%d",
            self.top_k,
            self.rerank_top_k,
        )

    def search(self, prompt: str) -> list[str]:
        """Search Weaviate for relevant paper IDs.

        Args:
            prompt: The user's research prompt / question.

        Returns:
            List of paper IDs relevant to the prompt.
        """
        logger.info("Searching Weaviate for query (%d chars)", len(prompt))

        initial_results = get_n_sim_records(
            N=self.top_k,
            user_query=prompt,
        )

        if not initial_results:
            logger.warning("No results from Weaviate search")
            return []

        logger.info("Reranking %d initial results", len(initial_results))

        reranked = rerank_and_limit(
            records=initial_results,
            user_query=prompt,
            top_k=self.rerank_top_k,
            only_id=True,
        )

        logger.info("Returning %d paper IDs", len(reranked))
        return reranked

    def __del__(self) -> None:
        try:
            close_connections()
        except Exception:
            pass