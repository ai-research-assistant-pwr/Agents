"""Neo4j PageRank explorer for knowledge graph traversal."""

import logging

from app.explorer.base import BaseExplorer
from app.explorer.tools.neo4j_tools import personalized_pagerank, close_connections
from app.models import ExplorerResult

logger = logging.getLogger(__name__)


class Neo4jPageRankExplorer(BaseExplorer):
    """Explorer that traverses Neo4j knowledge graph using Personalized PageRank.

    Starting from paper IDs obtained from search, computes personalized
    PageRank to find the most important related papers.
    """

    def __init__(self, config: dict) -> None:
        neo4j_cfg = config.get("explorer", {}).get("neo4j", {})
        self.top_n = neo4j_cfg.get("top_n", 15)
        self.direction = neo4j_cfg.get("direction", "both")
        self.max_iterations = neo4j_cfg.get("max_iterations", 20)
        self.damping_factor = neo4j_cfg.get("damping_factor", 0.85)
        logger.info(
            "Neo4jPageRankExplorer initialized: top_n=%d, direction=%s",
            self.top_n,
            self.direction,
        )

    def explore(self, prompt: str, paper_ids: list[str]) -> ExplorerResult:
        """Explore knowledge graph using PageRank from starting paper IDs.

        Args:
            prompt: The user's research prompt / question (for metadata).
            paper_ids: List of paper IDs to compute personalized PageRank from.

        Returns:
            ExplorerResult with ranked paper set.
        """
        if not paper_ids:
            logger.warning("No paper IDs provided for PageRank exploration")
            return ExplorerResult(
                content="No papers to explore.",
                metadata={"prompt": prompt, "paper_count": 0},
            )

        logger.info("PageRank exploration from %d starting papers", len(paper_ids))

        results = personalized_pagerank(
            start_paper_ids=paper_ids,
            top_n=self.top_n,
            direction=self.direction,
            max_iterations=self.max_iterations,
            damping_factor=self.damping_factor,
        )

        if not results:
            logger.warning("No papers found from PageRank")
            return ExplorerResult(
                content="No related papers found.",
                metadata={"prompt": prompt, "paper_count": 0},
            )

        sections = []
        for rank, paper in enumerate(results, start=1):
            score = paper.get("score", 0)
            title = paper.get("title", "No title")
            abstract = paper.get("abstract", "")
            summary = paper.get("summary", "")
            paper_id = paper.get("id", "")

            content_parts = []
            if title:
                content_parts.append(title)
            if abstract:
                content_parts.append(f"Abstract: {abstract}")
            if summary:
                content_parts.append(f"Summary: {summary}")

            header = f"[Rank {rank}, Score {score:.4f}] {title} (id={paper_id})"
            sections.append(f"{header}\n" + "\n".join(content_parts))

        full_content = "\n\n---\n\n".join(sections) if sections else "No papers found."

        return ExplorerResult(
            content=full_content,
            metadata={
                "prompt": prompt,
                "paper_count": len(results),
                "starting_papers": paper_ids,
                "top_n": self.top_n,
                "direction": self.direction,
                "papers": results,
            },
        )

    def __del__(self) -> None:
        try:
            close_connections()
        except Exception:
            pass