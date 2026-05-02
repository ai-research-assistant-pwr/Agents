"""Neo4j Random Walk explorer for knowledge graph traversal."""

import logging

from app.explorer.base import BaseExplorer
from app.explorer.tools.neo4j_tools import random_walk, close_connections
from app.models import ExplorerResult

logger = logging.getLogger(__name__)


class Neo4jRandomWalkExplorer(BaseExplorer):
    """Explorer that traverses Neo4j knowledge graph using Random Walk.

    Starting from paper IDs obtained from search, performs random walk
    traversal to find related papers.
    """

    def __init__(self, config: dict) -> None:
        neo4j_cfg = config.get("explorer", {}).get("neo4j", {})
        self.steps = neo4j_cfg.get("steps", 10)
        self.direction = neo4j_cfg.get("direction", "both")
        logger.info(
            "Neo4jRandomWalkExplorer initialized: steps=%d, direction=%s",
            self.steps,
            self.direction,
        )

    def explore(self, prompt: str, paper_ids: list[str]) -> ExplorerResult:
        """Explore knowledge graph using Random Walk from starting paper IDs.

        Args:
            prompt: The user's research prompt / question (for metadata).
            paper_ids: List of paper IDs to start random walk from.

        Returns:
            ExplorerResult with expanded paper set.
        """
        if not paper_ids:
            logger.warning("No paper IDs provided for Random Walk exploration")
            return ExplorerResult(
                content="No papers to explore.",
                metadata={"prompt": prompt, "paper_count": 0},
            )

        logger.info("Random Walk exploration from %d starting papers", len(paper_ids))

        results = random_walk(
            start_paper_ids=paper_ids,
            steps=self.steps,
            direction=self.direction,
        )

        if not results:
            logger.warning("No papers found from Random Walk traversal")
            return ExplorerResult(
                content="No related papers found.",
                metadata={"prompt": prompt, "paper_count": 0},
            )

        sections = []
        for paper in results:
            level = paper.get("level", 0)
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

            header = f"[Walk Step {level}] {title} (id={paper_id})"
            sections.append(f"{header}\n" + "\n".join(content_parts))

        full_content = "\n\n---\n\n".join(sections) if sections else "No papers found."

        unique_papers = {p["id"]: p for p in results if p.get("id")}

        return ExplorerResult(
            content=full_content,
            metadata={
                "prompt": prompt,
                "paper_count": len(results),
                "unique_papers": len(unique_papers),
                "starting_papers": paper_ids,
                "steps": self.steps,
                "direction": self.direction,
                "papers": list(unique_papers.values()),
            },
        )

    def __del__(self) -> None:
        try:
            close_connections()
        except Exception:
            pass