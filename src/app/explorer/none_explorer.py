"""Pass-through explorer that fetches paper details from Weaviate by ID.

When ``explorer.type = "none"`` in config, the pipeline skips graph
exploration and instead looks up each paper ID from the search step
directly in Weaviate to obtain titles, abstracts, and summaries.
"""

import logging

from weaviate.collections.classes.filters import Filter

from app.explorer.base import BaseExplorer
from app.explorer.tools.weaviate_tools import _get_weaviate_client
from app.models import ExplorerResult

logger = logging.getLogger(__name__)


class NoneExplorer(BaseExplorer):
    """Explorer that passes paper IDs through with Weaviate metadata lookup.

    For each paper ID obtained from the search step, fetches the
    title, abstract and summary from the Weaviate ``ResearchPapers``
    collection and returns them in the standard ``ExplorerResult`` format.
    """

    def __init__(self, config: dict) -> None:
        weaviate_cfg = config.get("explorer", {}).get("weaviate", {})
        self.collection_name = weaviate_cfg.get(
            "collection", "ResearchPapers"
        )

    def explore(self, prompt: str, paper_ids: list[str]) -> ExplorerResult:
        if not paper_ids:
            logger.warning("No paper IDs provided")
            return ExplorerResult(
                content="No papers to explore.",
                metadata={"prompt": prompt, "paper_count": 0},
            )

        logger.info(
            "Fetching %d papers from Weaviate collection=%s",
            len(paper_ids),
            self.collection_name,
        )

        client = _get_weaviate_client()
        collection = client.collections.get(self.collection_name)

        paper_filter = Filter.by_property("paperId").contains_any(paper_ids)
        type_filter = Filter.by_property("type").contains_any(
            ["ABSTRACT", "SUMMARY"]
        )

        response = collection.query.fetch_objects(
            filters=paper_filter & type_filter,
            limit=len(paper_ids) * 2,
            return_properties=["paperId", "title", "content", "type"],
        )

        papers_by_id: dict[str, dict] = {}
        for obj in response.objects:
            p = obj.properties
            pid = p.get("paperId")
            if not pid:
                continue
            if pid not in papers_by_id:
                papers_by_id[pid] = {
                    "id": pid,
                    "title": p.get("title", "No title"),
                    "abstract": "",
                    "summary": "",
                }
            if p.get("type") == "ABSTRACT":
                papers_by_id[pid]["abstract"] = p.get("content", "")
            elif p.get("type") == "SUMMARY":
                papers_by_id[pid]["summary"] = p.get("content", "")

        results = list(papers_by_id.values())

        sections = []
        for paper in results:
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

            header = f"[{title}] (id={paper_id})"
            sections.append(f"{header}\n" + "\n".join(content_parts))

        full_content = (
            "\n\n---\n\n".join(sections) if sections else "No papers found."
        )

        return ExplorerResult(
            content=full_content,
            metadata={
                "prompt": prompt,
                "paper_count": len(results),
                "starting_papers": paper_ids,
                "source": "none_explorer",
                "papers": results,
            },
        )
