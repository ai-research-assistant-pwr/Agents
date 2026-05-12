from pathlib import Path

from app.explorer.base import BaseExplorer
from app.explorer.tools.agentic_tools import (
    MAX_ITERATIONS,
    MAX_RESULTS_PER_TOOL,
    agentic_explorer,
)
from app.models import ExplorerResult


class AgenticExplorer(BaseExplorer):
    """Explorer that uses LLM-driven tool selection for graph traversal.

    The agent iteratively selects tools (BFS, Random Walk, PageRank) to explore
    the Neo4j knowledge graph based on relevance to the user query.
    """

    def __init__(self, config: dict) -> None:
        agentic_cfg = config.get("explorer", {}).get("agentic", {})
        self.iterations = agentic_cfg.get("iterations", MAX_ITERATIONS)
        self.tools = agentic_cfg.get("tools", ["bfs_from_papers", "random_walk", "ppr"])
        self.tool_selection_prompt = agentic_cfg.get(
            "tool_selection_prompt", "tool_selection_prompt.yaml"
        )
        self.node_filtering_prompt = agentic_cfg.get(
            "node_filtering_prompt", "node_filtering_prompt.yaml"
        )
        self.model_name = agentic_cfg.get("model_name", "Qwen/Qwen3-4B")
        self.temperature = agentic_cfg.get("temperature", 0.7)
        self.max_results_per_tool = agentic_cfg.get("max_results_per_tool", MAX_RESULTS_PER_TOOL)
        self.include_abstracts = agentic_cfg.get("include_abstracts", True)
        self.include_summary = agentic_cfg.get("include_summary", False)
        self.selected_nodes_count_low = agentic_cfg.get("selected_nodes_count_low", 2)
        self.selected_nodes_count_high = agentic_cfg.get("selected_nodes_count_high", 3)

    def explore(self, prompt: str, paper_ids: list[str]) -> ExplorerResult:
        """Explore knowledge graph using LLM-driven tool selection.

        Args:
            prompt: The user's research prompt / question.
            paper_ids: List of paper IDs to start exploration from.

        Returns:
            ExplorerResult with expanded paper set.
        """
        if not paper_ids:
            return ExplorerResult(
                content="No papers to explore.",
                metadata={"prompt": prompt, "paper_count": 0},
            )

        tool_selection_path = str(
            Path(__file__).parent / "prompts" / self.tool_selection_prompt
        )
        node_filtering_path = str(
            Path(__file__).parent / "prompts" / self.node_filtering_prompt
        )

        results = agentic_explorer(
            start_nodes=paper_ids,
            iterations=self.iterations,
            tools=self.tools,
            tool_selection_prompt_path=tool_selection_path,
            node_filtering_prompt_path=node_filtering_path,
            user_query=prompt,
            model_name=self.model_name,
            temperature=self.temperature,
            max_results_per_tool=self.max_results_per_tool,
            include_abstracts=self.include_abstracts,
            selected_nodes_count_low=self.selected_nodes_count_low,
            selected_nodes_count_high=self.selected_nodes_count_high,
            include_summary=self.include_summary,
        )

        if not results:
            return ExplorerResult(
                content="No related papers found.",
                metadata={"prompt": prompt, "paper_count": 0},
            )

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

            header = f"[Agentic] {title} (id={paper_id})"
            sections.append(f"{header}\n" + "\n".join(content_parts))

        full_content = "\n\n---\n\n".join(sections) if sections else "No papers found."

        return ExplorerResult(
            content=full_content,
            metadata={
                "prompt": prompt,
                "paper_count": len(results),
                "starting_papers": paper_ids,
                "iterations": self.iterations,
                "tools_used": self.tools,
                "papers": results,
            },
        )