from abc import ABC, abstractmethod

from app.models import ExplorerResult


class BaseSearchExplorer(ABC):
    """Abstract base class for search explorers.

    A search explorer takes a user prompt and returns relevant paper IDs
    from a search index (e.g., vector database).
    """

    @abstractmethod
    def search(self, prompt: str) -> list[str]:
        """Search for relevant paper IDs.

        Args:
            prompt: The user's research prompt / question.

        Returns:
            A list of paper IDs relevant to the prompt.
        """
        ...


class BaseExplorer(ABC):
    """Abstract base class for knowledge graph explorers.

    An explorer takes a user prompt and a list of starting paper IDs,
    then retrieves and expands relevant materials from a knowledge graph,
    returning them as an ExplorerResult.
    """

    @abstractmethod
    def explore(self, prompt: str, paper_ids: list[str]) -> ExplorerResult:
        """Explore the knowledge graph for materials relevant to the prompt.

        Args:
            prompt: The user's research prompt / question.
            paper_ids: List of paper IDs to start exploration from.

        Returns:
            An ExplorerResult containing relevant materials and metadata.
        """
        ...