from abc import ABC, abstractmethod

from app.models import ExplorerResult


class BaseExplorer(ABC):
    """Abstract base class for knowledge graph explorers.

    An explorer takes a user prompt and retrieves relevant materials
    from a knowledge graph, returning them as an ExplorerResult.
    """

    @abstractmethod
    def explore(self, prompt: str) -> ExplorerResult:
        """Explore the knowledge graph for materials relevant to the prompt.

        Args:
            prompt: The user's research prompt / question.

        Returns:
            An ExplorerResult whose ``chunk_ids`` lists the identifiers of the
            retrieved chunks in relevance order, and whose ``chunks`` dict maps
            each ID to its raw data (must include at least a ``"content"`` key).
            Text parsing / formatting is intentionally left to the downstream
            retriever so that this layer stays storage-agnostic.
        """
        ...
