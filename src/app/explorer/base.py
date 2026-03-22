from abc import ABC, abstractmethod


class BaseExplorer(ABC):
    """Abstract base class for knowledge graph explorers.

    An explorer takes a user prompt and retrieves relevant materials
    from a knowledge graph, returning them as a string.
    """

    @abstractmethod
    def explore(self, prompt: str) -> str:
        """Explore the knowledge graph for materials relevant to the prompt.

        Args:
            prompt: The user's research prompt / question.

        Returns:
            A string containing relevant materials from the knowledge graph.
        """
        ...
