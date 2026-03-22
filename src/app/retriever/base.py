from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Abstract base class for retrievers.

    A retriever takes a user prompt and the explorer's output,
    then filters for the most important information.
    """

    @abstractmethod
    def retrieve(self, prompt: str, explorer_output: str) -> str:
        """Filter and rank the explorer output for the most relevant information.

        Args:
            prompt: The user's research prompt / question.
            explorer_output: Raw materials returned by the explorer.

        Returns:
            A string containing the most important filtered information.
        """
        ...
