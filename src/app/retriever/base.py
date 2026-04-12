from abc import ABC, abstractmethod

from app.models import ExplorerResult, RetrieverResult


class BaseRetriever(ABC):
    """Abstract base class for retrievers.

    A retriever takes a user prompt and the explorer's output,
    then filters for the most important information.
    """

    @abstractmethod
    def retrieve(self, prompt: str, explorer_output: ExplorerResult) -> RetrieverResult:
        """Filter and rank the explorer output for the most relevant information.

        The ``explorer_output`` now carries raw chunk IDs and chunk data rather
        than pre-formatted text.  Implementations are responsible for converting
        the chunk data into a textual representation before processing.

        Args:
            prompt: The user's research prompt / question.
            explorer_output: Result returned by the explorer (chunk IDs + raw
                chunk data).

        Returns:
            A RetrieverResult containing the most important filtered information.
        """
        ...

    @abstractmethod
    def refine(
        self,
        prompt: str,
        current_context: RetrieverResult,
        generator_feedback: str,
    ) -> RetrieverResult:
        """Refine retrieved context based on feedback from the generator.

        Called during multi-turn refinement between the retriever and generator.

        Args:
            prompt: The user's research prompt / question.
            current_context: The current retriever result being refined.
            generator_feedback: Feedback string from the generator requesting changes.

        Returns:
            A refined RetrieverResult.
        """
        ...
