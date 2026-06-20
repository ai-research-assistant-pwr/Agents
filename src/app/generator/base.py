from abc import ABC, abstractmethod

from app.models import GeneratorResult, RetrieverResult


class BaseGenerator(ABC):
    """Abstract base class for hypothesis generators.

    A generator takes a user prompt and the retriever's filtered output,
    then produces a list of scientific hypotheses.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        retriever_output: RetrieverResult,
        model_name: str,
        **kwargs,
    ) -> GeneratorResult:
        """Generate scientific hypotheses based on the prompt and retrieved context.

        Args:
            prompt: The user's research prompt / question.
            retriever_output: Filtered information from the retriever.

        Returns:
            A GeneratorResult containing the list of hypotheses and metadata.
        """
        ...

    @abstractmethod
    def provide_feedback(
        self,
        prompt: str,
        retriever_output: RetrieverResult,
        model_name: str,
        **kwargs,
    ) -> str:
        """Review retriever output and request refinements.

        Called during multi-turn refinement between the retriever and generator.

        Args:
            prompt: The user's research prompt / question.
            retriever_output: Current result from the retriever.

        Returns:
            A feedback string describing what information is missing or needs change.
        """
        ...
