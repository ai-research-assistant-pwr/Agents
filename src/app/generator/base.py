from abc import ABC, abstractmethod

from app.models import FeedbackResult, GeneratorResult, RetrieverResult


class BaseGenerator(ABC):
    """Abstract base class for hypothesis generators.

    A generator takes a user prompt and the retriever's filtered output,
    then produces a list of scientific hypotheses.
    """

    @abstractmethod
    def generate(
        self, prompt: str, retriever_output: RetrieverResult
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
        self, prompt: str, retriever_output: RetrieverResult
    ) -> FeedbackResult:
        """Review retriever output and request refinements or signal readiness.

        Called during multi-turn refinement between the retriever and generator.
        The returned ``FeedbackResult.skip_feedback`` flag allows the generator
        to short-circuit the refinement loop when the current context is already
        sufficient — in that case the pipeline will proceed directly to
        hypothesis generation instead of going back to the retriever.

        Args:
            prompt: The user's research prompt / question.
            retriever_output: Current result from the retriever.

        Returns:
            A FeedbackResult with ``skip_feedback=True`` to end the loop early,
            or ``skip_feedback=False`` and a non-empty ``content`` string
            describing what information is missing or needs change.
        """
        ...
