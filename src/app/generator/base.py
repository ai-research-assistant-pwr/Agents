from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Abstract base class for hypothesis generators.

    A generator takes a user prompt and the retriever's filtered output,
    then produces a list of scientific hypotheses.
    """

    @abstractmethod
    def generate(self, prompt: str, retriever_output: str) -> list[str]:
        """Generate scientific hypotheses based on the prompt and retrieved context.

        Args:
            prompt: The user's research prompt / question.
            retriever_output: Filtered information from the retriever.

        Returns:
            A list of hypothesis strings.
        """
        ...
