from app.api_client.base import BaseAPIClient, Message
from app.models import ExplorerResult, RetrieverResult
from app.retriever.base import BaseRetriever
from app.retriever.prompts import (
    REFINE_SYSTEM_PROMPT,
    REFINE_USER_TEMPLATE,
    RETRIEVER_SYSTEM_PROMPT,
    RETRIEVE_USER_TEMPLATE,
)


class APILLMRetriever(BaseRetriever):
    """Retriever that uses an LLM API to filter and rank explorer output."""

    def __init__(self, api_client: BaseAPIClient) -> None:
        self.api_client = api_client

    def retrieve(self, prompt: str, explorer_output: ExplorerResult) -> RetrieverResult:
        """Call the LLM to extract the most relevant information.

        Args:
            prompt: The user's research prompt / question.
            explorer_output: Result returned by the explorer.

        Returns:
            A RetrieverResult with the LLM-filtered content.
        """
        user_content = RETRIEVE_USER_TEMPLATE.format(
            prompt=prompt,
            explorer_output=explorer_output.content,
        )
        messages = [
            Message(role="system", content=RETRIEVER_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        response = self.api_client.call(messages)

        return RetrieverResult(
            content=response,
            metadata={
                "source": "api_llm_retriever",
                "model": self.api_client.model,
            },
        )

    def refine(
        self,
        prompt: str,
        current_context: RetrieverResult,
        generator_feedback: str,
    ) -> RetrieverResult:
        """Call the LLM to refine the retrieved context based on generator feedback.

        Args:
            prompt: The user's research prompt / question.
            current_context: The current retriever result being refined.
            generator_feedback: Feedback string from the generator.

        Returns:
            A refined RetrieverResult.
        """
        user_content = REFINE_USER_TEMPLATE.format(
            prompt=prompt,
            current_context=current_context.content,
            generator_feedback=generator_feedback,
        )
        messages = [
            Message(role="system", content=REFINE_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        response = self.api_client.call(messages)

        return RetrieverResult(
            content=response,
            metadata={
                "source": "api_llm_retriever",
                "model": self.api_client.model,
                "step": "refine",
            },
        )
