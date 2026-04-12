from app.api_client.base import BaseAPIClient, Message
from app.generator.base import BaseGenerator
from app.generator.prompts import (
    FEEDBACK_SYSTEM_PROMPT,
    FEEDBACK_USER_TEMPLATE,
    GENERATE_USER_TEMPLATE,
    GENERATOR_SYSTEM_PROMPT,
    FeedbackResponse,
    HypothesesResponse,
)
from app.models import FeedbackResult, GeneratorResult, RetrieverResult


class APILLMGenerator(BaseGenerator):
    """Generator that uses an LLM API to produce scientific hypotheses."""

    def __init__(self, api_client: BaseAPIClient) -> None:
        self.api_client = api_client

    def generate(
        self, prompt: str, retriever_output: RetrieverResult
    ) -> GeneratorResult:
        """Call the LLM to generate hypotheses using structured output.

        Args:
            prompt: The user's research prompt / question.
            retriever_output: Filtered information from the retriever.

        Returns:
            A GeneratorResult containing the list of hypotheses.
        """
        user_content = GENERATE_USER_TEMPLATE.format(
            prompt=prompt,
            retriever_output=retriever_output.content,
        )
        messages = [
            Message(role="system", content=GENERATOR_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        result = self.api_client.call(messages, response_schema=HypothesesResponse)

        return GeneratorResult(
            hypotheses=result.content.hypotheses,
            metadata={
                "source": "api_llm_generator",
                "model": result.model,
            },
        )

    def provide_feedback(
        self, prompt: str, retriever_output: RetrieverResult
    ) -> FeedbackResult:
        """Call the LLM to review retriever output and decide whether to refine.

        Uses structured output so the LLM can explicitly signal that the
        context is already sufficient (``skip_feedback=True``) instead of
        always requesting retriever refinement.

        Args:
            prompt: The user's research prompt / question.
            retriever_output: Current result from the retriever.

        Returns:
            A FeedbackResult with ``skip_feedback`` set to ``True`` to bypass
            retriever refinement, or ``False`` with actionable feedback text.
        """
        user_content = FEEDBACK_USER_TEMPLATE.format(
            prompt=prompt,
            retriever_output=retriever_output.content,
        )
        messages = [
            Message(role="system", content=FEEDBACK_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        result = self.api_client.call(messages, response_schema=FeedbackResponse)

        return FeedbackResult(
            skip_feedback=result.content.skip_feedback,
            content=result.content.content,
            metadata={
                "source": "api_llm_generator",
                "model": result.model,
                "step": "feedback",
            },
        )
