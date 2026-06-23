from app.api_client.base import BaseAPIClient, Message
from app.generator.base import BaseGenerator
from app.generator.prompts import (
    FEEDBACK_SYSTEM_PROMPT,
    FEEDBACK_USER_TEMPLATE,
    GENERATE_USER_TEMPLATE,
    GENERATOR_SYSTEM_PROMPT,
    HypothesesResponse,
    build_persona_section,
)
from app.models import GeneratorResult, RetrieverResult


class APILLMGenerator(BaseGenerator):
    """Generator that uses an LLM API to produce scientific hypotheses."""

    def __init__(self, api_client: BaseAPIClient) -> None:
        self.api_client = api_client

    def generate(
        self, prompt: str, retriever_output: RetrieverResult, **kwargs
    ) -> GeneratorResult:
        """Call the LLM to generate hypotheses using structured output.

        Args:
            prompt: The user's research prompt / question.
            retriever_output: Filtered information from the retriever.

        Returns:
            A GeneratorResult containing the list of hypotheses.
        """
        persona = kwargs.pop("persona", None)
        system_prompt = GENERATOR_SYSTEM_PROMPT
        if persona:
            system_prompt = system_prompt + build_persona_section(persona)

        user_content = GENERATE_USER_TEMPLATE.format(
            prompt=prompt,
            retriever_output=retriever_output.content,
        )
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_content),
        ]

        result = self.api_client.call(messages, response_schema=HypothesesResponse)

        return GeneratorResult(
            hypotheses=result.content.hypotheses,
            reasoning=result.reasoning,
            metadata={
                "source": "api_llm_generator",
                "model": result.model,
            },
        )

    def provide_feedback(self, prompt: str, retriever_output: RetrieverResult) -> str:
        """Call the LLM to review retriever output and request refinements.

        Args:
            prompt: The user's research prompt / question.
            retriever_output: Current result from the retriever.

        Returns:
            A feedback string for the retriever.
        """
        user_content = FEEDBACK_USER_TEMPLATE.format(
            prompt=prompt,
            retriever_output=retriever_output.content,
        )
        messages = [
            Message(role="system", content=FEEDBACK_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        return self.api_client.call(messages).content
