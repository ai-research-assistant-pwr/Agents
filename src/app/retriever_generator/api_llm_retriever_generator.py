from app.api_client.base import BaseAPIClient, Message
from app.models import ExplorerResult, GeneratorResult
from app.retriever_generator.base import BaseRetrieverGenerator
from app.retriever_generator.prompts import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    HypothesesResponse,
)


class APILLMRetrieverGenerator(BaseRetrieverGenerator):
    def __init__(
        self,
        api_client: BaseAPIClient,
        temperature: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ) -> None:
        self.api_client = api_client
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def generate(
        self, prompt: str, explorer_output: ExplorerResult
    ) -> GeneratorResult:
        user_content = USER_TEMPLATE.format(
            prompt=prompt,
            explorer_output=explorer_output.content,
        )
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]

        result = self.api_client.call(
            messages,
            response_schema=HypothesesResponse,
            temperature=self.temperature,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )

        hypotheses = result.content.hypotheses
        return GeneratorResult(
            hypotheses=[h.hypothesis_text for h in hypotheses],
            metadata={
                "source": "api_llm_retriever_generator",
                "model": result.model,
                "synthesis_rationales": [h.synthesis_rationale for h in hypotheses],
            },
        )
