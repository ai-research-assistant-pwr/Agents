from app.models import ExplorerResult, RetrieverResult
from app.openai_chat import OpenAIChatRouter
from app.retriever.base import BaseRetriever
from app.retriever.prompts import (
    REFINE_SYSTEM_PROMPT,
    REFINE_USER_TEMPLATE,
    RETRIEVER_SYSTEM_PROMPT,
    RETRIEVE_USER_TEMPLATE,
)


class OpenAIChatRetriever(BaseRetriever):
    """Retriever that calls OpenAI-compatible Chat Completions directly."""

    def __init__(self, model_configs: dict[str, dict]) -> None:
        self.router = OpenAIChatRouter(model_configs)

    def retrieve(
        self,
        prompt: str,
        explorer_output: ExplorerResult,
        model_name: str,
        **kwargs,
    ) -> RetrieverResult:
        user_content = RETRIEVE_USER_TEMPLATE.format(
            prompt=prompt,
            explorer_output=explorer_output.content,
        )
        response = self.router.create_completion(
            model_name=model_name,
            messages=[
                {"role": "system", "content": RETRIEVER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            **kwargs,
        )

        return RetrieverResult(
            content=response.choices[0].message.content or "",
            metadata={
                "source": "openai_chat_retriever",
                "model": model_name,
            },
        )

    def refine(
        self,
        prompt: str,
        current_context: RetrieverResult,
        generator_feedback: str,
        model_name: str,
        **kwargs,
    ) -> RetrieverResult:
        user_content = REFINE_USER_TEMPLATE.format(
            prompt=prompt,
            current_context=current_context.content,
            generator_feedback=generator_feedback,
        )
        response = self.router.create_completion(
            model_name=model_name,
            messages=[
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            **kwargs,
        )

        return RetrieverResult(
            content=response.choices[0].message.content or "",
            metadata={
                "source": "openai_chat_retriever",
                "model": model_name,
                "step": "refine",
            },
        )
