import json

from pydantic import ValidationError

from app.generator.base import BaseGenerator
from app.generator.prompts import (
    FEEDBACK_SYSTEM_PROMPT,
    FEEDBACK_USER_TEMPLATE,
    GENERATE_USER_TEMPLATE,
    GENERATOR_SYSTEM_PROMPT,
    HypothesesResponse,
)
from app.models import GeneratorResult, RetrieverResult
from app.openai_chat import OpenAIChatRouter


class OpenAIChatGenerator(BaseGenerator):
    """Generator that calls OpenAI-compatible Chat Completions directly."""

    def __init__(self, model_configs: dict[str, dict]) -> None:
        self.router = OpenAIChatRouter(model_configs)

    def generate(
        self,
        prompt: str,
        retriever_output: RetrieverResult,
        model_name: str,
        **kwargs,
    ) -> GeneratorResult:
        user_content = (
            GENERATE_USER_TEMPLATE.format(
                prompt=prompt,
                retriever_output=retriever_output.content,
            )
            + "\n\nReturn only valid JSON with this shape: "
            '{"hypotheses": ["first hypothesis", "second hypothesis"]}'
        )
        messages = [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            response = self.router.create_completion(
                model_name=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs,
            )
        except Exception:
            response = self.router.create_completion(
                model_name=model_name,
                messages=messages,
                **kwargs,
            )

        raw = response.choices[0].message.content or "{}"
        parsed = self._parse_hypotheses(raw)
        return GeneratorResult(
            hypotheses=parsed.hypotheses,
            metadata={
                "source": "openai_chat_generator",
                "model": model_name,
            },
        )

    def provide_feedback(
        self,
        prompt: str,
        retriever_output: RetrieverResult,
        model_name: str,
        **kwargs,
    ) -> str:
        user_content = FEEDBACK_USER_TEMPLATE.format(
            prompt=prompt,
            retriever_output=retriever_output.content,
        )
        response = self.router.create_completion(
            model_name=model_name,
            messages=[
                {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_hypotheses(raw: str) -> HypothesesResponse:
        try:
            return HypothesesResponse(**json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError("Generator response did not match HypothesesResponse JSON.") from exc
