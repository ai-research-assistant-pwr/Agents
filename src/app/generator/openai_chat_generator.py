import json
import logging
from typing import Type

from pydantic import BaseModel, ValidationError

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
from app.openai_chat import OpenAIChatRouter

logger = logging.getLogger(__name__)

_MAX_RETRIES = 6


def _correction_messages(original_messages: list[dict], raw: str) -> list[dict]:
    schema = json.dumps(HypothesesResponse.model_json_schema(), indent=2)
    return [
        *original_messages,
        {"role": "assistant", "content": raw},
        {"role": "user", "content": (
            "Your previous response could not be parsed. "
            f"Respond with ONLY a valid JSON object conforming to this schema:\n{schema}"
        )},
    ]


def _json_schema_format(schema: Type[BaseModel]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        },
    }


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
        persona = kwargs.pop("persona", None)
        system_prompt = GENERATOR_SYSTEM_PROMPT
        if persona:
            system_prompt = system_prompt + build_persona_section(persona)

        user_content = GENERATE_USER_TEMPLATE.format(
            prompt=prompt,
            retriever_output=retriever_output.content,
        )
        original_messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        messages = original_messages

        last_exc: Exception = ValueError("No attempts made")
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.router.create_completion(
                    model_name=model_name,
                    messages=messages,
                    response_format=_json_schema_format(HypothesesResponse),
                    **kwargs,
                )

                message = response.choices[0].message
                raw = message.content or "{}"
                reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
                if reasoning:
                    logger.info("Generator reasoning:\n%s", reasoning)
                parsed = self._parse_hypotheses(raw)
                if attempt > 1:
                    logger.info("Generator parse succeeded on attempt %d", attempt)
                return GeneratorResult(
                    hypotheses=parsed.hypotheses,
                    metadata={"source": "openai_chat_generator", "model": model_name},
                )
            except ValueError as exc:
                last_exc = exc
                logger.warning("Generator attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc)
                if attempt < _MAX_RETRIES:
                    messages = _correction_messages(original_messages, raw)

        raise last_exc

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
            return HypothesesResponse.model_validate_json(raw)
        except ValidationError as exc:
            logger.debug("Unparseable generator output:\n%s", raw)
            raise ValueError("Generator response did not match HypothesesResponse JSON.") from exc
