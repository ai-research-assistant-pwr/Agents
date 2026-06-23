"""Small OpenAI Chat Completions router used by pipeline LLM components."""

import logging
from typing import Any

from openai import OpenAI  # noqa: PLC0415

logger = logging.getLogger(__name__)


class OpenAIChatRouter:
    def __init__(self, model_configs: dict[str, dict[str, Any]]) -> None:
        self.model_configs = model_configs

    def create_completion(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        **kwargs,
    ):
        config = self.model_configs[model_name]
        logger.debug(
            "create_completion → model=%s  endpoint=%s  params=%s",
            config["api_model"],
            config.get("base_url"),
            kwargs,
        )
        client = self._client(config)
        response = client.chat.completions.create(
            model=config["api_model"],
            messages=messages,
            **kwargs,
        )
        logger.debug(
            "create_completion ← finish_reason=%s  usage=%s",
            response.choices[0].finish_reason,
            response.usage,
        )
        return response

    def _client(self, config: dict[str, Any]):
        api_key = config.get("api_key")

        return OpenAI(
            api_key=api_key,
            base_url=config.get("base_url"),
        )
