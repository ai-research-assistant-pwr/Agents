"""Small OpenAI Chat Completions router used by pipeline LLM components."""

from typing import Any

from openai import OpenAI  # noqa: PLC0415

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
        client = self._client(config)
        return client.chat.completions.create(
            model=config["api_model"],
            messages=messages,
            **kwargs,
        )

    def _client(self, config: dict[str, Any]):
        api_key = config.get("api_key")

        return OpenAI(
            api_key=api_key,
            base_url=config.get("base_url"),
        )
