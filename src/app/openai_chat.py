"""Small OpenAI Chat Completions router used by pipeline LLM components."""

from typing import Any


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
        from openai import OpenAI  # noqa: PLC0415

        api_key = config.get("api_key")
        if config.get("provider") == "vertex_ai":
            api_key = self._vertex_access_token()

        return OpenAI(
            api_key=api_key,
            base_url=config.get("base_url"),
        )

    @staticmethod
    def _vertex_access_token() -> str:
        from google.auth import default  # noqa: PLC0415
        import google.auth.transport.requests  # noqa: PLC0415

        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token
