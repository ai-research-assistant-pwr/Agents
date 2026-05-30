import os
from typing import TypeVar

from google import genai
from google.genai import types

from app.api_client.base import BaseAPIClient, CallResult, Message

T = TypeVar("T")


class GoogleAPIClient(BaseAPIClient):
    """Google Gemini API client using the google-genai SDK.

    Loads the API key from the GOOGLE_API_KEY environment variable.
    """

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.model = model
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Set it or add it to a .env file."
            )
        self.client = genai.Client(api_key=api_key)

    def call(  # type: ignore[override]
        self,
        messages: list[Message],
        response_schema: type[T] | None = None,
        temperature: float | None = 1.0,  # Zmiana: Domyślna temperatura
        presence_penalty: float | None = 0.0,  # Zmiana: Domyślny presence_penalty
        frequency_penalty: float | None = 0.0,  # Zmiana: Domyślny frequency_penalty
    ) -> CallResult[str] | CallResult[T]:
        """Send messages to the Google Gemini model.

        Args:
            messages: Ordered list of conversation messages.
                      Messages with role 'system' are passed as system_instruction.
                      Messages with role 'user' or 'assistant' become conversation turns.
            response_schema: Optional Pydantic model or dataclass. When provided,
                the response is returned as a parsed instance of this type using
                Gemini's native structured output (JSON mode).

        Returns:
            A CallResult with .model set to the Gemini model identifier.
            .content is a plain string when response_schema is None, or a
            parsed instance of response_schema otherwise.
        """
        system_parts: list[str] = []
        contents: list[types.Content] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        config = types.GenerateContentConfig(
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        if system_parts:
            config.system_instruction = "\n".join(system_parts)

        if response_schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema  # type: ignore[assignment]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,  # type: ignore[arg-type]
            config=config,
        )

        if response_schema is not None:
            if response.parsed is None:
                raise RuntimeError(
                    "Google API returned an empty structured response. "
                    "Check that the model supports structured output."
                )
            return CallResult(content=response.parsed, model=self.model)  # type: ignore[return-value]

        if response.text is None:
            raise RuntimeError("Google API returned an empty response.")
        return CallResult(content=response.text, model=self.model)
