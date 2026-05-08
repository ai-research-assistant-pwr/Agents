import os
from typing import TypeVar

from openai import OpenAI

from app.api_client.base import BaseAPIClient, CallResult, Message

T = TypeVar("T")


class OpenAIAPIClient(BaseAPIClient):
    """OpenAI API client using the Responses API.

    Loads the API key from the OPENAI_API_KEY environment variable.
    System messages are passed as the ``instructions`` parameter; user and
    assistant messages are forwarded as the ``input`` conversation turns.

    Structured output is handled via ``client.responses.parse``, which
    requires the response_schema to be a Pydantic BaseModel subclass.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it or add it to a .env file."
            )
        self.client = OpenAI(api_key=api_key)

    def call(  # type: ignore[override]
        self, messages: list[Message], response_schema: type[T] | None = None
    ) -> CallResult[str] | CallResult[T]:
        """Send messages to an OpenAI model via the Responses API.

        Args:
            messages: Ordered list of conversation messages.
                      Messages with role 'system' are concatenated and passed
                      as the ``instructions`` parameter.
                      Messages with role 'user' or 'assistant' become input turns.
            response_schema: Optional Pydantic BaseModel subclass. When provided,
                the response is parsed and returned as an instance of this type
                using the Responses API structured output (``responses.parse``).

        Returns:
            A CallResult with .model set to the OpenAI model identifier.
            .content is a plain string when response_schema is None, or a
            parsed instance of response_schema otherwise.
        """
        system_parts: list[str] = []
        input_messages: list[dict] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                input_messages.append({"role": msg.role, "content": msg.content})

        instructions = "\n".join(system_parts) if system_parts else None

        if response_schema is not None:
            kwargs: dict = dict(
                model=self.model,
                input=input_messages,
                text_format=response_schema,
                reasoning={"effort": "low"}
            )
            if instructions is not None:
                kwargs["instructions"] = instructions

            response = self.client.responses.parse(**kwargs)

            if response.output_parsed is None:
                raise RuntimeError(
                    "OpenAI API returned an empty structured response. "
                    "Check that the model supports structured output."
                )
            return CallResult(content=response.output_parsed, model=self.model)  # type: ignore[return-value]

        kwargs = dict(model=self.model, input=input_messages)
        if instructions is not None:
            kwargs["instructions"] = instructions

        response = self.client.responses.create(**kwargs)

        if not response.output_text:
            raise RuntimeError("OpenAI API returned an empty response.")
        return CallResult(content=response.output_text, model=self.model)
