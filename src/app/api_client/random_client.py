import random
from typing import TypeVar

from app.api_client.base import BaseAPIClient, CallResult, Message

T = TypeVar("T")


class RandomAPIClient(BaseAPIClient):
    """API client that routes each call to a randomly selected model.

    Models are chosen by weighted random selection. Each model is
    associated with a provider class (e.g. GoogleAPIClient) which is
    instantiated eagerly in __init__ — one instance per model.

    All provider classes must accept ``model: str`` as their sole
    constructor argument (API keys and other credentials are expected
    to be loaded from the environment inside the provider).

    Example::

        client = RandomAPIClient(
            providers={"google": GoogleAPIClient},
            models={
                "gemini-2.5-flash": {"provider": "google", "weight": 8},
                "gemini-2.0-flash": {"provider": "google", "weight": 2},
            },
        )
    """

    def __init__(
        self,
        providers: dict[str, type[BaseAPIClient]],
        models: dict[str, dict],
    ) -> None:
        """Initialise the client and eagerly instantiate all provider clients.

        Args:
            providers: Mapping of provider name to provider class.
                       e.g. ``{"google": GoogleAPIClient}``.
            models: Mapping of model name to configuration dict with keys:
                    - ``provider`` (str): key into ``providers``.
                    - ``weight`` (int): relative probability of selection;
                      higher values make the model more likely to be chosen.
        """
        if not models:
            raise ValueError("At least one model must be specified.")

        self._clients: dict[str, BaseAPIClient] = {}
        self._model_names: list[str] = []
        self._weights: list[int] = []

        for model_name, cfg in models.items():
            provider_name = cfg["provider"]
            weight = cfg["weight"]

            if provider_name not in providers:
                raise ValueError(
                    f"Model '{model_name}' references unknown provider "
                    f"'{provider_name}'. Available: {list(providers)}"
                )

            provider_cls = providers[provider_name]
            self._clients[model_name] = provider_cls(model=model_name)  # type: ignore[call-arg]
            self._model_names.append(model_name)
            self._weights.append(weight)

    def call(  # type: ignore[override]
        self, messages: list[Message], response_schema: type[T] | None = None
    ) -> CallResult[str] | CallResult[T]:
        """Route the call to a randomly weighted model.

        The model is re-sampled on every call, so successive calls may
        use different models. The selected model is always reported back
        via CallResult.model.

        Args:
            messages: Ordered list of conversation messages.
            response_schema: Optional Pydantic model or dataclass for
                structured output. Forwarded verbatim to the chosen client.

        Returns:
            A CallResult from the selected client, with .model set to the
            name of the model that was actually used.
        """
        model_name = random.choices(self._model_names, weights=self._weights, k=1)[0]
        client = self._clients[model_name]
        return client.call(messages, response_schema)  # type: ignore[return-value]
