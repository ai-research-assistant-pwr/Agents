"""Weaviate search explorer backed by a vLLM-hosted embedding endpoint."""

import logging
from urllib.parse import urlparse

from openai import OpenAI

from app.explorer.base import BaseSearchExplorer

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised via runtime/test monkeypatching
    import weaviate as weaviate_client
except Exception:  # pragma: no cover - keep module importable even if client import is noisy
    class _WeaviatePlaceholder:
        def connect_to_custom(self, **kwargs):
            raise RuntimeError("weaviate client is not available")

    weaviate_client = _WeaviatePlaceholder()


class ApiWeaviateSearchExplorer(BaseSearchExplorer):
    """Search Weaviate using embeddings fetched from an OpenAI-compatible server."""

    def __init__(self, config: dict) -> None:
        weaviate_cfg = config.get("search", {}).get("api_weaviate", {})
        self.collection_name = weaviate_cfg.get("collection", "ResearchPapers")
        self.top_k = weaviate_cfg.get("top_k", 10)
        self.embedding_model = weaviate_cfg.get("embedding_model", "Qwen/Qwen3-Embedding-4B")
        self.embedding_host = weaviate_cfg.get("embedding_host", "localhost")
        self.embedding_port = weaviate_cfg.get("embedding_port", 8000)
        self.embedding_api_key = weaviate_cfg.get("embedding_api_key")

        url = weaviate_cfg.get("url", "http://localhost:8080")
        grpc_port = weaviate_cfg.get("grpc_port", 50051)
        api_key = weaviate_cfg.get("api_key")

        parsed = urlparse(url)
        http_host = parsed.hostname or "localhost"
        http_port = parsed.port or 8080
        http_secure = parsed.scheme == "https"

        logger.info(
            "Connecting to Weaviate at %s:%d (gRPC :%d)",
            http_host,
            http_port,
            grpc_port,
        )
        if not hasattr(weaviate_client, "connect_to_custom"):
            import weaviate as weaviate_client_local  # noqa: PLC0415

            globals()["weaviate_client"] = weaviate_client_local

        connect_kwargs = dict(
            http_host=http_host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=http_host,
            grpc_port=grpc_port,
            grpc_secure=False,
        )
        if api_key:
            from weaviate.classes.init import Auth  # noqa: PLC0415

            connect_kwargs["auth_credentials"] = Auth.api_key(api_key)
        self._client = weaviate_client.connect_to_custom(**connect_kwargs)
        self._collection = self._client.collections.get(self.collection_name)

        self._embed_client = OpenAI(
            api_key=self.embedding_api_key,
            base_url=f"http://{self.embedding_host}:{self.embedding_port}/v1",
        )

        logger.info(
            "ApiWeaviateSearchExplorer initialized: collection=%s top_k=%d embedder=%s:%d model=%s",
            self.collection_name,
            self.top_k,
            self.embedding_host,
            self.embedding_port,
            self.embedding_model,
        )

    def __del__(self) -> None:
        try:
            if hasattr(self, "_client"):
                self._client.close()
        except Exception:
            pass

    def _embed(self, text: str) -> list[float]:
        response = self._embed_client.embeddings.create(
            model=self.embedding_model,
            input=text.strip(),
        )
        return list(response.data[0].embedding)

    def search(self, prompt: str) -> list[str]:
        logger.info("Embedding query (%d chars)", len(prompt))
        vector = self._embed(prompt)

        logger.info(
            "Querying Weaviate collection=%s top_k=%d", self.collection_name, self.top_k
        )
        response = self._collection.query.near_vector(
            near_vector=vector,
            limit=self.top_k,
            return_properties=["paperId", "content", "type", "title"],
        )

        paper_ids: list[str] = []
        seen: set[str] = set()
        for obj in response.objects:
            paper_id = obj.properties.get("paperId")
            if paper_id and paper_id not in seen:
                seen.add(paper_id)
                paper_ids.append(paper_id)

        logger.info("Returning %d paper IDs", len(paper_ids))
        return paper_ids
