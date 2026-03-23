"""Weaviate vector-database explorer using Qwen3-Embedding-8B."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
import weaviate
from transformers import AutoModel, AutoTokenizer
from weaviate.classes.query import MetadataQuery

from app.explorer.base import BaseExplorer
from app.models import ExplorerResult

logger = logging.getLogger(__name__)


def _last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool the last real token hidden state (handles left/right padding)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


class WeaviateExplorer(BaseExplorer):
    """Explorer that queries a Weaviate vector database using Qwen3-Embedding-8B.

    Embeds the user prompt with Qwen/Qwen3-Embedding-8B and performs a
    nearest-neighbour search against the configured Weaviate collection.
    """

    def __init__(self, config: dict) -> None:
        weaviate_cfg = config.get("explorer", {}).get("weaviate", {})
        url = weaviate_cfg.get("url", "http://localhost:8080")
        self.collection_name = weaviate_cfg.get("collection", "ResearchPapers")
        self.top_k = weaviate_cfg.get("top_k", 10)
        self.max_tokens = weaviate_cfg.get("max_tokens", 8192)
        model_name = weaviate_cfg.get("embedding_model", "Qwen/Qwen3-Embedding-8B")

        parsed = urlparse(url)
        http_host = parsed.hostname or "localhost"
        http_port = parsed.port or 8080
        grpc_port = weaviate_cfg.get("grpc_port", 50051)

        logger.info("Connecting to Weaviate at %s:%d (gRPC :%d)", http_host, http_port, grpc_port)
        self._client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            http_secure=parsed.scheme == "https",
            grpc_host=http_host,
            grpc_port=grpc_port,
            grpc_secure=False,
        )
        self._collection = self._client.collections.get(self.collection_name)
        logger.info("Connected to collection: %s", self.collection_name)

        logger.info("Loading embedding model: %s", model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16,
        ).to(self.device)
        self.model.eval()
        logger.info("Embedding model loaded on %s", self.device)

    def __del__(self) -> None:
        try:
            if hasattr(self, "_client"):
                self._client.close()
        except Exception:
            pass

    def _embed(self, text: str) -> list[float]:
        """Embed a single string and return a normalised float list."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding[0].cpu().float().tolist()

    def explore(self, prompt: str) -> ExplorerResult:
        """Embed the prompt and retrieve the closest research paper chunks.

        Args:
            prompt: The user's research prompt / question.

        Returns:
            ExplorerResult with concatenated paper excerpts as content.
        """
        logger.info("Embedding query (%d chars)", len(prompt))
        vector = self._embed(prompt)

        logger.info("Querying Weaviate collection=%s top_k=%d", self.collection_name, self.top_k)
        response = self._collection.query.near_vector(
            near_vector=vector,
            limit=self.top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        sections: list[str] = []
        for obj in response.objects:
            p = obj.properties
            distance = obj.metadata.distance
            dist_str = f" (distance={distance:.4f})" if distance is not None else ""
            header = f"[{p.get('type', '')}] {p.get('title', 'Unknown')} (id={p.get('paperId', '')}){dist_str}"
            sections.append(f"{header}\n{p.get('content', '')}")

        full_content = (
            "\n\n---\n\n".join(sections)
            if sections
            else "No relevant papers found."
        )

        return ExplorerResult(
            content=full_content,
            metadata={
                "source": "weaviate_explorer",
                "prompt": prompt,
                "top_k": self.top_k,
                "num_results": len(response.objects),
                "collection": self.collection_name,
            },
        )
