"""Typed settings for the FastAPI application.

Configuration is defined by code defaults and can be overridden with `.env` or
environment variables. Nested pipeline settings use `__` as a delimiter, e.g.
`EXPLORER__TYPE=neo4j_pagerank`.
"""

from functools import lru_cache
from typing import Any

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseModel):
    refinement_turns: int = 0
    save_steps: bool = True
    save_dir: str = "outputs"


class APIClientConfig(BaseModel):
    type: str = "google"
    model: str = "gemini-3-flash-preview"


class SearchWeaviateConfig(BaseModel):
    top_k: int = 10
    rerank_top_k: int = 3


class SearchConfig(BaseModel):
    type: str = "weaviate"
    weaviate: SearchWeaviateConfig = Field(default_factory=SearchWeaviateConfig)


class Neo4jConfig(BaseModel):
    max_level: int = 1
    direction: str = "both"
    steps: int = 5
    top_n: int = 15
    max_iterations: int = 20
    damping_factor: float = 0.85


class AgenticConfig(BaseModel):
    iterations: int = 5
    tools: list[str] = Field(default_factory=lambda: ["bfs_from_papers", "random_walk", "ppr"])
    tool_selection_prompt: str = "tool_selection_prompt.yaml"
    node_filtering_prompt: str = "node_filtering_prompt.yaml"
    model_name: str = "Qwen/Qwen3-4B"
    temperature: float = 0.7
    max_results_per_tool: int = 10
    include_abstracts: bool = True
    include_summary: bool = False
    selected_nodes_count_low: int = 2
    selected_nodes_count_high: int = 3


class ExplorerWeaviateConfig(BaseModel):
    url: str = "http://localhost:8080"
    collection: str = "ResearchPapers"
    top_k: int = 10
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    max_tokens: int = 8192
    grpc_port: int = 50051


class ExplorerConfig(BaseModel):
    type: str = "neo4j_pagerank"
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    agentic: AgenticConfig = Field(default_factory=AgenticConfig)
    weaviate: ExplorerWeaviateConfig = Field(default_factory=ExplorerWeaviateConfig)
    const_text: str = "No knowledge graph connected. Using placeholder context."


class RetrieverConfig(BaseModel):
    type: str = "llm"
    top_k: int = 5


class GeneratorConfig(BaseModel):
    type: str = "llm"
    temperature: float = 0.7


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    api_title: str = "Hypothesis Forge API"
    api_version: str = "0.3.0"
    pipeline_use_mock: bool = True
    database_url: str = "sqlite:///./hypothesis_forge.db"
    jwt_secret: str = "change-me-in-production"
    jwt_expire_hours: int = 168
    cors_origins: str | list[str] = Field(default_factory=lambda: ["*"])
    chat_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("CHAT_MODEL", "OPENAI_MODEL", "chat_model"),
    )
    openai_api_key: str = "dummy"
    openai_base_url: str | None = None

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    api_client: APIClientConfig = Field(default_factory=APIClientConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    explorer: ExplorerConfig = Field(default_factory=ExplorerConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)

    @property
    def cors_origin_list(self) -> list[str]:
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return self.cors_origins

    def pipeline_config(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline.model_dump(),
            "api_client": self.api_client.model_dump(),
            "search": self.search.model_dump(),
            "explorer": self.explorer.model_dump(),
            "retriever": self.retriever.model_dump(),
            "generator": self.generator.model_dump(),
        }


@lru_cache
def get_settings() -> APISettings:
    return APISettings()
