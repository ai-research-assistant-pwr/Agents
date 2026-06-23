"""Typed settings for the FastAPI application.

Configuration is defined by code defaults and can be overridden with `.env` or
environment variables. Nested pipeline settings use `__` as a delimiter, e.g.
`EXPLORER__NEO4J__STEPS=8`.
"""

from functools import lru_cache
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseModel):
    refinement_turns: int = 0
    save_steps: bool = True
    save_dir: str = "outputs"


class SupportedModelConfig(BaseModel):
    provider: Literal["openai", "vertex_ai", "custom"]
    api_model: str
    base_url: str | None = None


class SearchWeaviateConfig(BaseModel):
    top_k: int = 10
    rerank_top_k: int = 3


class ApiWeaviateConfig(BaseModel):
    url: str = "http://localhost:8080"
    grpc_port: int = 50051
    collection: str = "ResearchPapers"
    embedding_host: str = "localhost"
    embedding_port: int = 8005
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_api_key: str | None = None
    embedding_url: str | None = None
    top_k: int = 7
    api_key: str | None = None


class SearchConfig(BaseModel):
    type: str = "weaviate"
    weaviate: SearchWeaviateConfig = Field(default_factory=SearchWeaviateConfig)
    api_weaviate: ApiWeaviateConfig = Field(default_factory=ApiWeaviateConfig)


class Neo4jRandomWalkConfig(BaseModel):
    steps: int = 5
    direction: str = "both"


class ExplorerConfig(BaseModel):
    neo4j: Neo4jRandomWalkConfig = Field(default_factory=Neo4jRandomWalkConfig)


class RetrieverConfig(BaseModel):
    type: str = "llm"
    top_k: int = 5


class GeneratorConfig(BaseModel):
    type: str = "llm"
    temperature: float = 0.7
    max_tokens: int = 2048


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    api_title: str = "Hypothesis Forge API"
    api_version: str = "0.3.0"
    pipeline_use_mock: bool = False
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
    google_api_key: str | None = None
    google_base_url: str | None = "https://generativelanguage.googleapis.com/v1beta/openai/"
    vllm_api_key: str | None = None
    vllm_base_url: str | None = "http://localhost:8021/v1"
    # When a persona is selected, generation is routed to this model (name must
    # appear in model_configs). Typically set to a more capable model such as
    # a GPT-5 variant. Leave unset to use the user-chosen generator model.
    persona_model: str | None = None

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    model_configs: dict[str, SupportedModelConfig] = Field(
        default_factory=lambda: {
            "gpt-5.4-mini": SupportedModelConfig(
                provider="openai",
                api_model="gpt-5.4-mini",
            ),
            "gemini-3.1-flash-lite": SupportedModelConfig(
                provider="vertex_ai",
                api_model="gemini-3.1-flash-lite",
            ),
            "Qwen3-4B": SupportedModelConfig(
                provider="custom",
                api_model="Qwen/Qwen3-4B",
            ),
        }
    )
    retriever_models: list[str] = Field(
        default_factory=lambda: [
            "gpt-5.4-mini",
            "gemini-3.1-flash-lite"
        ]
    )
    generator_models: list[str] = Field(
        default_factory=lambda: [
            "gpt-5.4-mini",
            "gemini-3.1-flash-lite",
            "Qwen3-4B",
        ]
    )
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
            "model_configs": {
                name: config.model_dump()
                for name, config in self.model_configs.items()
            },
            "retriever_models": self.retriever_models,
            "generator_models": self.generator_models,
            "search": self.search.model_dump(),
            "explorer": self.explorer.model_dump(),
            "retriever": self.retriever.model_dump(),
            "generator": self.generator.model_dump(),
        }

    @property
    def retriever_model_names(self) -> list[str]:
        return self.retriever_models

    @property
    def generator_model_names(self) -> list[str]:
        return self.generator_models

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        config = self.model_configs[model_name].model_dump()
        if config["provider"] == "openai":
            config["api_key"] = self.openai_api_key
            config["base_url"] = config["base_url"] or self.openai_base_url
        elif config["provider"] == "vertex_ai":
            config["api_key"] = self.google_api_key
            config["base_url"] = config["base_url"] or self.google_base_url
        elif config["provider"] == "custom":
            config["api_key"] = self.vllm_api_key
            config["base_url"] = config["base_url"] or self.vllm_base_url
        return config


@lru_cache
def get_settings() -> APISettings:
    return APISettings()
