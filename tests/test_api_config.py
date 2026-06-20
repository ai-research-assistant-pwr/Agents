import app as app_package
from api.config import APISettings
from app.app import Pipeline
from app.models import ExplorerResult, GeneratorResult, RetrieverResult


ENV_VARS = [
    "API_TITLE",
    "API_VERSION",
    "PIPELINE_USE_MOCK",
    "DATABASE_URL",
    "JWT_SECRET",
    "JWT_EXPIRE_HOURS",
    "CORS_ORIGINS",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "CHAT_MODEL",
    "OPENAI_MODEL",
    "API_CLIENT__TYPE",
    "API_CLIENT__MODEL",
    "EXPLORER__TYPE",
    "EXPLORER__NEO4J__TOP_N",
]


def _clear_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_api_settings_defaults(monkeypatch):
    _clear_env(monkeypatch)

    settings = APISettings(_env_file=None)

    assert settings.api_title == "Hypothesis Forge API"
    assert settings.pipeline_use_mock is True
    assert settings.database_url == "sqlite:///./hypothesis_forge.db"
    assert settings.api_client.type == "google"
    assert settings.explorer.neo4j.top_n == 15
    assert settings.cors_origin_list == ["*"]


def test_api_settings_flat_env_overrides(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PIPELINE_USE_MOCK", "false")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("JWT_EXPIRE_HOURS", "24")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    monkeypatch.setenv("OPENAI_API_KEY", "local-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("CHAT_MODEL", "local-chat")

    settings = APISettings(_env_file=None)

    assert settings.pipeline_use_mock is False
    assert settings.database_url == "sqlite:///:memory:"
    assert settings.jwt_secret == "test-secret"
    assert settings.jwt_expire_hours == 24
    assert settings.cors_origin_list == ["http://localhost:3000", "http://localhost:5173"]
    assert settings.openai_api_key == "local-key"
    assert settings.openai_base_url == "http://localhost:11434/v1"
    assert settings.chat_model == "local-chat"


def test_api_settings_openai_model_fallback(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL", "fallback-model")

    settings = APISettings(_env_file=None)

    assert settings.chat_model == "fallback-model"


def test_api_settings_nested_env_overrides(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("API_CLIENT__TYPE", "openai_compatible")
    monkeypatch.setenv("API_CLIENT__MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EXPLORER__TYPE", "neo4j_pagerank")
    monkeypatch.setenv("EXPLORER__NEO4J__TOP_N", "42")

    settings = APISettings(_env_file=None)
    config = settings.pipeline_config()

    assert config["api_client"]["type"] == "openai_compatible"
    assert config["api_client"]["model"] == "gpt-4o-mini"
    assert config["explorer"]["type"] == "neo4j_pagerank"
    assert config["explorer"]["neo4j"]["top_n"] == 42


def test_pipeline_accepts_injected_config():
    pipeline = Pipeline(
        search_explorer=_Search(),
        explorer=_Explorer(),
        retriever=_Retriever(),
        generator=_Generator(),
        config={"pipeline": {"save_steps": False, "refinement_turns": 0}},
    )

    result = pipeline.run("question")

    assert result.hypotheses == ["hypothesis"]
    assert pipeline.config["pipeline"]["save_steps"] is False


def test_app_package_exports_pipeline_without_app_alias():
    assert app_package.Pipeline is Pipeline
    assert not hasattr(app_package, "App")


class _Search:
    def search(self, prompt: str) -> list[str]:
        return ["paper"]


class _Explorer:
    def explore(self, prompt: str, paper_ids: list[str]) -> ExplorerResult:
        return ExplorerResult(content="explored")


class _Retriever:
    def retrieve(self, prompt: str, explorer_output: ExplorerResult) -> RetrieverResult:
        return RetrieverResult(content="retrieved")

    def refine(
        self,
        prompt: str,
        current_context: RetrieverResult,
        generator_feedback: str,
    ) -> RetrieverResult:
        return current_context


class _Generator:
    def generate(self, prompt: str, retriever_output: RetrieverResult) -> GeneratorResult:
        return GeneratorResult(hypotheses=["hypothesis"], metadata={"model": "test"})

    def provide_feedback(self, prompt: str, retriever_output: RetrieverResult) -> str:
        return "feedback"
