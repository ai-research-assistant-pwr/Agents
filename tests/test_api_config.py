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
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "EXPLORER__NEO4J__STEPS",
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
    assert settings.retriever_model_names == ["gpt-5.4-mini", "gemini-3-flash-preview"]
    assert settings.generator_model_names == ["gpt-5.4-mini", "gemini-3-flash-preview"]
    assert settings.explorer.neo4j.steps == 5
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
    monkeypatch.setenv("EXPLORER__NEO4J__STEPS", "42")

    settings = APISettings(_env_file=None)
    config = settings.pipeline_config()

    assert config["explorer"]["neo4j"]["steps"] == 42
    assert config["retriever_models"] == ["gpt-5.4-mini", "gemini-3-flash-preview"]
    assert config["generator_models"] == ["gpt-5.4-mini", "gemini-3-flash-preview"]


def test_api_settings_supported_model_config(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    settings = APISettings(_env_file=None)
    config = settings.get_model_config("gpt-5.4-mini")

    assert config["provider"] == "openai"
    assert config["api_model"] == "gpt-5.4-mini"
    assert config["api_key"] == "openai-key"


def test_api_settings_vertex_model_config(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project-123")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")

    settings = APISettings(_env_file=None)
    config = settings.get_model_config("gemini-3-flash-preview")

    assert config["provider"] == "vertex_ai"
    assert config["api_model"] == "google/gemini-3-flash-preview"
    assert config["base_url"] == (
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/project-123/"
        "locations/europe-west4/endpoints/openapi"
    )


def test_pipeline_accepts_injected_config():
    calls = {"retriever": None, "generator": None}
    pipeline = Pipeline(
        search_explorer=_Search(),
        explorer=_Explorer(),
        retriever=_Retriever(calls),
        generator=_Generator(calls),
        config={"pipeline": {"save_steps": False, "refinement_turns": 0}},
    )

    result = pipeline.run(
        "question",
        retriever_model_name="gpt-5.4-mini",
        generator_model_name="gemini-3-flash-preview",
    )

    assert result.hypotheses == ["hypothesis"]
    assert pipeline.config["pipeline"]["save_steps"] is False
    assert calls == {
        "retriever": "gpt-5.4-mini",
        "generator": "gemini-3-flash-preview",
    }


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
    def __init__(self, calls: dict[str, str | None]) -> None:
        self.calls = calls

    def retrieve(
        self,
        prompt: str,
        explorer_output: ExplorerResult,
        model_name: str,
        **kwargs,
    ) -> RetrieverResult:
        self.calls["retriever"] = model_name
        return RetrieverResult(content="retrieved")

    def refine(
        self,
        prompt: str,
        current_context: RetrieverResult,
        generator_feedback: str,
        model_name: str,
        **kwargs,
    ) -> RetrieverResult:
        return current_context


class _Generator:
    def __init__(self, calls: dict[str, str | None]) -> None:
        self.calls = calls

    def generate(
        self,
        prompt: str,
        retriever_output: RetrieverResult,
        model_name: str,
        **kwargs,
    ) -> GeneratorResult:
        self.calls["generator"] = model_name
        return GeneratorResult(hypotheses=["hypothesis"], metadata={"model": "test"})

    def provide_feedback(
        self,
        prompt: str,
        retriever_output: RetrieverResult,
        model_name: str,
        **kwargs,
    ) -> str:
        return "feedback"
