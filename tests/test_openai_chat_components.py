from app.generator.openai_chat_generator import OpenAIChatGenerator
from app.models import ExplorerResult, RetrieverResult
from app.retriever.openai_chat_retriever import OpenAIChatRetriever


def test_openai_chat_retriever_passes_model_and_kwargs(monkeypatch):
    retriever = OpenAIChatRetriever(model_configs={"gpt-5.4-mini": {"api_model": "gpt-5.4-mini"}})
    calls = []

    def fake_create_completion(**kwargs):
        calls.append(kwargs)
        return _response("retrieved context")

    monkeypatch.setattr(retriever.router, "create_completion", fake_create_completion)

    result = retriever.retrieve(
        "question",
        ExplorerResult(content="explorer output"),
        model_name="gpt-5.4-mini",
        temperature=0.2,
    )

    assert result.content == "retrieved context"
    assert result.metadata["model"] == "gpt-5.4-mini"
    assert calls[0]["model_name"] == "gpt-5.4-mini"
    assert calls[0]["temperature"] == 0.2


def test_openai_chat_generator_parses_json_response(monkeypatch):
    generator = OpenAIChatGenerator(model_configs={"gpt-5.4-mini": {"api_model": "gpt-5.4-mini"}})
    calls = []

    def fake_create_completion(**kwargs):
        calls.append(kwargs)
        return _response('{"hypotheses": ["h1", "h2"]}')

    monkeypatch.setattr(generator.router, "create_completion", fake_create_completion)

    result = generator.generate(
        "question",
        RetrieverResult(content="retrieved context"),
        model_name="gpt-5.4-mini",
        temperature=0.7,
    )

    assert result.hypotheses == ["h1", "h2"]
    assert result.metadata["model"] == "gpt-5.4-mini"
    assert calls[0]["model_name"] == "gpt-5.4-mini"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["temperature"] == 0.7


def test_openai_chat_generator_falls_back_without_response_format(monkeypatch):
    generator = OpenAIChatGenerator(model_configs={"gpt-5.4-mini": {"api_model": "gpt-5.4-mini"}})
    calls = []

    def fake_create_completion(**kwargs):
        calls.append(kwargs)
        if "response_format" in kwargs:
            raise RuntimeError("response_format unsupported")
        return _response('{"hypotheses": ["fallback"]}')

    monkeypatch.setattr(generator.router, "create_completion", fake_create_completion)

    result = generator.generate(
        "question",
        RetrieverResult(content="retrieved context"),
        model_name="gpt-5.4-mini",
    )

    assert result.hypotheses == ["fallback"]
    assert len(calls) == 2
    assert "response_format" not in calls[1]


def test_openai_chat_generator_feedback(monkeypatch):
    generator = OpenAIChatGenerator(model_configs={"gpt-5.4-mini": {"api_model": "gpt-5.4-mini"}})

    def fake_create_completion(**kwargs):
        return _response("feedback")

    monkeypatch.setattr(generator.router, "create_completion", fake_create_completion)

    assert (
        generator.provide_feedback(
            "question",
            RetrieverResult(content="retrieved context"),
            model_name="gpt-5.4-mini",
        )
        == "feedback"
    )


def _response(content: str):
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()
