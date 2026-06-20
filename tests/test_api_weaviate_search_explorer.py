from app.explorer.api_weaviate_search_explorer import ApiWeaviateSearchExplorer


def test_api_weaviate_search_explorer_uses_embeddings_endpoint_and_skips_rerank(monkeypatch):
    query_calls = []
    embed_calls = []

    class FakeEmbeddings:
        def create(self, *, model, input):
            embed_calls.append({"model": model, "input": input})
            return type(
                "Resp",
                (),
                {"data": [type("Item", (), {"embedding": [0.1, 0.2, 0.3]})()]},
            )()

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.embeddings = FakeEmbeddings()

    class FakeCollectionQuery:
        def near_vector(self, **kwargs):
            query_calls.append(kwargs)
            return type(
                "Resp",
                (),
                {
                    "objects": [
                        type("Obj", (), {"properties": {"paperId": "p1"}})(),
                        type("Obj", (), {"properties": {"paperId": "p2"}})(),
                        type("Obj", (), {"properties": {"paperId": "p1"}})(),
                    ]
                },
            )()

    class FakeCollection:
        query = FakeCollectionQuery()

    class FakeClient:
        def __init__(self):
            self.collections = type("Collections", (), {"get": lambda self, name: FakeCollection()})()

        def close(self):
            pass

    monkeypatch.setattr(
        "app.explorer.api_weaviate_search_explorer.OpenAI",
        FakeOpenAI,
    )
    monkeypatch.setattr(
        "app.explorer.api_weaviate_search_explorer.weaviate_client.connect_to_custom",
        lambda **kwargs: FakeClient(),
    )

    explorer = ApiWeaviateSearchExplorer(
        {
            "search": {
                "api_weaviate": {
                    "url": "http://weaviate:8080",
                    "grpc_port": 50051,
                    "collection": "ResearchPapers",
                    "embedding_host": "embedder",
                    "embedding_port": 8000,
                    "embedding_model": "Qwen/Qwen3-Embedding-4B",
                    "top_k": 2,
                }
            }
        }
    )

    result = explorer.search("what is emergence?")

    assert result == ["p1", "p2"]
    assert embed_calls == [
        {
            "model": "Qwen/Qwen3-Embedding-4B",
            "input": "what is emergence?",
        }
    ]
    assert query_calls[0]["limit"] == 2
    assert "rerank" not in query_calls[0]
