# Integration Guide — Hypothesis Forge

This document explains how to swap the mock pipeline for real models and
databases so the app can be deployed as a production service.

---

## Architecture overview

```
Frontend (React/Vite)
  └── POST /api/sessions            ← question in, Session out
  └── POST /api/sessions/{id}/select
  └── POST /api/sessions/{id}/chat  ← follow-up messages
  └── GET  /api/sessions[/{id}]

Backend (FastAPI)
  ├── src/api/main.py               ← HTTP routing
  ├── src/api/pipeline.py           ← run_pipeline() / generate_chat_reply()
  ├── src/api/session_store.py      ← DB read/write
  └── src/api/db.py                 ← SQLAlchemy engine

Pipeline (existing codebase)
  ├── search_explorer               ← WeaviateSearchExplorer
  ├── explorer                      ← Neo4jBFSExplorer / Neo4jPageRankExplorer / …
  ├── retriever                     ← APILLMRetriever
  └── generator                     ← APILLMGenerator
```

---

## 1. PostgreSQL

Set `DATABASE_URL` to a PostgreSQL connection string:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/hypothesis_forge"
```

Tables are created automatically on first start (`init_db()` is called at
import time in `session_store.py`).  For production, pin the schema with
Alembic migrations instead.

**Local dev with Docker:**

```bash
docker run -d --name hf-postgres \
  -e POSTGRES_USER=hf \
  -e POSTGRES_PASSWORD=hf \
  -e POSTGRES_DB=hypothesis_forge \
  -p 5432:5432 \
  postgres:16
export DATABASE_URL="postgresql://hf:hf@localhost:5432/hypothesis_forge"
```

---

## 2. LLM API keys

Pick one or more providers and export the matching key:

| Provider | Env var | Default model |
|----------|---------|---------------|
| Google   | `GOOGLE_API_KEY` | `gemini-3-flash-preview` |
| OpenAI   | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Cerebras | `CEREBRAS_API_KEY` | — |

Configure the active provider in `config/app/config.yaml`:

```yaml
api_client:
  type: "google"          # or "openai" / "cerebras"
  model: "gemini-3-flash-preview"
```

---

## 3. Knowledge graph (Neo4j)

Start a Neo4j instance with your paper graph loaded:

```bash
docker run -d --name hf-neo4j \
  -e NEO4J_AUTH=neo4j/password \
  -p 7474:7474 -p 7687:7687 \
  neo4j:5
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
```

Set the explorer type in `config/app/config.yaml`:

```yaml
explorer:
  type: "neo4j_pagerank"     # or neo4j_bfs / neo4j_random_walk / agentic
  neo4j:
    top_n: 15
    max_iterations: 20
    damping_factor: 0.85
```

---

## 4. Weaviate (paper search)

```bash
docker run -d --name hf-weaviate \
  -p 8080:8080 -p 50051:50051 \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

Configure in `config/app/config.yaml`:

```yaml
search:
  type: "weaviate"
  weaviate:
    top_k: 10
    rerank_top_k: 3
```

---

## 5. Enable the real pipeline

Once the services above are running:

```bash
export PIPELINE_USE_MOCK=false
python run_api.py
```

The single toggle `PIPELINE_USE_MOCK=false` activates `_run_real()` in
`src/api/pipeline.py`.  The function wires up the components from
`config/app/config.yaml` — no other code changes needed.

---

## 6. Chat continuation with real LLM

`generate_chat_reply()` in `src/api/pipeline.py` currently raises
`NotImplementedError` in real mode.  To wire it:

```python
# src/api/pipeline.py  →  _real_chat_reply()

def _real_chat_reply(session: SessionOut, user_message: str) -> str:
    _ensure_src_on_path()
    from app.api_client.google_client import GoogleAPIClient
    from app.api_client.base import Message as LLMMessage

    client = GoogleAPIClient(model="gemini-3-flash-preview")

    selected_hyp = next(
        (h for h in session.hypotheses if h.id == session.selectedId), None
    )
    context = "\n\n".join(
        f"{m.role.upper()}: {m.content}" for m in session.messages[-6:]
    )

    system = (
        "You are a research assistant helping to develop a scientific hypothesis. "
        f"The user selected hypothesis {session.selectedId}: "
        f'"{selected_hyp.headline if selected_hyp else "unknown"}". '
        f"Their rationale: {session.rationale or '(none provided)'}. "
        "Continue the scientific discussion helpfully and rigorously."
    )

    messages = [
        LLMMessage(role="system", content=system),
    ]
    if context:
        messages.append(LLMMessage(role="user", content=f"Conversation so far:\n{context}"))
    messages.append(LLMMessage(role="user", content=user_message))

    return client.call(messages).content
```

---

## 7. Returning richer hypothesis structure from the generator

The current `APILLMGenerator` returns plain-text hypothesis strings.  To get
structured output (headline, classification, drawnFrom, etc.) that populates
the hypothesis cards fully:

**Step 1 — update `HypothesesResponse` in `src/app/generator/prompts.py`:**

```python
from pydantic import BaseModel

class StructuredHypothesis(BaseModel):
    classification: str
    headline: str
    statement: str
    drawnFrom: list[str]
    falsifiablePrediction: str

class HypothesesResponse(BaseModel):
    hypotheses: list[StructuredHypothesis]
```

**Step 2 — update `APILLMGenerator.generate()` in `src/app/generator/api_llm_generator.py`:**

The call already passes `response_schema=HypothesesResponse`.  With the
updated schema the LLM will return structured JSON automatically (works with
Google GenAI and OpenAI structured outputs).

**Step 3 — update `_run_real()` in `src/api/pipeline.py`** to read the
structured fields instead of calling `_wrap_hypotheses()`.

---

## 8. Knowledge graph → frontend

The backend stores `knowledgeGraph` in the session.  To populate it from real
neo4j data, update `_run_real()` to extract nodes and edges from the explorer
metadata:

```python
# After app.run():
kg_data = app.explorer.last_graph  # add this property to your explorer
if kg_data:
    knowledge_graph = KnowledgeGraph(
        nodes=[KGNode(id=n["id"], label=n["title"], type="concept")
               for n in kg_data["nodes"]],
        edges=[KGEdge(source=e["from"], target=e["to"], relation=e["type"])
               for e in kg_data["edges"]],
    )
```

---

## 9. Deployment

### Docker Compose (recommended)

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
    environment:
      DATABASE_URL: postgresql://hf:hf@db/hypothesis_forge
      PIPELINE_USE_MOCK: "false"
      GOOGLE_API_KEY: ${GOOGLE_API_KEY}
    ports: ["8000:8000"]
    depends_on: [db]

  db:
    image: postgres:16
    environment:
      POSTGRES_USER: hf
      POSTGRES_PASSWORD: hf
      POSTGRES_DB: hypothesis_forge
    volumes: ["pgdata:/var/lib/postgresql/data"]

  frontend:
    build: ./frontend
    ports: ["80:80"]

volumes:
  pgdata:
```

### Environment variables summary

| Variable | Default | Required for prod |
|----------|---------|------------------|
| `DATABASE_URL` | `sqlite:///./hypothesis_forge.db` | Yes — PostgreSQL URL |
| `PIPELINE_USE_MOCK` | `true` | Set to `false` |
| `GOOGLE_API_KEY` | — | Yes (or another provider key) |
| `NEO4J_URI` | — | Yes (if using neo4j explorer) |
| `NEO4J_USER` | — | Yes |
| `NEO4J_PASSWORD` | — | Yes |
| `CORS_ORIGINS` | `*` | Restrict to your frontend origin |
| `PORT` | `8000` | Optional override |

### Frontend build

```bash
cd frontend
VITE_USE_MOCK=false npm run build
# → dist/ can be served by nginx, Caddy, or a CDN.
# Proxy /api/* to the backend origin.
```

---

## 10. Testing

```bash
# Backend tests (SQLite in-memory, no external services)
python -m pytest

# Frontend type check
cd frontend && npm run typecheck
```

Tests always use `PIPELINE_USE_MOCK=true` and `DATABASE_URL=sqlite:///:memory:`,
set in `tests/conftest.py`.  Add integration tests for the real pipeline by
creating a separate `tests/test_pipeline_real.py` that guards itself with
`pytest.mark.skipif(not os.getenv("PIPELINE_USE_MOCK") == "false", ...)`.
