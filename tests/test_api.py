"""Integration tests for all REST API endpoints.

All tests use the mock pipeline (PIPELINE_USE_MOCK=true) and an in-memory
SQLite database (DATABASE_URL=sqlite:///:memory:) — both set in conftest.py.
No external services are required.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api import session_store
from api.db import Base, engine


@pytest.fixture(autouse=True)
def clear_sessions():
    # Re-create tables for each test (in-memory DB is shared via StaticPool)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session_store.clear()
    yield
    session_store.clear()


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/sessions
# ---------------------------------------------------------------------------


def test_create_session_returns_valid_structure(client):
    resp = client.post("/api/sessions", json={"question": "What drives ICL emergence?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["question"] == "What drives ICL emergence?"
    assert body["sessionId"]
    assert body["createdAt"]

    exp = body["exploration"]
    assert isinstance(exp["nodesTraversed"], int)
    assert isinstance(exp["durationSec"], float)

    hyps = body["hypotheses"]
    assert len(hyps) == 2
    assert hyps[0]["id"] == "A"
    assert hyps[1]["id"] == "B"
    for h in hyps:
        assert h["headline"]
        assert h["statement"]
        assert isinstance(h["drawnFrom"], list)

    assert len(body["reasoningTrace"]) > 0
    assert body["knowledgeGraph"] is not None
    assert len(body["knowledgeGraph"]["nodes"]) > 0


def test_create_session_missing_question_returns_422(client):
    assert client.post("/api/sessions", json={}).status_code == 422


def test_create_session_is_persisted(client):
    client.post("/api/sessions", json={"question": "MoE routing"})
    assert len(session_store.list_all()) == 1


# ---------------------------------------------------------------------------
# GET /api/sessions
# ---------------------------------------------------------------------------


def test_list_sessions_empty(client):
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    assert resp.json() == {"sessions": []}


def test_list_sessions_shows_awaiting_status(client):
    client.post("/api/sessions", json={"question": "Attention mechanism?"})
    sessions = client.get("/api/sessions").json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["status"] == "awaiting"
    assert "Attention mechanism" in sessions[0]["title"]


# ---------------------------------------------------------------------------
# GET /api/sessions/{id}
# ---------------------------------------------------------------------------


def test_get_session_returns_full_data(client):
    sid = _make_session(client, "Full data test")
    resp = client.get(f"/api/sessions/{sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sessionId"] == sid
    assert body["question"] == "Full data test"
    assert body["reasoningTrace"]
    assert body["knowledgeGraph"]


def test_get_session_unknown_returns_404(client):
    assert client.get("/api/sessions/no-such-id").status_code == 404


# ---------------------------------------------------------------------------
# POST /api/sessions/{id}/select
# ---------------------------------------------------------------------------


def test_select_hypothesis_a(client):
    sid = _make_session(client, "Select A")
    resp = client.post(f"/api/sessions/{sid}/select", json={"selectedId": "A"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_select_hypothesis_b_with_rationale(client):
    sid = _make_session(client, "Select B")
    resp = client.post(
        f"/api/sessions/{sid}/select",
        json={"selectedId": "B", "rationale": "More rigorous."},
    )
    assert resp.status_code == 200


def test_select_unknown_session_returns_404(client):
    assert (
        client.post("/api/sessions/nope/select", json={"selectedId": "A"}).status_code
        == 404
    )


def test_select_invalid_id_returns_422(client):
    sid = _make_session(client, "Bad ID")
    assert (
        client.post(f"/api/sessions/{sid}/select", json={"selectedId": "C"}).status_code
        == 422
    )


def test_select_updates_status_to_developed(client):
    sid = _make_session(client, "Status check")
    client.post(f"/api/sessions/{sid}/select", json={"selectedId": "A"})
    sessions = client.get("/api/sessions").json()["sessions"]
    assert sessions[0]["status"] == "developed"


def test_selection_persisted_in_get_session(client):
    sid = _make_session(client, "Persistence")
    client.post(
        f"/api/sessions/{sid}/select",
        json={"selectedId": "B", "rationale": "noted"},
    )
    full = client.get(f"/api/sessions/{sid}").json()
    assert full["selectedId"] == "B"
    assert full["rationale"] == "noted"


# ---------------------------------------------------------------------------
# POST /api/sessions/{id}/chat
# ---------------------------------------------------------------------------


def test_chat_returns_reply(client):
    sid = _make_session(client, "Chat test")
    client.post(f"/api/sessions/{sid}/select", json={"selectedId": "A"})
    resp = client.post(
        f"/api/sessions/{sid}/chat",
        json={"message": "Can you elaborate on the mechanism?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "reply" in body
    assert body["reply"]


def test_chat_persists_messages(client):
    sid = _make_session(client, "Message persistence")
    client.post(f"/api/sessions/{sid}/select", json={"selectedId": "A"})
    client.post(f"/api/sessions/{sid}/chat", json={"message": "Tell me more."})
    full = client.get(f"/api/sessions/{sid}").json()
    assert len(full["messages"]) == 2  # user + assistant
    assert full["messages"][0]["role"] == "user"
    assert full["messages"][1]["role"] == "assistant"


def test_chat_unknown_session_returns_404(client):
    assert (
        client.post("/api/sessions/ghost/chat", json={"message": "hi"}).status_code
        == 404
    )


def test_multi_turn_conversation(client):
    sid = _make_session(client, "Multi-turn")
    client.post(f"/api/sessions/{sid}/select", json={"selectedId": "B"})
    client.post(f"/api/sessions/{sid}/chat", json={"message": "First question."})
    client.post(f"/api/sessions/{sid}/chat", json={"message": "Follow-up."})
    full = client.get(f"/api/sessions/{sid}").json()
    assert len(full["messages"]) == 4  # 2 pairs


# ---------------------------------------------------------------------------
# Full end-to-end flow
# ---------------------------------------------------------------------------


def test_full_flow(client):
    question = "Why do large models exhibit emergent abilities?"

    create_resp = client.post("/api/sessions", json={"question": question})
    assert create_resp.status_code == 200
    session = create_resp.json()
    sid = session["sessionId"]

    # Two hypotheses, each complete
    for h in session["hypotheses"]:
        assert h["id"] in ("A", "B")
        assert h["headline"]

    # Reasoning trace present
    assert session["reasoningTrace"]

    # Select and check status
    client.post(f"/api/sessions/{sid}/select", json={"selectedId": "B"})
    assert client.get("/api/sessions").json()["sessions"][0]["status"] == "developed"

    # Chat turn
    chat_resp = client.post(
        f"/api/sessions/{sid}/chat", json={"message": "What experiments would test this?"}
    )
    assert chat_resp.status_code == 200

    # Full session via GET has all data
    full = client.get(f"/api/sessions/{sid}").json()
    assert full["selectedId"] == "B"
    assert len(full["messages"]) == 2
    assert full["knowledgeGraph"]["nodes"]


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------


def test_register_returns_token(client):
    resp = client.post(
        "/api/auth/register",
        json={"username": "alice", "email": "alice@example.com", "password": "secret123"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"]
    assert body["user"]["username"] == "alice"
    assert body["user"]["email"] == "alice@example.com"


def test_register_duplicate_username_returns_409(client):
    data = {"username": "bob", "email": "bob@example.com", "password": "pass"}
    client.post("/api/auth/register", json=data)
    resp = client.post("/api/auth/register", json={**data, "email": "other@example.com"})
    assert resp.status_code == 409


def test_login_returns_token(client):
    client.post(
        "/api/auth/register",
        json={"username": "carol", "email": "carol@example.com", "password": "pw123"},
    )
    resp = client.post(
        "/api/auth/login",
        json={"username_or_email": "carol", "password": "pw123"},
    )
    assert resp.status_code == 200
    assert resp.json()["token"]


def test_login_wrong_password_returns_401(client):
    client.post(
        "/api/auth/register",
        json={"username": "dave", "email": "dave@example.com", "password": "correct"},
    )
    resp = client.post(
        "/api/auth/login",
        json={"username_or_email": "dave", "password": "wrong"},
    )
    assert resp.status_code == 401


def test_me_requires_auth(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_returns_user_with_valid_token(client):
    reg = client.post(
        "/api/auth/register",
        json={"username": "eve", "email": "eve@example.com", "password": "pw"},
    )
    token = reg.json()["token"]
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "eve"


def test_sessions_are_scoped_to_user(client):
    # Register two users
    r1 = client.post(
        "/api/auth/register",
        json={"username": "u1", "email": "u1@example.com", "password": "pw"},
    )
    r2 = client.post(
        "/api/auth/register",
        json={"username": "u2", "email": "u2@example.com", "password": "pw"},
    )
    t1 = r1.json()["token"]
    t2 = r2.json()["token"]

    # u1 creates a session
    client.post(
        "/api/sessions",
        json={"question": "User 1 question"},
        headers={"Authorization": f"Bearer {t1}"},
    )

    # u1 sees their session; u2 sees none
    s1 = client.get("/api/sessions", headers={"Authorization": f"Bearer {t1}"}).json()["sessions"]
    s2 = client.get("/api/sessions", headers={"Authorization": f"Bearer {t2}"}).json()["sessions"]
    assert len(s1) == 1
    assert len(s2) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(client: TestClient, question: str) -> str:
    resp = client.post("/api/sessions", json={"question": question})
    assert resp.status_code == 200
    return resp.json()["sessionId"]
