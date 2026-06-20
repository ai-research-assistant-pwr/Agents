"""FastAPI application for the Hypothesis Forge pipeline.

Endpoints
---------
POST  /api/auth/register                    Register a new user.
POST  /api/auth/login                       Log in, receive a JWT.
GET   /api/auth/me                          Get current user (requires JWT).
POST  /api/sessions                         Run pipeline, return two hypotheses.
GET   /api/sessions                         List sessions for the current user.
GET   /api/sessions/{id}                    Fetch a full session.
POST  /api/sessions/{id}/select             Save hypothesis choice + rationale.
POST  /api/sessions/{id}/chat               Send a follow-up message.

Authentication
--------------
Send  Authorization: Bearer <token>  on all session endpoints to scope data
per user.  Without the header the user is "anonymous" — backward-compatible
with tests and dev use.

Configuration
-------------
PIPELINE_USE_MOCK   "true" (default) -> mock, no external services.
DATABASE_URL        SQLAlchemy URL; defaults to sqlite:///./hypothesis_forge.db.
JWT_SECRET          Secret for signing tokens. Override in production.
CORS_ORIGINS        Comma-separated allowed origins (defaults to "*").
"""

from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import session_store
from api.auth import (
    authenticate_user,
    create_token,
    get_current_user,
    get_user_by_id,
    register_user,
)
from api.config import get_settings
from api.models import (
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    LoginRequest,
    Message,
    RegisterRequest,
    SelectRequest,
    SelectResponse,
    SessionListItem,
    SessionListOut,
    SessionOut,
    TokenOut,
    UserOut,
)
from api.pipeline import generate_chat_reply, run_pipeline

settings = get_settings()

app = FastAPI(title=settings.api_title, version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------


@app.post("/api/auth/register", response_model=TokenOut)
def auth_register(body: RegisterRequest) -> TokenOut:
    user = register_user(body.username, body.email, body.password)
    token = create_token(user.id)
    return TokenOut(token=token, user=UserOut(id=user.id, username=user.username, email=user.email))


@app.post("/api/auth/login", response_model=TokenOut)
def auth_login(body: LoginRequest) -> TokenOut:
    user = authenticate_user(body.username_or_email, body.password)
    token = create_token(user.id)
    return TokenOut(token=token, user=UserOut(id=user.id, username=user.username, email=user.email))


@app.get("/api/auth/me", response_model=UserOut)
def auth_me(user_id: str = Depends(get_current_user)) -> UserOut:
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="Not authenticated.")
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserOut(id=user.id, username=user.username, email=user.email)


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


@app.post("/api/sessions", response_model=SessionOut)
def create_session(
    body: CreateSessionRequest,
    user_id: str = Depends(get_current_user),
) -> SessionOut:
    session = run_pipeline(body.question)
    session_store.save_session(session, user_id=user_id)
    return session


# ---------------------------------------------------------------------------
# Session retrieval
# ---------------------------------------------------------------------------


@app.get("/api/sessions", response_model=SessionListOut)
def list_sessions(user_id: str = Depends(get_current_user)) -> SessionListOut:
    all_sessions = session_store.list_all(user_id=user_id)
    items: list[SessionListItem] = []
    for s in all_sessions:
        status = "developed" if s.selectedId else "awaiting"
        title = s.question[:60] + ("…" if len(s.question) > 60 else "")
        items.append(
            SessionListItem(id=s.sessionId, title=title, createdAt=s.createdAt, status=status)
        )
    return SessionListOut(sessions=items)


@app.get("/api/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: str) -> SessionOut:
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    return session


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@app.post("/api/sessions/{session_id}/select", response_model=SelectResponse)
def select_hypothesis(session_id: str, body: SelectRequest) -> SelectResponse:
    ok = session_store.record_selection(session_id, body)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    return SelectResponse(ok=True)


# ---------------------------------------------------------------------------
# Conversation continuation
# ---------------------------------------------------------------------------


@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, body: ChatRequest) -> ChatResponse:
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    now = datetime.now(timezone.utc).isoformat()
    session_store.add_message(session_id, Message(role="user", content=body.message, createdAt=now))

    # Refresh so generate_chat_reply sees the updated message history.
    session = session_store.get_session(session_id)
    reply_text = generate_chat_reply(session, body.message)  # type: ignore[arg-type]

    session_store.add_message(session_id, Message(role="assistant", content=reply_text, createdAt=now))
    return ChatResponse(reply=reply_text)
