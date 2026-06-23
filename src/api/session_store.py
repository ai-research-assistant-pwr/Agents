"""Session persistence layer backed by SQLAlchemy.

Works with any SQLAlchemy-compatible database (SQLite for dev/tests,
PostgreSQL for production).  Set DATABASE_URL before the first import of
this module (or of api.db) to pick the backend.
"""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy.orm import Session

from api.db import DBSession, SessionLocal, init_db
from api.models import (
    ExplorationStats,
    HypothesisOut,
    KnowledgeGraph,
    Message,
    ReasoningStep,
    SelectRequest,
    SessionOut,
)

# Create tables on first import.
init_db()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@contextmanager
def _db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _row_to_out(row: DBSession) -> SessionOut:
    exploration = ExplorationStats(**row.exploration)
    hypotheses = [HypothesisOut(**h) for h in row.hypotheses]
    messages = [Message(**m) for m in (row.messages or [])]
    reasoning_trace = [ReasoningStep(**s) for s in (row.reasoning_trace or [])]
    kg = KnowledgeGraph(**row.knowledge_graph) if row.knowledge_graph else None

    return SessionOut(
        sessionId=row.id,
        question=row.question,
        retrieverModelName=row.retriever_model_name or row.model_name or "",
        generatorModelName=row.generator_model_name or row.model_name or "",
        createdAt=row.created_at,
        exploration=exploration,
        hypotheses=hypotheses,
        personaId=getattr(row, "persona_id", None),
        selectedId=row.selected_id,
        rationale=row.rationale,
        messages=messages,
        reasoningTrace=reasoning_trace,
        knowledgeGraph=kg,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_session(session: SessionOut, user_id: str = "anonymous") -> None:
    with _db() as db:
        row = DBSession(
            id=session.sessionId,
            user_id=user_id,
            question=session.question,
            retriever_model_name=session.retrieverModelName,
            generator_model_name=session.generatorModelName,
            persona_id=session.personaId,
            created_at=session.createdAt,
            exploration=session.exploration.model_dump(),
            hypotheses=[h.model_dump() for h in session.hypotheses],
            messages=[],
            reasoning_trace=[s.model_dump() for s in session.reasoningTrace],
            knowledge_graph=session.knowledgeGraph.model_dump()
            if session.knowledgeGraph
            else None,
        )
        db.add(row)


def get_session(session_id: str) -> SessionOut | None:
    with _db() as db:
        row = db.get(DBSession, session_id)
        if row is None:
            return None
        return _row_to_out(row)


def record_selection(session_id: str, selection: SelectRequest) -> bool:
    with _db() as db:
        row = db.get(DBSession, session_id)
        if row is None:
            return False
        row.selected_id = selection.selectedId
        row.rationale = selection.rationale
        return True


def add_message(session_id: str, message: Message) -> bool:
    with _db() as db:
        row = db.get(DBSession, session_id)
        if row is None:
            return False
        msgs = list(row.messages or [])
        msgs.append(message.model_dump())
        row.messages = msgs
        return True


def list_all(user_id: str | None = None) -> list[SessionOut]:
    with _db() as db:
        query = db.query(DBSession)
        if user_id:
            query = query.filter(DBSession.user_id == user_id)
        rows = query.order_by(DBSession.created_at.desc()).all()
        return [_row_to_out(r) for r in rows]


def delete_session(session_id: str) -> bool:
    with _db() as db:
        row = db.get(DBSession, session_id)
        if row is None:
            return False
        db.delete(row)
        return True


def clear() -> None:
    """Remove all sessions — only for tests."""
    with _db() as db:
        db.query(DBSession).delete()
