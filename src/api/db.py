"""SQLAlchemy database layer.

DATABASE_URL controls the backend:
  - Not set / empty -> defaults to sqlite:///./hypothesis_forge.db (dev)
  - sqlite:///:memory: -> in-process SQLite, useful for tests (use StaticPool)
  - postgresql://user:pass@host/db -> production PostgreSQL

The engine and table definitions live here.  Call init_db() once at startup
to create any missing tables.
"""

from uuid import uuid4
from datetime import datetime, timezone

from sqlalchemy import Column, JSON, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from api.config import get_settings

DATABASE_URL: str = get_settings().database_url

_kwargs: dict = {}
_pool_kwargs: dict = {}

if DATABASE_URL.startswith("sqlite"):
    _kwargs["check_same_thread"] = False
    if DATABASE_URL == "sqlite:///:memory:":
        from sqlalchemy.pool import StaticPool  # noqa: PLC0415

        _pool_kwargs["poolclass"] = StaticPool

engine = create_engine(DATABASE_URL, connect_args=_kwargs, **_pool_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class DBUser(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(
        String(50),
        nullable=False,
        default=lambda: datetime.now(timezone.utc).isoformat(),
    )


class DBSession(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(255), nullable=False, default="anonymous")
    question = Column(Text, nullable=False)
    model_name = Column(String(255), nullable=True)
    created_at = Column(String(50), nullable=False)
    exploration = Column(JSON, nullable=False)
    hypotheses = Column(JSON, nullable=False)
    selected_id = Column(String(2), nullable=True)
    rationale = Column(Text, nullable=True)
    messages = Column(JSON, nullable=False, default=list)
    reasoning_trace = Column(JSON, nullable=False, default=list)
    knowledge_graph = Column(JSON, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_model_name_column()


def _ensure_model_name_column() -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("sessions")}
    if "model_name" in columns:
        return
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE sessions ADD COLUMN model_name VARCHAR(255)"))
