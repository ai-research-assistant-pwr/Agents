from typing import Literal
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ExplorationStats(BaseModel):
    nodesTraversed: int = 0
    relations: int = 0
    sourcePapers: int = 0
    clusters: int = 0
    communities: list[str] = []
    durationSec: float = 0.0


class HypothesisOut(BaseModel):
    id: Literal["A", "B"]
    classification: str
    headline: str
    statement: str
    drawnFrom: list[str]
    falsifiablePrediction: str


class ReasoningStep(BaseModel):
    step: str
    description: str
    durationSec: float
    details: str = ""


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    createdAt: str


class KGNode(BaseModel):
    id: str
    label: str
    type: Literal["query", "community", "concept", "paper"]
    paperId: str | None = None
    abstract: str | None = None
    summary: str | None = None


class KGEdge(BaseModel):
    source: str
    target: str
    relation: str


class KnowledgeGraph(BaseModel):
    nodes: list[KGNode]
    edges: list[KGEdge]


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionOut(BaseModel):
    sessionId: str
    question: str
    createdAt: str
    exploration: ExplorationStats
    hypotheses: list[HypothesisOut]
    # populated after selection
    selectedId: str | None = None
    rationale: str | None = None
    # conversation
    messages: list[Message] = []
    # pipeline trace
    reasoningTrace: list[ReasoningStep] = []
    knowledgeGraph: KnowledgeGraph | None = None


# ---------------------------------------------------------------------------
# Session list
# ---------------------------------------------------------------------------


class SessionListItem(BaseModel):
    id: str
    title: str
    createdAt: str
    status: Literal["awaiting", "developed", "archived"]


class SessionListOut(BaseModel):
    sessions: list[SessionListItem]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class UserOut(BaseModel):
    id: str
    username: str
    email: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class TokenOut(BaseModel):
    token: str
    user: UserOut


# ---------------------------------------------------------------------------
# Requests / responses
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    question: str


class SelectRequest(BaseModel):
    selectedId: Literal["A", "B"]
    rationale: str | None = None


class SelectResponse(BaseModel):
    ok: bool = True


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
