from dataclasses import dataclass, field


@dataclass
class ExplorerResult:
    """Result returned by an explorer."""

    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrieverResult:
    """Result returned by a retriever."""

    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class GeneratorResult:
    """Result returned by a generator."""

    hypotheses: list[str]
    metadata: dict = field(default_factory=dict)
    reasoning: str = ""  # inner monologue from reasoning models, covers all hypotheses
