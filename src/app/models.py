from dataclasses import dataclass, field


@dataclass
class ExplorerResult:
    """Result returned by an explorer.

    Rather than pre-formatted text, the explorer now surfaces the raw chunk
    identifiers and their associated data so that downstream components (e.g.
    the retriever) can decide how to format or filter the material.
    """

    chunk_ids: list[str]
    chunks: dict[str, dict] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrieverResult:
    """Result returned by a retriever."""

    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class FeedbackResult:
    """Result returned by a generator's provide_feedback step.

    Attributes:
        skip_feedback: When True the retriever refinement step is skipped and
            the pipeline proceeds directly to hypothesis generation.
        content: The feedback text for the retriever (may be empty when
            skip_feedback is True).
        metadata: Arbitrary key/value metadata from the feedback call.
    """

    skip_feedback: bool
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class GeneratorResult:
    """Result returned by a generator."""

    hypotheses: list[str]
    metadata: dict = field(default_factory=dict)
