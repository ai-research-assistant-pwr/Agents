from pydantic import BaseModel


class HypothesesResponse(BaseModel):
    """Structured output schema for the hypothesis generator."""

    hypotheses: list[str]


GENERATOR_SYSTEM_PROMPT = (
    "You are a Research Scientist specializing in hypothesis generation. "
    "Given a research prompt and supporting evidence, generate novel, testable "
    "scientific hypotheses. Each hypothesis should be specific, falsifiable, and "
    "grounded in the provided evidence."
)

GENERATE_USER_TEMPLATE = (
    "Research prompt: {prompt}\n\n"
    "Supporting evidence:\n{retriever_output}\n\n"
    "Generate a list of scientific hypotheses based on the above."
)

FEEDBACK_SYSTEM_PROMPT = (
    "You are a Research Scientist reviewing information provided by a retrieval "
    "system. Analyze the retrieved context and identify gaps, missing details, or "
    "areas that need deeper exploration to generate strong hypotheses. Provide "
    "clear, actionable feedback on what additional information is needed."
)

FEEDBACK_USER_TEMPLATE = (
    "Research prompt: {prompt}\n\n"
    "Retrieved context:\n{retriever_output}\n\n"
    "What additional information or clarification is needed to generate strong "
    "scientific hypotheses? Be specific about what is missing or insufficient."
)
