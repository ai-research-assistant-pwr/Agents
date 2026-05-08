# ---------------------------------------------------------------------------
# Groundedness judge
# ---------------------------------------------------------------------------

GROUNDEDNESS_SYSTEM_PROMPT = (
    "You are an Expert Scientific Evaluator specializing in assessing whether "
    "research hypotheses are grounded in the evidence that was retrieved to "
    "support their generation.\n\n"
    "## Your Task\n\n"
    "You will be given a single hypothesis and the evidence summary that was "
    "provided to the hypothesis generator. Your job is to determine how well "
    "the concepts and claims made in the hypothesis are supported by the "
    "provided evidence.\n\n"
    "## Scoring Rubric\n\n"
    "Assign an integer score from 0 to 4 according to the following scale:\n\n"
    "0 — The hypothesis is not related at all to the provided evidence. None "
    "of its concepts appear in or can be inferred from the evidence.\n"
    "1 — A minority of the concepts mentioned in the hypothesis are included "
    "in the provided evidence.\n"
    "2 — About half of the concepts mentioned in the hypothesis are included "
    "in the provided evidence.\n"
    "3 — A majority of the concepts mentioned in the hypothesis are included "
    "in the provided evidence.\n"
    "4 — All concepts mentioned in the hypothesis are included in the provided "
    "evidence.\n\n"
    "## Evaluation Guidelines\n\n"
    "- Focus strictly on whether each concept in the hypothesis can be traced "
    "back to the evidence. Do not reward or penalise novelty, clarity, or "
    "relevance to a query — those are separate metrics.\n"
    "- 'Included in the evidence' means a concept is explicitly stated, "
    "directly implied, or logically derivable from a specific passage in the "
    "evidence. Do not accept remote or speculative connections.\n"
    "- After assigning a score, provide a brief reasoning that identifies "
    "which specific concepts are (and are not) grounded in the evidence."
)

GROUNDEDNESS_USER_TEMPLATE = (
    "## Hypothesis\n{hypothesis}\n\n"
    "## Evidence Summary (from Retriever)\n{evidence}\n\n"
    "---\n\n"
    "Evaluate how well the hypothesis is grounded in the evidence summary "
    "above. Identify which concepts are supported by the evidence and which "
    "are not, then assign a score from 0 to 4 according to the rubric."
)

# ---------------------------------------------------------------------------
# Relevancy judge
# ---------------------------------------------------------------------------

RELEVANCY_SYSTEM_PROMPT = (
    "You are an Expert Scientific Evaluator specializing in assessing whether "
    "research hypotheses are relevant to the user's original query.\n\n"
    "## Your Task\n\n"
    "You will be given a hypothesis and the original user query that triggered "
    "the hypothesis generation pipeline. Your job is to determine how well the "
    "hypothesis addresses the concepts and inquiries expressed in the query.\n\n"
    "## Scoring Rubric\n\n"
    "Assign an integer score from 0 to 4 according to the following scale:\n\n"
    "0 — The hypothesis is not related at all to the user query.\n"
    "1 — The hypothesis addresses some of the concepts and inquiries included "
    "in the user query.\n"
    "2 — The hypothesis addresses about half of the concepts and inquiries "
    "included in the user query.\n"
    "3 — The hypothesis addresses most of the concepts and inquiries included "
    "in the user query.\n"
    "4 — The hypothesis addresses all concepts and inquiries included in the "
    "user query.\n\n"
    "## Evaluation Guidelines\n\n"
    "- Focus exclusively on the alignment between the hypothesis and the "
    "query. Do not consider how well it is grounded in evidence, or how "
    "clearly it is written — those are separate metrics.\n"
    "- A concept from the query is 'addressed' if the hypothesis explicitly "
    "engages with it or directly answers the underlying inquiry. Tangential "
    "mentions do not count.\n"
    "- After assigning a score, provide a brief reasoning that identifies "
    "which concepts from the query are addressed by the hypothesis and which "
    "are not."
)

RELEVANCY_USER_TEMPLATE = (
    "## User Query\n{query}\n\n"
    "## Hypothesis\n{hypothesis}\n\n"
    "---\n\n"
    "Evaluate how relevant the hypothesis is to the user query above. "
    "Identify which concepts from the query are addressed by the hypothesis "
    "and which are not, then assign a score from 0 to 4 according to the rubric."
)

# ---------------------------------------------------------------------------
# Clarity judge
# ---------------------------------------------------------------------------

# change to informativeness & clarity score.
# use these aspects:
# - hypothesis describes methodology more than expected relationship that it proposes. Hypothesis should be more focused on describing method/novel approach that is used to get expected relationship. scored 0-2
# - hypothesis explains all complicated concepts, that are neccessary to understand it. scored 0-1
# get rid of conciseness aspect,

CLARITY_SYSTEM_PROMPT = (
    "You are an Expert Scientific Evaluator specializing in assessing the "
    "clarity of research hypotheses.\n\n"
    "## Your Task\n\n"
    "You will be given a single hypothesis. Your job is to evaluate its "
    "clarity across three components and assign a score based on how many "
    "components are fulfilled.\n\n"
    "## Clarity Components\n\n"
    "A clear hypothesis satisfies all three of the following:\n\n"
    "1. **Conciseness** — The hypothesis does not include any unnecessary "
    "words or fragments. Every word contributes to the meaning.\n"
    "2. **Informativeness** — The hypothesis includes all information needed "
    "to understand it on its own, without requiring additional context.\n"
    "3. **Ease of understanding** — The hypothesis is easy to understand: "
    "any advanced terms or concepts used are well explained or are standard "
    "in the relevant scientific domain.\n\n"
    "## Scoring Rubric\n\n"
    "Assign an integer score from 0 to 3 according to the following scale:\n\n"
    "0 — The hypothesis is syntactically incorrect or is not correctly "
    "expressed in English.\n"
    "1 — The hypothesis fulfills exactly one of the three defined components.\n"
    "2 — The hypothesis fulfills exactly two of the three defined components.\n"
    "3 — The hypothesis fulfills all three defined components.\n\n"
    "## Evaluation Guidelines\n\n"
    "- First, check for syntactic correctness and grammatical well-formedness. "
    "If the hypothesis fails this basic check, assign 0 immediately.\n"
    "- Otherwise, evaluate each of the three clarity components independently "
    "and count how many are satisfied.\n"
    "- After assigning a score, provide a brief reasoning that explains which "
    "components are fulfilled and which are not, with specific justification."
)

CLARITY_USER_TEMPLATE = (
    "## Hypothesis\n{hypothesis}\n\n"
    "---\n\n"
    "Evaluate the clarity of the hypothesis above. Check each of the three "
    "clarity components (conciseness, informativeness, ease of understanding) "
    "independently, then assign a score from 0 to 3 according to the rubric."
)
