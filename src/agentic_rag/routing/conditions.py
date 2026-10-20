from ..state import AgenticRAGState
from ..config import AgenticRAGConfig

def route_relevance(state: AgenticRAGState) -> str:
    """Route the workflow based on whether relevant documents were found."""

    if state.get("relevant_documents"):
        return "generate"

    return "no_answer_found"


def route_support(
    state: AgenticRAGState,
    here_config: AgenticRAGConfig,
) -> str:
    """Route the workflow based on answer support and the support retry limit."""

    if state.get("supported", False):
        return "usefulness"

    support_attempt = state.get("support_attempt", 0)

    if support_attempt < here_config.max_attempts:
        return "revise"

    return "no_answer_found"


def route_usefulness(
    state: AgenticRAGState,
    here_config: AgenticRAGConfig,
) -> str:
    """Route the workflow based on answer usefulness and retrieval retry limit."""

    if state.get("useful", False):
        return "accepted"

    retrieval_attempt = state.get("retrieval_attempt", 0)

    if retrieval_attempt < here_config.max_attempts:
        return "rewrite"

    return "no_answer_found"