from typing import Any, TypedDict

class ExecutionEvent(TypedDict):
    """Structured record describing one stage of a workflow execution."""

    stage: str
    outcome: str
    details: dict[str, Any]


def append_execution_event(
    state: "AgenticRAGState",
    stage: str,
    outcome: str,
    details: dict[str, Any] | None = None,
) -> list[ExecutionEvent]:
    """Return the existing execution log with one new event appended."""

    event: ExecutionEvent = {
        "stage": stage,
        "outcome": outcome,
        "details": details or {},
    }

    return [*state.get("logs", []), event]


class AgenticRAGState(TypedDict, total=False):
    """State carried between nodes during an Agentic RAG execution."""

    original_query: str
    current_query: str

    retrieved_documents: list[Any]
    relevant_documents: list[Any]

    answer: str
    revised_answer: str

    supported: bool
    useful: bool

    support_attempt: int
    retrieval_attempt: int

    logs: list[ExecutionEvent]
