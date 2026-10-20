from typing import Any

from pydantic import BaseModel, Field

from ..state import AgenticRAGState, append_execution_event


class UsefulnessEvaluation(BaseModel):
    """Structured evaluation of whether the answer addresses the user query."""

    useful: bool = Field(
        description=(
            "Whether the generated answer directly and sufficiently answers "
            "the user's query."
        )
    )

    reason: str = Field(
        description="Brief reason for the usefulness decision."
    )


def is_useful(
    state: AgenticRAGState,
    llm: Any,
) -> AgenticRAGState:
    """Check whether the supported answer is useful for the user's query."""

    query = state.get("current_query")
    answer = state.get("answer")

    if not query:
        raise ValueError("current_query is required for usefulness evaluation.")
    if not answer:
        raise ValueError("answer is required for usefulness evaluation.")

    structured_llm = llm.with_structured_output(UsefulnessEvaluation)

    prompt = f"""
Determine whether the generated answer actually answers the user's query.

User query:
{query}

Generated answer:
{answer}

Evaluate only whether the answer is useful for the specific query.

Return useful=True when:
- The answer directly addresses what the user asked.
- The answer is relevant to the query.
- The answer provides sufficient information to satisfy the query based on
  the available evidence.

Return useful=False when:
- The answer is about a different topic.
- The answer only partially addresses the actual question.
- The answer avoids or fails to answer the question.
- The answer is technically supported by the documents but does not answer
  what the user actually asked.

Do not use outside knowledge to judge factual correctness.
Also provide a brief reason for the decision.
"""

    evaluation = structured_llm.invoke(prompt)

    return {
        **state,
        "useful": evaluation.useful,
        "logs": append_execution_event(
            state,
            "usefulness",
            "accepted" if evaluation.useful else "rejected",
            {"reason": evaluation.reason},
        ),
    }
