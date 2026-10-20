from typing import Any

from pydantic import BaseModel, Field

from ..state import AgenticRAGState, append_execution_event


class SupportEvaluation(BaseModel):
    """Structured evaluation of whether the answer is supported by the evidence."""

    supported: bool = Field(
        description=(
            "Whether the generated answer is fully supported by the "
            "provided relevant documents."
        )
    )

    reason: str = Field(
        description="Brief evidence-based reason for the support decision."
    )


def is_supported(
    state: AgenticRAGState,
    llm: Any,
) -> AgenticRAGState:
    """Check whether the generated answer is supported by the evidence."""

    query = state.get("current_query")
    answer = state.get("answer")
    documents = state.get("relevant_documents", [])

    if not query:
        raise ValueError("current_query is required for support evaluation.")
    if not answer:
        raise ValueError("answer is required for support evaluation.")

    support_attempt = state.get("support_attempt", 0) + 1

    if not documents:
        return {
            **state,
            "supported": False,
            "support_attempt": support_attempt,
            "logs": append_execution_event(
                state,
                "support",
                "rejected",
                {
                    "attempt": support_attempt,
                    "reason": "No relevant documents were available.",
                },
            ),
        }

    context = "\n\n".join(
        f"[Document {index}]\n{document.page_content}"
        for index, document in enumerate(documents)
    )

    structured_llm = llm.with_structured_output(SupportEvaluation)

    prompt = f"""
Determine whether the generated answer is fully supported by the provided evidence.

User query:
{query}

Evidence:
{context}

Generated answer:
{answer}

Evaluate the answer strictly against the evidence.

Return supported=True only when the claims made in the answer are supported
by the provided evidence.

Return supported=False if the answer contains unsupported, invented, or
hallucinated information.

Do not use outside knowledge to justify the answer.
Also provide a brief evidence-based reason for the decision.
"""

    evaluation = structured_llm.invoke(prompt)

    return {
        **state,
        "supported": evaluation.supported,
        "support_attempt": support_attempt,
        "logs": append_execution_event(
            state,
            "support",
            "accepted" if evaluation.supported else "rejected",
            {
                "attempt": support_attempt,
                "reason": evaluation.reason,
            },
        ),
    }
