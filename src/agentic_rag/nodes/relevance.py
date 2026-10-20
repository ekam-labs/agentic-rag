from typing import Any

from pydantic import BaseModel, Field

from ..state import AgenticRAGState, append_execution_event


class RelevanceEvaluation(BaseModel):
    """Structured evaluation containing the indices of relevant documents."""

    relevant_document_indices: list[int] = Field(
        description=(
            "Zero-based indices of retrieved documents that are relevant "
            "to answering the user's query."
        )
    )

    reason: str = Field(
        description="Brief evidence-based reason for the relevance decision."
    )


def is_relevant(
    state: AgenticRAGState,
    llm: Any,
) -> AgenticRAGState:
    """Evaluate retrieved documents and retain only relevant evidence."""

    query = state.get("current_query")
    documents = state.get("retrieved_documents", [])

    if not query:
        raise ValueError("current_query is required for relevance evaluation.")

    if not documents:
        return {
            **state,
            "relevant_documents": [],
            "logs": append_execution_event(
                state,
                "relevance",
                "rejected",
                {"relevant_count": 0, "reason": "No documents were retrieved."},
            ),
        }

    structured_llm = llm.with_structured_output(RelevanceEvaluation)

    context = "\n\n".join(
        f"[Document {index}]\n{document.page_content}"
        for index, document in enumerate(documents)
    )

    prompt = f"""
Determine which retrieved documents are relevant to answering the user's query.

User query:
{query}

Retrieved documents:
{context}

Select only documents that contain information directly relevant to the
user's query.

A document should not be considered relevant merely because it shares
general terminology with the query.

Return the zero-based indices of all relevant documents.
If none of the documents contain information relevant to the query,
return an empty list.
Also provide a brief evidence-based reason for the decision.
"""

    evaluation = structured_llm.invoke(prompt)

    valid_indices = list(
        dict.fromkeys(
            index
            for index in evaluation.relevant_document_indices
            if 0 <= index < len(documents)
        )
    )

    relevant_documents = [documents[index] for index in valid_indices]
    outcome = "accepted" if relevant_documents else "rejected"

    return {
        **state,
        "relevant_documents": relevant_documents,
        "logs": append_execution_event(
            state,
            "relevance",
            outcome,
            {
                "relevant_count": len(relevant_documents),
                "relevant_indices": valid_indices,
                "reason": evaluation.reason,
            },
        ),
    }
