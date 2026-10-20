from typing import Any

from ..state import AgenticRAGState, append_execution_event


def revise_answer(
    state: AgenticRAGState,
    llm: Any,
) -> AgenticRAGState:
    """Revise an unsupported answer using only the available evidence."""

    query = state.get("current_query")
    answer = state.get("answer")
    documents = state.get("relevant_documents", [])
    logs = state.get("logs", [])
    reason = logs[-1]["details"].get("reason") if logs else "No reason provided."
    
    if not query:
        raise ValueError("current_query is required for answer revision.")
    if not answer:
        raise ValueError("answer is required for answer revision.")
    if not documents:
        raise ValueError(
            "relevant_documents must contain at least one document before answer revision."
        )

    context = "\n\n".join(
        f"[Document {index}]\n{document.page_content}"
        for index, document in enumerate(documents)
    )

    prompt = f"""
Revise the answer so that it is fully supported by the provided evidence.

User query:
{query}

Evidence:
{context}

Current answer:
{answer}

Reason for revision:
{reason}

Instructions:
- Keep the answer focused on the user's actual query.
- Remove any claim that is not supported by the evidence.
- Do not introduce information from outside the evidence.
- Preserve supported information when it remains relevant.
- If the evidence is insufficient, clearly state that the available
  information is insufficient rather than inventing an answer.
- Return only the revised answer.
"""

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        **state,
        "revised_answer": answer,
        "logs": append_execution_event(
            state,
            "revise_answer",
            "completed",
            {"document_count": len(documents)},
        ),
    }
