from typing import Any

from ..state import AgenticRAGState, append_execution_event


def generate_from_context(
    state: AgenticRAGState,
    llm: Any,
) -> AgenticRAGState:
    """Generate an answer using only the documents selected as relevant."""

    query = state.get("current_query")
    documents = state.get("relevant_documents", [])

    if not query:
        raise ValueError("current_query is required for answer generation.")

    if not documents:
        raise ValueError(
            "relevant_documents must contain at least one document before answer generation."
        )

    context = "\n\n".join(
        f"[Document {index}]\n{document.page_content}"
        for index, document in enumerate(documents)
    )

    prompt = f"""
Answer the user's query using only the supplied context.

User query:
{query}

Context:
{context}

Instructions:
- Use the supplied context as the source of evidence.
- Do not introduce factual claims that are not supported by the context.
- Answer the user's actual question directly.
- If the context does not provide enough information to answer the query,
  state that the available context is insufficient.
"""

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        **state,
        "answer": answer,
        "logs": append_execution_event(
            state,
            "generate",
            "completed",
            {"relevant_document_count": len(documents)},
        ),
    }
