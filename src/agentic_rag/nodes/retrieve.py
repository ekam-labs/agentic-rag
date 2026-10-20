from rag_search import RAGSearch

from ..state import AgenticRAGState, append_execution_event


def retrieve(
    state: AgenticRAGState,
    rag_search: RAGSearch,
) -> AgenticRAGState:
    """Retrieve documents for the current query."""

    query = state.get("current_query")

    if not query:
        raise ValueError("current_query is required for retrieval.")

    documents = rag_search.search(query)
    retrieval_attempt = state.get("retrieval_attempt", 0) + 1

    return {
        **state,
        "retrieved_documents": documents,
        "relevant_documents": [],
        "retrieval_attempt": retrieval_attempt,
        "logs": append_execution_event(
            state,
            "retrieve",
            "completed",
            {
                "attempt": retrieval_attempt,
                "document_count": len(documents),
            },
        ),
    }
