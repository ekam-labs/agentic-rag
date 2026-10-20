from typing import Any

from ..state import AgenticRAGState, append_execution_event


def rewrite_query(
    state: AgenticRAGState,
    llm: Any,
) -> AgenticRAGState:
    """Rewrite the current query to improve the next retrieval attempt."""

    original_query = state.get("original_query")
    current_query = state.get("current_query")
    answer = state.get("answer")

    if not original_query:
        raise ValueError("original_query is required for query rewriting.")
    if not current_query:
        raise ValueError("current_query is required for query rewriting.")
    if not answer:
        raise ValueError("answer is required for query rewriting.")

    prompt = f"""
Rewrite the user's query to improve document retrieval.

Original user query:
{original_query}

Current retrieval query:
{current_query}

Previous generated answer:
{answer}

The previous answer was evaluated as not useful for the user's actual
question.

Create a new retrieval query that preserves the user's original intent
while making the information required to answer the query clearer and
more explicit.

Do not answer the question.
Do not add information that is not implied by the original query.
Return only the rewritten retrieval query.
"""

    response = llm.invoke(prompt)
    rewritten_query = (
        response.content if hasattr(response, "content") else str(response)
    ).strip()

    if not rewritten_query:
        raise ValueError("The LLM returned an empty rewritten query.")

    return {
        **state,
        "current_query": rewritten_query,
        "answer": "",
        "supported": False,
        "useful": False,
        "support_attempt": 0,
        "relevant_documents": [],
        "logs": append_execution_event(
            state,
            "rewrite_query",
            "completed",
            {"new_query": rewritten_query},
        ),
    }
