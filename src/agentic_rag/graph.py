from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from .config import AgenticRAGConfig
from .nodes.generate import generate_from_context
from .nodes.relevance import is_relevant
from .nodes.retrieve import retrieve
from .nodes.revise_answer import revise_answer
from .nodes.rewrite_query import rewrite_query
from .nodes.support import is_supported
from .nodes.usefulness import is_useful
from .routing.conditions import (
    route_relevance,
    route_support,
    route_usefulness,
)
from .state import AgenticRAGState


def build_graph(
    llm: Any,
    rag_search: Any,
    config: AgenticRAGConfig,
):
    """Build and compile the Agentic RAG workflow."""

    workflow = StateGraph(AgenticRAGState)

    workflow.add_node(
        "retrieve",
        partial(retrieve, rag_search=rag_search),
    )

    workflow.add_node(
        "is_relevant",
        partial(is_relevant, llm=llm),
    )

    workflow.add_node(
        "generate",
        partial(generate_from_context, llm=llm),
    )

    workflow.add_node(
        "is_supported",
        partial(is_supported, llm=llm),
    )

    workflow.add_node(
        "revise_answer",
        partial(revise_answer, llm=llm),
    )

    workflow.add_node(
        "is_useful",
        partial(is_useful, llm=llm),
    )

    workflow.add_node(
        "rewrite_query",
        partial(rewrite_query, llm=llm),
    )

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "is_relevant")

    workflow.add_conditional_edges(
        "is_relevant",
        route_relevance,
        {
            "generate": "generate",
            "no_answer_found": END,
        },
    )

    workflow.add_edge("generate", "is_supported")

    workflow.add_conditional_edges(
        "is_supported",
        partial(route_support, here_config=config),
        {
            "usefulness": "is_useful",
            "revise": "revise_answer",
            "no_answer_found": END,
        },
    )

    workflow.add_edge("revise_answer", "is_supported")

    workflow.add_conditional_edges(
        "is_useful",
        partial(route_usefulness, here_config=config),
        {
            "accepted": END,
            "rewrite": "rewrite_query",
            "no_answer_found": END,
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")
   
    return workflow.compile()
