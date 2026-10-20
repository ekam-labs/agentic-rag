# Agentic RAG

A configurable, stateful RAG workflow built with LangGraph for retrieval, answer validation, and iterative refinement.

## Workflow

```mermaid
flowchart TD
    START([__start__]) --> retrieve[retrieve]
    retrieve --> relevance{is_relevant}

    relevance -.->|no_answer_found| END([__end__])
    relevance -.->|generate| generate[generate]

    generate --> support{is_supported}

    support -.->|no_answer_found| END
    support -.->|usefulness| usefulness{is_useful}
    support -.->|revise| revise[revise_answer]

    revise --> support

    usefulness -.->|accepted| END
    usefulness -.->|rewrite| rewrite[rewrite_query]

    rewrite --> retrieve

    style START fill:#ffffff,stroke:#8b5cf6,color:#333333
    style retrieve fill:#f5f0ff,stroke:#8b5cf6,color:#333333
    style relevance fill:#f5f0ff,stroke:#8b5cf6,color:#333333
    style generate fill:#f5f0ff,stroke:#8b5cf6,color:#333333
    style support fill:#f5f0ff,stroke:#8b5cf6,color:#333333
    style usefulness fill:#f5f0ff,stroke:#8b5cf6,color:#333333
    style revise fill:#f5f0ff,stroke:#8b5cf6,color:#333333
    style rewrite fill:#f5f0ff,stroke:#8b5cf6,color:#333333
    style END fill:#c4b5fd,stroke:#8b5cf6,color:#333333
```

## Installation

```bash
pip install agentic-rag
```

## Usage

```python
from agentic_rag import AgenticRAGConfig, build_graph
```

Create a compatible LangChain chat model and provide a retrieval callable, then build the workflow:

```python
config = AgenticRAGConfig(max_attempts=2)

graph = build_graph(
    llm=llm,
    rag_search=rag_search,
    config=config,
)
```

Invoke the compiled graph with the initial state required by the workflow.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.