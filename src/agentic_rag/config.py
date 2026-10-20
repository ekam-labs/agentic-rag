from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgenticRAGConfig:
    """Configuration for an Agentic RAG workflow execution."""

    max_attempts: int = 2

    def __post_init__(self) -> None:
        """Validate workflow configuration values."""

        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
