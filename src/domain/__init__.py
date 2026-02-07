"""Domain layer for Veritas."""

from .events import (
    DomainEvent,
    FactCheckCompleted,
    ReportReviewed,
    ReportWritten,
    ResearchCompleted,
    SynthesisCompleted,
)
from .interfaces import (
    Agent,
    AgentContext,
    AgentRegistry,
    AgentResult,
    CriticAgent,
    FactCheckAgent,
    ResearchAgent,
    SynthesizerAgent,
    WriterAgent,
)

__all__ = [
    "DomainEvent",
    "FactCheckCompleted",
    "ReportReviewed",
    "ReportWritten",
    "ResearchCompleted",
    "SynthesisCompleted",
    "Agent",
    "AgentContext",
    "AgentRegistry",
    "AgentResult",
    "CriticAgent",
    "FactCheckAgent",
    "ResearchAgent",
    "SynthesizerAgent",
    "WriterAgent",
]
