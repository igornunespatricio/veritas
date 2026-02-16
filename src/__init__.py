"""Veritas - Autonomous Research & Report Generation Platform."""

from .agents import (
    BaseAgent,
    CriticAgent,
    FactCheckerAgent,
    ResearcherAgent,
    SynthesizerAgent,
    WriterAgent,
)
from .api import app
from .config import settings
from .domain import (
    Agent,
    AgentContext,
    AgentRegistry,
    AgentResult,
    FactCheckAgent,
    ResearchAgent,
)
from .domain import (
    CriticAgent as DomainCriticAgent,
)
from .domain import (
    SynthesizerAgent as DomainSynthesizerAgent,
)
from .domain import (
    WriterAgent as DomainWriterAgent,
)
from .orchestration import ResearchWorkflow, WorkflowResult, WorkflowStage

__version__ = "0.1.0"

__all__ = [
    # Agents
    "BaseAgent",
    "ResearcherAgent",
    "FactCheckerAgent",
    "SynthesizerAgent",
    "WriterAgent",
    "CriticAgent",
    # Domain
    "Agent",
    "AgentContext",
    "AgentRegistry",
    "AgentResult",
    "ResearchAgent",
    "FactCheckAgent",
    "SynthesizerAgent",
    "DomainSynthesizerAgent",
    "DomainWriterAgent",
    "DomainCriticAgent",
    # Orchestration
    "ResearchWorkflow",
    "WorkflowResult",
    "WorkflowStage",
    # Config
    "settings",
    # API
    "app",
]
