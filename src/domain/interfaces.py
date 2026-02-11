"""Agent interfaces and contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from src.domain.events import DomainEvent


@dataclass
class AgentContext:
    """Context passed to agents during execution."""

    correlation_id: str
    request_id: str
    created_at: datetime
    metadata: dict[str, Any]

    @classmethod
    def create(cls, correlation_id: str | None = None) -> "AgentContext":
        """Factory method to create agent context."""
        return cls(
            correlation_id=correlation_id or "",
            request_id="",
            created_at=datetime.now(timezone.utc),
            metadata={},
        )


AgentResult = TypeVar("AgentResult")


class Agent(ABC, Generic[AgentResult]):
    """Base interface for all agents in the system.

    Agents must implement the `execute` method and should
    communicate only through explicit inputs and outputs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for the agent."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of the agent's purpose."""
        ...

    @abstractmethod
    async def execute(
        self,
        input: Any,
        context: AgentContext,
    ) -> AgentResult:
        """Execute the agent's core logic.

        Args:
            input: The input data for this agent
            context: Shared context including correlation ID

        Returns:
            The agent's output result
        """
        ...

    @abstractmethod
    async def validate_input(self, input: Any) -> bool:
        """Validate the input before processing.

        Args:
            input: The input to validate

        Returns:
            True if input is valid, False otherwise
        """
        ...


class ResearchAgent(Agent[DomainEvent]):
    """Interface for the Researcher Agent."""

    @abstractmethod
    async def research(
        self,
        topic: str,
        context: AgentContext,
    ) -> DomainEvent:
        """Conduct research on a topic."""
        ...


class FactCheckAgent(Agent[DomainEvent]):
    """Interface for the Fact-Checker Agent."""

    @abstractmethod
    async def verify_claims(
        self,
        claims: list[str],
        sources: list[dict[str, str]],
        context: AgentContext,
    ) -> DomainEvent:
        """Verify claims against sources."""
        ...


class SynthesizerAgent(Agent[DomainEvent]):
    """Interface for the Synthesizer Agent."""

    @abstractmethod
    async def synthesize(
        self,
        research: DomainEvent,
        fact_check: DomainEvent,
        context: AgentContext,
    ) -> DomainEvent:
        """Merge research and fact-check into coherent insights."""
        ...


class WriterAgent(Agent[DomainEvent]):
    """Interface for the Writer Agent."""

    @abstractmethod
    async def write_report(
        self,
        synthesis: DomainEvent,
        format: str,
        context: AgentContext,
    ) -> DomainEvent:
        """Write a structured report from synthesis."""
        ...


class CriticAgent(Agent[DomainEvent]):
    """Interface for the Critic Agent."""

    @abstractmethod
    async def review(
        self,
        report: DomainEvent,
        context: AgentContext,
    ) -> DomainEvent:
        """Review the report and suggest improvements."""
        ...


class AgentRegistry:
    """Registry for available agents."""

    _agents: dict[str, Agent] = {}

    @classmethod
    def register(cls, agent: Agent) -> None:
        """Register an agent."""
        cls._agents[agent.name] = agent

    @classmethod
    def get(cls, name: str) -> Agent | None:
        """Get an agent by name."""
        return cls._agents.get(name)

    @classmethod
    def list_agents(cls) -> list[str]:
        """List all registered agent names."""
        return list(cls._agents.keys())
