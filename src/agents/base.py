"""Base agent class with common functionality."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.runnables import Runnable
from tenacity import retry, stop_after_attempt, wait_exponential

from src.domain.interfaces import Agent, AgentContext, AgentResult
from src.infrastructure.llm import get_llm

logger = logging.getLogger(__name__)


class BaseAgent(ABC, Agent[AgentResult]):
    """Base class for all agents with common functionality.

    Provides:
    - Structured logging
    - Retry logic with exponential backoff
    - LLM client access
    - Correlation ID tracking
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        llm_temperature: float = 0.7,
    ):
        self._name = name
        self._description = description
        self._llm = get_llm(
            provider=llm_provider,
            model=llm_model,
            temperature=llm_temperature,
        )
        self._correlation_id: str | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def llm(self) -> Runnable:
        """Access the configured LLM client."""
        return self._llm

    def _set_correlation_id(self, context: AgentContext) -> None:
        """Set correlation ID from context for logging."""
        self._correlation_id = context.correlation_id

    def _log(
        self,
        level: int,
        msg: str,
        **kwargs: Any,
    ) -> None:
        """Log with correlation ID context."""
        extra = {"correlation_id": self._correlation_id}
        logger.log(level, msg, extra=extra, **kwargs)

    async def execute(
        self,
        input: Any,
        context: AgentContext,
    ) -> AgentResult:
        """Execute agent logic with retry and error handling."""
        self._set_correlation_id(context)
        self._log(logging.INFO, f"Executing {self.name} with input: {type(input)}")

        # Validate input
        if not await self.validate_input(input):
            self._log(logging.ERROR, f"Invalid input for {self.name}")
            raise ValueError(f"Invalid input for agent {self.name}")

        # Execute with retry
        try:
            result = await self._execute_with_retry(input, context)
            self._log(logging.INFO, f"{self.name} completed successfully")
            return result
        except Exception as e:
            self._log(logging.ERROR, f"{self.name} failed: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _execute_with_retry(
        self,
        input: Any,
        context: AgentContext,
    ) -> AgentResult:
        """Execute agent logic with retry logic."""
        return await self._run(input, context)

    @abstractmethod
    async def _run(
        self,
        input: Any,
        context: AgentContext,
    ) -> AgentResult:
        """Internal run method - implement agent logic here."""
        ...

    async def validate_input(self, input: Any) -> bool:
        """Default validation - override in subclasses for specific logic."""
        return input is not None
