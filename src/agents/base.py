"""Base agent class with common functionality."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.retry import RetryConfig, RETRY_CONFIG_DEFAULT
from src.domain.interfaces import Agent, AgentContext, AgentResult
from src.infrastructure.llm import (
    get_llm,
    get_resilient_llm,
    ResilientLLMWrapper,
)

logger = logging.getLogger(__name__)


class BaseAgent(Agent[AgentResult]):
    """Base class for all agents with common functionality.

    Provides:
    - Structured logging
    - Retry logic with exponential backoff
    - LLM client access with resilience wrapper
    - Correlation ID tracking
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        llm_temperature: float = 0.7,
        retry_config: RetryConfig | None = None,
    ):
        """Initialize base agent.

        Args:
            name: Agent name identifier
            description: Brief description of agent's purpose
            llm_provider: LLM provider ("openai" or "anthropic")
            llm_model: Model name to use
            llm_temperature: Sampling temperature
            retry_config: Custom retry configuration
        """
        self._name = name
        self._description = description
        self._retry_config = retry_config or RETRY_CONFIG_DEFAULT

        # Create resilient LLM wrapper with retry and circuit breaker
        self._llm: ResilientLLMWrapper = get_resilient_llm(
            provider=llm_provider,
            model=llm_model,
            temperature=llm_temperature,
            retry_config=self._retry_config,
        )
        self._correlation_id: str | None = None

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name

    @property
    def description(self) -> str:
        """Get agent description."""
        return self._description

    @property
    def llm(self) -> ResilientLLMWrapper:
        """Access the configured resilient LLM client."""
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
        self._log(
            logging.INFO,
            f"Executing {self.name} with input type: {type(input).__name__}",
        )

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
        """Execute agent logic with retry logic.

        Note: The ResilientLLMWrapper already has built-in retry,
        but this provides an additional layer of retry at the agent level.
        """
        return await self._run(input, context)

    @abstractmethod
    async def _run(
        self,
        input: Any,
        context: AgentContext,
    ) -> AgentResult:
        """Internal run method - implement agent logic here.

        Subclasses should implement this method with their specific logic.
        The LLM calls should use self.llm.ainvoke() which includes
        automatic retry and circuit breaker protection.
        """
        ...

    async def validate_input(self, input: Any) -> bool:
        """Default validation - override in subclasses for specific logic.

        Args:
            input: The input to validate

        Returns:
            True if input is valid, False otherwise
        """
        return input is not None
