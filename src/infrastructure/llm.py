"""LLM client infrastructure using LangChain with resilience features."""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.config.retry import (
    RETRY_CONFIG_DEFAULT,
    RetryConfig,
)
from src.infrastructure.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_openai_llm(
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_retries: int = 5,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Get configured OpenAI LLM client with retry configuration.

    Args:
        model: Model name to use
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_retries: Maximum number of retries on rate limit errors
        max_tokens: Maximum number of tokens to generate (None = unlimited)

    Returns:
        Configured ChatOpenAI instance
    """
    kwargs = dict(
        model=model,
        temperature=temperature,
        api_key=settings.openai_api_key,
        max_retries=max_retries,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def get_anthropic_llm(
    model_name: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs,
) -> ChatAnthropic:
    """Get configured Anthropic LLM client.

    Args:
        model_name: Model name to use
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens to generate (None = unlimited)
        **kwargs: Additional arguments passed to ChatAnthropic

    Returns:
        Configured ChatAnthropic instance
    """
    kwargs["max_tokens"] = max_tokens if max_tokens is not None else 4096
    return ChatAnthropic(
        model_name=model_name,
        temperature=temperature,
        api_key=settings.anthropic_api_key,
        **kwargs,
    )


def get_openrouter_llm(
    model: str = "openai/gpt-5-nano",
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs,
) -> ChatOpenAI:
    """Get configured OpenRouter LLM client using LangChain.

    Args:
        model: Model name to use (default: "openai/gpt-5-nano")
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens to generate (None = unlimited)
        **kwargs: Additional arguments passed to ChatOpenAI

    Returns:
        Configured ChatOpenAI instance for OpenRouter
    """
    if not settings.openrouter_api_key:
        raise ValueError("OpenRouter API key not configured")

    config_kwargs = dict(
        model=model,
        temperature=temperature,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    if max_tokens is not None:
        config_kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**config_kwargs, **kwargs)


def get_ollama_llm(
    model: str = "llama3.2:3b",
    temperature: float = 0.7,
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> ChatOllama:
    """Get configured Ollama LLM client for local models.

    Args:
        model: Model name to use (default: "llama3.2:3b")
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        base_url: Base URL for Ollama server (default: "http://localhost:11434")
        **kwargs: Additional arguments passed to ChatOllama

    Returns:
        Configured ChatOllama instance
    """
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
        **kwargs,
    )


def get_llm(
    provider: str = "openai", **kwargs
) -> ChatOpenAI | ChatAnthropic | ChatOllama:
    """Factory function to get LLM client based on provider.

    Args:
        provider: LLM provider ("openai", "anthropic", "openrouter", or "ollama")
        **kwargs: Additional arguments passed to LLM constructor

    Returns:
        Configured LLM client
    """
    if provider == "anthropic":
        return get_anthropic_llm(**kwargs)
    if provider == "openrouter":
        return get_openrouter_llm(**kwargs)
    if provider == "ollama":
        return get_ollama_llm(**kwargs)
    return get_openai_llm(**kwargs)


class ResilientLLMWrapper:
    """Wrapper for LLM clients adding retry and circuit breaker resilience.

    Provides:
    - Automatic retry with exponential backoff for rate limits
    - Circuit breaker for cascading failure prevention
    - Correlation ID tracking for observability
    """

    def __init__(
        self,
        llm: ChatOpenAI | ChatAnthropic | ChatOllama,
        retry_config: RetryConfig | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
        correlation_id: str | None = None,
    ):
        """Initialize resilient LLM wrapper.

        Args:
            llm: Base LLM client to wrap
            retry_config: Retry configuration
            circuit_config: Circuit breaker configuration
            correlation_id: Optional correlation ID for logging
        """
        self._llm = llm
        self._retry_config = retry_config or RETRY_CONFIG_DEFAULT
        self._correlation_id = correlation_id

        # Get or create circuit breaker for this LLM
        llm_name = getattr(llm, "model", "unknown")
        self._circuit = CircuitBreakerRegistry.get_or_create(
            name=f"llm_{llm_name}",
            config=circuit_config
            or CircuitBreakerConfig(
                failure_threshold=5,
                cooldown_seconds=30.0,
                timeout_seconds=60.0,
            ),
        )

    @property
    def llm(self) -> ChatOpenAI | ChatAnthropic | ChatOllama:
        """Access the underlying LLM client."""
        return self._llm

    def _get_retry_decorator(self) -> Callable:
        """Create retry decorator with configuration."""
        return retry(
            stop=stop_after_attempt(self._retry_config.max_attempts),
            wait=wait_exponential(
                multiplier=self._retry_config.exponential_base,
                min=self._retry_config.base_delay,
                max=self._retry_config.max_delay,
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO),
            reraise=True,
        )

    async def ainvoke(
        self,
        messages: Any,
        correlation_id: str | None = None,
    ) -> Any:
        """Invoke LLM with retry and circuit breaker protection.

        Args:
            messages: Messages to send to LLM
            correlation_id: Optional correlation ID for tracing

        Returns:
            LLM response

        Raises:
            CircuitOpenError: If circuit breaker is open
            Exception: After all retries exhausted
        """
        cid = correlation_id or self._correlation_id

        async def _do_invoke() -> Any:
            return await self._llm.ainvoke(messages)

        # Check circuit breaker
        if not self._circuit.allow_request():
            raise CircuitOpenError(
                f"Circuit breaker is open for LLM calls. "
                f"Last failure: {self._circuit.stats.last_failure_time}"
            )

        # Apply retry decorator
        retry_decorator = self._get_retry_decorator()

        try:
            result = await retry_decorator(_do_invoke)()
            logger.info(
                f"LLM invocation successful (correlation_id={cid})",
                extra={"correlation_id": cid},
            )
            return result
        except Exception as e:
            logger.error(
                f"LLM invocation failed after retries (correlation_id={cid}): {e}",
                extra={"correlation_id": cid},
            )
            raise

    async def invoke(
        self,
        messages: Any,
        correlation_id: str | None = None,
    ) -> Any:
        """Synchronous invoke with retry and circuit breaker protection.

        Args:
            messages: Messages to send to LLM
            correlation_id: Optional correlation ID for tracing

        Returns:
            LLM response
        """
        # Use asyncio to run sync invoke in thread pool

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_invoke_with_retry(messages, correlation_id),
        )

    def _sync_invoke_with_retry(
        self,
        messages: Any,
        correlation_id: str | None = None,
    ) -> Any:
        """Synchronous invoke with retry logic."""
        cid = correlation_id or self._correlation_id

        if not self._circuit.allow_request():
            raise CircuitOpenError(
                f"Circuit breaker is open for LLM calls. "
                f"Last failure: {self._circuit.stats.last_failure_time}"
            )

        @self._get_retry_decorator()
        def _do_invoke() -> Any:
            return self._llm.invoke(messages)

        try:
            result = _do_invoke()
            logger.info(
                f"LLM sync invocation successful (correlation_id={cid})",
                extra={"correlation_id": cid},
            )
            return result
        except Exception as e:
            logger.error(
                f"LLM sync invocation failed (correlation_id={cid}): {e}",
                extra={"correlation_id": cid},
            )
            raise


def get_resilient_llm(
    provider: str = "openai",
    retry_config: RetryConfig | None = None,
    circuit_config: CircuitBreakerConfig | None = None,
    **kwargs,
) -> ResilientLLMWrapper:
    """Factory function to get resilient LLM client.

    Args:
        provider: LLM provider ("openai", "anthropic", "openrouter", or "ollama")
        retry_config: Custom retry configuration
        circuit_config: Custom circuit breaker configuration
        **kwargs: Additional arguments passed to LLM constructor

    Returns:
        ResilientLLMWrapper instance
    """
    llm = get_llm(provider, **kwargs)
    return ResilientLLMWrapper(
        llm=llm,
        retry_config=retry_config,
        circuit_config=circuit_config,
    )
