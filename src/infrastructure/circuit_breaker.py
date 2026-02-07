"""Circuit breaker pattern implementation for preventing cascading failures."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject all requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in half-open state to close
        cooldown_seconds: Seconds to wait before attempting half-open
        timeout_seconds: Max time for a single call before failure
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    cooldown_seconds: float = 30.0
    timeout_seconds: float = 30.0


class CircuitBreaker:
    """Circuit breaker implementation for graceful degradation.

    Prevents cascading failures by:
    - Tracking failures over a window
    - Opening circuit after threshold reached
    - Allowing periodic probe attempts (half-open)
    - Auto-closing circuit when health is restored
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: (
            Callable[[str, CircuitState, CircuitState], None] | None
        ) = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker
            config: Circuit breaker configuration
            on_state_change: Callback when state changes (name, old_state, new_state)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._opened_at: datetime | None = None

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        if self._state == CircuitState.OPEN:
            # Check if cooldown has passed
            if (
                self._opened_at
                and self._opened_at + timedelta(seconds=self.config.cooldown_seconds)
                < datetime.utcnow()
            ):
                self._transition_to(CircuitState.HALF_OPEN)
                return CircuitState.HALF_OPEN
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            logger.warning(
                f"Circuit '{self.name}' state changed: {old_state.value} -> {new_state.value}"
            )
            if self.on_state_change:
                self.on_state_change(self.name, old_state, new_state)

    def record_success(self) -> None:
        """Record a successful call."""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.last_success_time = datetime.utcnow()

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._failure_count = 0
                self._success_count = 0
                self._transition_to(CircuitState.CLOSED)

    def record_failure(self) -> None:
        """Record a failed call."""
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.last_failure_time = datetime.utcnow()
        self._last_failure_time = datetime.utcnow()

        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
            self._opened_at = datetime.utcnow()
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._opened_at = datetime.utcnow()
                self._transition_to(CircuitState.OPEN)

    def allow_request(self) -> bool:
        """Check if a request should be allowed.

        Returns:
            True if request is allowed, False if circuit is open
        """
        return self.state != CircuitState.OPEN

    async def call(
        self,
        coro: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a coroutine with circuit breaker protection.

        Args:
            coro: Coroutine function to execute
            *args: Positional arguments for coro
            **kwargs: Keyword arguments for coro

        Returns:
            Result of the coroutine

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Any exception from the coroutine
        """
        if not self.allow_request():
            raise CircuitOpenError(
                f"Circuit '{self.name}' is open. Requests blocked until cooldown completes."
            )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                coro(*args, **kwargs),
                timeout=self.config.timeout_seconds,
            )
            self.record_success()
            return result
        except asyncio.TimeoutError:
            self.record_failure()
            raise CircuitTimeoutError(
                f"Circuit '{self.name}' call timed out after {self.config.timeout_seconds}s"
            )
        except Exception as e:
            self.record_failure()
            raise


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitTimeoutError(Exception):
    """Raised when circuit breaker call times out."""

    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    _breakers: dict[str, CircuitBreaker] = {}

    @classmethod
    def get_or_create(
        cls,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        if name not in cls._breakers:
            cls._breakers[name] = CircuitBreaker(name, config)
        return cls._breakers[name]

    @classmethod
    def get(cls, name: str) -> CircuitBreaker | None:
        """Get a circuit breaker by name."""
        return cls._breakers.get(name)

    @classmethod
    def reset(cls, name: str) -> bool:
        """Reset (close) a circuit breaker by name."""
        breaker = cls._breakers.get(name)
        if breaker:
            breaker._state = CircuitState.CLOSED
            breaker._failure_count = 0
            breaker._success_count = 0
            return True
        return False

    @classmethod
    def all_states(cls) -> dict[str, tuple[CircuitState, CircuitStats]]:
        """Get state and stats for all circuit breakers."""
        return {
            name: (breaker.state, breaker.stats)
            for name, breaker in cls._breakers.items()
        }
