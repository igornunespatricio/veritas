"""Unit tests for circuit breaker implementation."""

import asyncio
import pytest
from src.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitTimeoutError,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig class."""

    def test_default_values(self):
        """Test default circuit breaker configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.cooldown_seconds == 30.0
        assert config.timeout_seconds == 30.0

    def test_custom_values(self):
        """Test custom circuit breaker configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            cooldown_seconds=60.0,
            timeout_seconds=120.0,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.cooldown_seconds == 60.0
        assert config.timeout_seconds == 120.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Test that circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED

    def test_record_success(self):
        """Test recording successful calls."""
        cb = CircuitBreaker("test")
        cb.record_success()
        assert cb.stats.total_calls == 1
        assert cb.stats.successful_calls == 1
        assert cb.stats.failed_calls == 0

    def test_record_failure(self):
        """Test recording failed calls."""
        cb = CircuitBreaker("test")
        cb.record_failure()
        assert cb.stats.total_calls == 1
        assert cb.stats.successful_calls == 0
        assert cb.stats.failed_calls == 1

    def test_opens_after_failure_threshold(self):
        """Test that circuit opens after failure threshold is reached."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config=config)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_allows_request_when_closed(self):
        """Test that requests are allowed when circuit is closed."""
        cb = CircuitBreaker("test")
        assert cb.allow_request() is True

    def test_blocks_request_when_open(self):
        """Test that requests are blocked when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config=config)
        cb.record_failure()  # Opens the circuit

        assert cb.allow_request() is False

    def test_transition_to_half_open_after_cooldown(self):
        """Test that circuit transitions to half-open after cooldown."""
        config = CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.1)
        cb = CircuitBreaker("test", config=config)
        cb.record_failure()  # Opens the circuit

        # Wait for cooldown
        import time

        time.sleep(0.2)

        # Should now be half-open
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_success_threshold_in_half_open(self):
        """Test that circuit closes after success threshold in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            cooldown_seconds=0.1,
        )
        cb = CircuitBreaker("test", config=config)
        cb.record_failure()  # Opens the circuit

        # Wait for cooldown
        import time

        time.sleep(0.2)

        # Should be half-open
        assert cb.state == CircuitState.HALF_OPEN

        # Record successes
        cb.record_success()
        cb.record_success()  # Meets threshold

        # Should be closed
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        """Test that circuit reopens on failure in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=3,
            cooldown_seconds=0.5,  # Longer cooldown for test stability
        )
        cb = CircuitBreaker("test", config=config)
        cb.record_failure()  # Opens the circuit

        # Immediately record failure without waiting for cooldown
        # This simulates failure while in half-open state
        cb._state = CircuitState.HALF_OPEN  # Force half-open state
        cb.record_failure()

        # Should be open again
        assert cb.state == CircuitState.OPEN

    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        cb = CircuitBreaker("test")
        cb.record_success()
        cb.record_failure()
        cb.record_failure()

        assert cb.stats.failure_rate == pytest.approx(2 / 3)

    def test_callback_on_state_change(self):
        """Test that callback is called on state change."""
        callback_calls = []

        def callback(name, old_state, new_state):
            callback_calls.append((name, old_state, new_state))

        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config=config, on_state_change=callback)
        cb.record_failure()

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "test"
        assert callback_calls[0][1] == CircuitState.CLOSED
        assert callback_calls[0][2] == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_call_success(self):
        """Test successful async call through circuit breaker."""
        cb = CircuitBreaker("test")

        async def success_coro():
            return "success"

        result = await cb.call(success_coro)
        assert result == "success"
        assert cb.stats.successful_calls == 1

    @pytest.mark.asyncio
    async def test_call_failure(self):
        """Test that failure is recorded when coroutine raises."""
        cb = CircuitBreaker("test")

        async def fail_coro():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await cb.call(fail_coro)

        assert cb.stats.failed_calls == 1

    @pytest.mark.asyncio
    async def test_call_blocks_when_open(self):
        """Test that calls are blocked when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config=config)
        cb.record_failure()  # Opens the circuit

        async def dummy_coro():
            return "should not execute"

        with pytest.raises(CircuitOpenError):
            await cb.call(dummy_coro)


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry class."""

    def teardown_method(self):
        """Clear registry after each test."""
        CircuitBreakerRegistry._breakers.clear()

    def test_get_or_create(self):
        """Test getting or creating a circuit breaker."""
        cb1 = CircuitBreakerRegistry.get_or_create("test1")
        cb2 = CircuitBreakerRegistry.get_or_create("test1")

        assert cb1 is cb2  # Same instance

    def test_get_existing(self):
        """Test getting an existing circuit breaker."""
        cb = CircuitBreakerRegistry.get_or_create("test")
        retrieved = CircuitBreakerRegistry.get("test")

        assert cb is retrieved

    def test_get_nonexistent(self):
        """Test getting a nonexistent circuit breaker returns None."""
        result = CircuitBreakerRegistry.get("nonexistent")
        assert result is None

    def test_reset(self):
        """Test resetting a circuit breaker."""
        cb = CircuitBreakerRegistry.get_or_create("test")
        cb.record_failure()

        result = CircuitBreakerRegistry.reset("test")

        assert result is True
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_all_states(self):
        """Test getting all circuit breaker states."""
        cb1 = CircuitBreakerRegistry.get_or_create("test1")
        cb2 = CircuitBreakerRegistry.get_or_create("test2")

        states = CircuitBreakerRegistry.all_states()

        assert "test1" in states
        assert "test2" in states
        assert len(states) == 2
