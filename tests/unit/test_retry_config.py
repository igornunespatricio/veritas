"""Unit tests for retry configuration."""

import pytest
from src.config.retry import (
    RetryConfig,
    RETRY_CONFIG_DEFAULT,
    RETRY_CONFIG_AGGRESSIVE,
    is_retryable_error,
)


class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_standard_rate_limit_is_retryable(self):
        """Test that standard rate limit errors are retryable."""
        # Simulate a standard RateLimitError
        error = Exception(
            "RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit exceeded', 'type': 'rate_limit_exceeded'}}"
        )
        assert is_retryable_error(error) is True

    def test_quota_error_is_not_retryable(self):
        """Test that insufficient_quota errors are NOT retryable."""
        error = Exception(
            "RateLimitError: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details.', 'type': 'insufficient_quota', 'code': 'insufficient_quota'}}"
        )
        assert is_retryable_error(error) is False

    def test_billing_error_is_not_retryable(self):
        """Test that billing-related errors are NOT retryable."""
        error = Exception("Billing error: Account has exceeded quota")
        assert is_retryable_error(error) is False

    def test_too_many_requests_is_retryable(self):
        """Test that 'too many requests' errors are retryable."""
        error = Exception("Too many requests. Please slow down.")
        assert is_retryable_error(error) is True

    def test_service_unavailable_is_retryable(self):
        """Test that service unavailable errors are retryable."""
        error = Exception("Service temporarily unavailable")
        assert is_retryable_error(error) is True

    def test_connection_error_is_retryable(self):
        """Test that connection errors are retryable."""
        error = ConnectionError("Connection reset by peer")
        assert is_retryable_error(error) is True

    def test_timeout_error_is_retryable(self):
        """Test that timeout errors are retryable."""
        error = TimeoutError("Request timed out")
        assert is_retryable_error(error) is True

    def test_arbitrary_error_is_not_retryable(self):
        """Test that arbitrary errors without retry keywords are NOT retryable."""
        error = ValueError("Some validation error")
        assert is_retryable_error(error) is False


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_values(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter == 0.1

    def test_custom_values(self):
        """Test custom retry configuration values."""
        config = RetryConfig(
            max_attempts=10,
            base_delay=1.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=0.2,
        )
        assert config.max_attempts == 10
        assert config.base_delay == 1.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter == 0.2

    def test_validation_min_values(self):
        """Test that validation clamps values to minimums."""
        config = RetryConfig(
            max_attempts=0,  # Should be clamped to 1
            base_delay=0.0,  # Should be clamped to 0.1
            max_delay=0.05,  # Should be clamped to base_delay
            jitter=-0.5,  # Should be clamped to 0.0
        )
        assert config.max_attempts == 1
        assert config.base_delay == 0.1
        assert config.max_delay == 0.1  # max_delay should be >= base_delay
        assert config.jitter == 0.0

    def test_jitter_clamping(self):
        """Test that jitter is clamped to valid range."""
        config = RetryConfig(jitter=2.0)  # Should be clamped to 1.0
        assert config.jitter == 1.0

        config = RetryConfig(jitter=-1.0)  # Should be clamped to 0.0
        assert config.jitter == 0.0

    def test_get_delay_progression(self):
        """Test that delay increases exponentially."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=100.0,
            exponential_base=2.0,
            jitter=0.0,  # No jitter for predictable values
        )
        # Expected: 2*2^1=4, 2*2^2=8, 2*2^3=16, 2*2^4=32, 2*2^5=64
        assert config.get_delay(1) == 4.0
        assert config.get_delay(2) == 8.0
        assert config.get_delay(3) == 16.0
        assert config.get_delay(4) == 32.0
        assert config.get_delay(5) == 64.0

    def test_get_delay_caps_at_max(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            max_attempts=10,
            base_delay=2.0,
            max_delay=10.0,  # Very low max
            exponential_base=2.0,
            jitter=0.0,
        )
        # Even though exponential would grow, it should be capped
        delays = [config.get_delay(i) for i in range(1, 11)]
        for delay in delays:
            assert delay <= config.max_delay

    def test_predefined_configs(self):
        """Test predefined retry configurations."""
        # Default config
        assert RETRY_CONFIG_DEFAULT.max_attempts == 5
        assert RETRY_CONFIG_DEFAULT.base_delay == 2.0
        assert RETRY_CONFIG_DEFAULT.max_delay == 60.0

        # Aggressive config (more retries, shorter initial delay)
        assert RETRY_CONFIG_AGGRESSIVE.max_attempts == 8
        assert RETRY_CONFIG_AGGRESSIVE.base_delay == 1.0
        assert RETRY_CONFIG_AGGRESSIVE.max_delay == 120.0

    def test_retryable_exceptions_tuple(self):
        """Test that retryable exceptions is a tuple."""
        config = RetryConfig()
        assert isinstance(config.retryable_exceptions, tuple)
        assert "RateLimitError" in config.retryable_exceptions
        assert "APIError" in config.retryable_exceptions
