"""Retry configuration for agent operations."""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def is_retryable_error(exception: Exception) -> bool:
    """Determine if an exception is retryable based on type and message.

    Args:
        exception: The exception to check

    Returns:
        True if the error is retryable, False otherwise
    """
    exception_type = type(exception).__name__
    exception_message = str(exception).lower()

    # Non-retryable quota/billing errors
    non_retryable_keywords = [
        "insufficient_quota",
        "billing",
        "quota",
        "exceeded your current quota",
        "account",
    ]

    for keyword in non_retryable_keywords:
        if keyword in exception_message:
            logger.warning(
                f"Non-retryable error detected ({keyword}): {exception}. "
                "This is a billing/account issue that will not resolve with retries."
            )
            return False

    # Retryable rate limit errors (standard rate limiting)
    retryable_keywords = [
        "rate_limit",
        "rate limit",
        "too many requests",
        "service_unavailable",
        "service unavailable",
        "temporarily unavailable",
        "api_error",
    ]

    for keyword in retryable_keywords:
        if keyword in exception_message:
            return True

    # Check by exception type
    retryable_types = [
        "RateLimitError",
        "APIError",
        "APITimeoutError",
        "ServiceUnavailableError",
        "ConnectionError",
        "TimeoutError",
    ]

    return exception_type in retryable_types


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff.

    Attributes:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential calculation (2 = doubling)
        jitter: Random jitter factor to add variance (0.0-1.0)
        retryable_exceptions: Tuple of exception types that are retriable
    """

    max_attempts: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retryable_exceptions: tuple = (
        # Rate limit errors
        "RateLimitError",
        # Temporary API errors
        "APIError",
        "APITimeoutError",
        "ServiceUnavailableError",
        # Network errors
        "ConnectionError",
        "TimeoutError",
    )

    def __post_init__(self):
        """Validate and normalize configuration."""
        self.max_attempts = max(1, self.max_attempts)
        self.base_delay = max(0.1, self.base_delay)
        self.max_delay = max(self.base_delay, self.max_delay)
        self.jitter = max(0.0, min(1.0, self.jitter))

    def is_retryable(self, exception: Exception) -> bool:
        """Check if an exception should be retried.

        Args:
            exception: The exception to check

        Returns:
            True if the error is retryable
        """
        return is_retryable_error(exception)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a specific attempt number.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Delay in seconds before next retry
        """
        # Exponential backoff: base * (base^attempt)
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        # Add jitter for variance
        if self.jitter > 0:
            import random

            jitter_range = delay * self.jitter
            delay = delay + random.uniform(-jitter_range, jitter_range)

        return max(0, delay)


# Predefined configurations for different scenarios
RETRY_CONFIG_DEFAULT = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=0.1,
)

RETRY_CONFIG_AGGRESSIVE = RetryConfig(
    max_attempts=8,
    base_delay=1.0,
    max_delay=120.0,
    exponential_base=2.0,
    jitter=0.15,
)

RETRY_CONSERVATIVE = RetryConfig(
    max_attempts=3,
    base_delay=5.0,
    max_delay=30.0,
    exponential_base=1.5,
    jitter=0.05,
)
