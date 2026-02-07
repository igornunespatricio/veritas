"""Configuration module for Veritas."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Providers
    openai_api_key: str
    anthropic_api_key: str

    # Optional: LangSmith Observability
    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False

    # Environment
    environment: str = "development"

    # Retry Configuration
    retry_max_attempts: int = 5
    retry_max_backoff: float = 60.0
    retry_base_delay: float = 2.0

    # Circuit Breaker Configuration
    circuit_failure_threshold: int = 5
    circuit_cooldown_seconds: float = 30.0
    circuit_timeout_seconds: float = 60.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
