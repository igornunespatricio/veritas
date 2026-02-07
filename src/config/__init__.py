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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
