"""Pydantic request models for the API."""

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Request model for submitting a research job."""

    topic: str = Field(..., description="Research topic or question")
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum review iterations (1-10)",
    )
    auto_approve_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Score threshold for auto-approval (0.0-1.0)",
    )
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: openai, anthropic, ollama, openrouter",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="Model name (e.g., gpt-4o, claude-sonnet-4-20250514, llama3.2:3b)",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=100,
        le=128000,
        description="Maximum tokens per LLM call (100-128000)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "topic": "What are the environmental impacts of electric vehicles?",
                    "max_iterations": 3,
                    "auto_approve_threshold": 0.8,
                    "llm_provider": "openai",
                    "llm_model": "gpt-4o",
                }
            ]
        }
    }
