"""LLM client infrastructure using LangChain."""

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.config import settings


def get_openai_llm(
    model: str = "gpt-4o",
    temperature: float = 0.7,
) -> ChatOpenAI:
    """Get configured OpenAI LLM client.

    Args:
        model: Model name to use
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)

    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


def get_anthropic_llm(
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
) -> ChatAnthropic:
    """Get configured Anthropic LLM client.

    Args:
        model: Model name to use
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)

    Returns:
        Configured ChatAnthropic instance
    """
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=settings.anthropic_api_key,
    )


def get_llm(provider: str = "openai", **kwargs) -> ChatOpenAI | ChatAnthropic:
    """Factory function to get LLM client based on provider.

    Args:
        provider: LLM provider ("openai" or "anthropic")
        **kwargs: Additional arguments passed to LLM constructor

    Returns:
        Configured LLM client
    """
    if provider == "anthropic":
        return get_anthropic_llm(**kwargs)
    return get_openai_llm(**kwargs)
