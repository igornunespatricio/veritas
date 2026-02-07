"""Test script to verify API key availability for all LLM providers."""

import sys

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from src.config import settings


def test_openai_api() -> bool:
    """Test OpenAI API key availability."""
    print("Testing OpenAI API...")
    try:
        client = ChatOpenAI(api_key=settings.openai_api_key, model="gpt-4o-mini")
        response = client.invoke([{"role": "user", "content": "test"}])
        print(
            f"  ✓ OpenAI API: OK (model: {response.response_metadata.get('model_name', 'unknown')})"
        )
        return True
    except Exception as e:
        print(f"  ✗ OpenAI API: FAILED - {e}")
        return False


def test_anthropic_api() -> bool:
    """Test Anthropic API key availability."""
    print("Testing Anthropic API...")
    try:
        client = ChatAnthropic(
            api_key=settings.anthropic_api_key, model_name="claude-sonnet-4-20250514"
        )
        response = client.invoke([{"role": "user", "content": "test"}])
        print(f"  ✓ Anthropic API: OK (model: {response.id})")
        return True
    except Exception as e:
        print(f"  ✗ Anthropic API: FAILED - {e}")
        return False


def test_openrouter_api() -> bool:
    """Test OpenRouter API key availability using LangChain."""
    print("Testing OpenRouter API...")
    if not settings.openrouter_api_key:
        print("  ⊘ OpenRouter API: SKIPPED (no API key configured)")
        return True
    try:
        client = ChatOpenAI(
            model="openai/gpt-5-nano",
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.invoke([{"role": "user", "content": "test"}])
        print(
            f"  ✓ OpenRouter API: OK (model: {response.response_metadata.get('model_name', 'unknown')})"
        )
        return True
    except Exception as e:
        print(f"  ✗ OpenRouter API: FAILED - {e}")
        return False


def main() -> int:
    """Run all API availability tests."""
    print("=" * 50)
    print("API Key Availability Test")
    print("=" * 50)
    print()

    results = {
        "OpenAI": test_openai_api(),
        "Anthropic": test_anthropic_api(),
        "OpenRouter": test_openrouter_api(),
    }

    print()
    print("=" * 50)
    passed = sum(results.values())
    total = len(results)
    print(f"Results: {passed}/{total} APIs available")
    print("=" * 50)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
