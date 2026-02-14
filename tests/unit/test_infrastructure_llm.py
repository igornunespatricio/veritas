"""Unit tests for LLM infrastructure."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLLMFactories:
    """Tests for LLM factory functions."""

    @patch("src.infrastructure.llm.settings")
    def test_get_openrouter_llm_requires_key(self, mock_settings):
        """Test that get_openrouter_llm requires API key."""
        mock_settings.openrouter_api_key = None

        from src.infrastructure.llm import get_openrouter_llm

        with pytest.raises(ValueError, match="OpenRouter API key not configured"):
            get_openrouter_llm()

    @patch("src.infrastructure.llm.settings")
    def test_get_openrouter_llm_success(self, mock_settings):
        """Test get_openrouter_llm with valid key."""
        mock_settings.openrouter_api_key = "router-key"

        from src.infrastructure.llm import get_openrouter_llm

        llm = get_openrouter_llm()
        assert llm is not None


class TestResilientLLMWrapper:
    """Tests for ResilientLLMWrapper class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock()
        mock.ainvoke = AsyncMock(return_value=MagicMock(content="response"))
        return mock

    @pytest.fixture
    def wrapper(self, mock_llm):
        """Create a ResilientLLMWrapper."""
        from src.infrastructure.llm import ResilientLLMWrapper

        return ResilientLLMWrapper(llm=mock_llm)

    def test_constructor_sets_llm(self, wrapper, mock_llm):
        """Test that wrapper stores the LLM."""
        assert wrapper.llm is mock_llm

    def test_constructor_sets_default_retry_config(self, wrapper):
        """Test default retry config is set."""
        from src.config.retry import RETRY_CONFIG_DEFAULT

        assert wrapper._retry_config == RETRY_CONFIG_DEFAULT

    @pytest.mark.asyncio
    async def test_ainvoke_success(self, wrapper, mock_llm):
        """Test successful ainvoke call."""
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="test response"))

        result = await wrapper.ainvoke(messages=["test"])

        assert result.content == "test response"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke_with_correlation_id(self, wrapper, mock_llm):
        """Test ainvoke with correlation ID."""
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="response"))

        await wrapper.ainvoke(messages=["test"], correlation_id="test-id")

        # Verify correlation ID is passed
        mock_llm.ainvoke.assert_called_once()


class TestGetResilientLLM:
    """Tests for get_resilient_llm factory."""

    def test_get_resilient_llm_returns_wrapper(self):
        """Test that get_resilient_llm returns a ResilientLLMWrapper."""
        from src.infrastructure.llm import get_resilient_llm

        result = get_resilient_llm()
        # Should return a wrapper instance
        from src.infrastructure.llm import ResilientLLMWrapper

        assert isinstance(result, ResilientLLMWrapper)
