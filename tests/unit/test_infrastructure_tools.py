"""Unit tests for web search tools infrastructure."""

from unittest.mock import MagicMock, patch

import pytest


class TestWebSearchTools:
    """Tests for web search tool functions."""

    @patch("src.infrastructure.tools.settings")
    def test_get_tavily_client_success(self, mock_settings):
        """Test creating Tavily client with valid API key."""
        mock_settings.tavily_api_key = "test-key"

        from src.infrastructure.tools import get_tavily_client

        client = get_tavily_client()
        assert client is not None

    @patch("src.infrastructure.tools.settings")
    def test_get_tavily_client_no_api_key(self, mock_settings):
        """Test that missing API key raises ValueError."""
        mock_settings.tavily_api_key = None

        from src.infrastructure.tools import get_tavily_client

        with pytest.raises(
            ValueError, match="TAVILY_API_KEY environment variable is required"
        ):
            get_tavily_client()

    @patch("src.infrastructure.tools.settings")
    def test_get_web_search_tool_success(self, mock_settings):
        """Test creating web search tool with valid API key."""
        mock_settings.tavily_api_key = "test-key"

        from src.infrastructure.tools import get_web_search_tool

        tool = get_web_search_tool()
        assert tool is not None
        assert callable(tool)

    @patch("src.infrastructure.tools.settings")
    def test_get_web_search_tool_no_api_key(self, mock_settings):
        """Test that missing API key returns None."""
        mock_settings.tavily_api_key = None

        from src.infrastructure.tools import get_web_search_tool

        tool = get_web_search_tool()
        assert tool is None


class TestWebSearchFunction:
    """Tests for the web search function."""

    @patch("src.infrastructure.tools.get_tavily_client")
    def test_search_cleans_query(self, mock_get_client):
        """Test that search cleans query string."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_get_client.return_value = mock_client

        from src.infrastructure.tools import get_web_search_tool

        tool = get_web_search_tool(max_results=5)
        result = tool("What is AI?")

        # Should call search with cleaned query
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args[1]["query"] == "What is AI?"

    @patch("src.infrastructure.tools.get_tavily_client")
    def test_search_handles_react_format(self, mock_get_client):
        """Test that search handles ReAct tool format."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_get_client.return_value = mock_client

        from src.infrastructure.tools import get_web_search_tool

        tool = get_web_search_tool()
        # Query with Action Input format
        result = tool(
            "Search for: What is machine learning? Action Input: What is machine learning?"
        )

        # Should extract the actual query
        mock_client.search.assert_called()

    @patch("src.infrastructure.tools.get_tavily_client")
    def test_search_truncates_long_query(self, mock_get_client):
        """Test that long queries are truncated."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_get_client.return_value = mock_client

        from src.infrastructure.tools import get_web_search_tool

        tool = get_web_search_tool()
        long_query = "a" * 500  # Exceeds 400 char limit
        result = tool(long_query)

        call_args = mock_client.search.call_args
        assert len(call_args[1]["query"]) <= 400

    @patch("src.infrastructure.tools.get_tavily_client")
    def test_search_formats_results(self, mock_get_client):
        """Test that search formats results nicely."""
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Test Article",
                    "url": "http://example.com",
                    "content": "This is the content of the article...",
                }
            ]
        }
        mock_get_client.return_value = mock_client

        from src.infrastructure.tools import get_web_search_tool

        tool = get_web_search_tool()
        result = tool("test query")

        assert "Test Article" in result
        assert "http://example.com" in result

    @patch("src.infrastructure.tools.get_tavily_client")
    def test_search_handles_no_results(self, mock_get_client):
        """Test handling when no results found."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_get_client.return_value = mock_client

        from src.infrastructure.tools import get_web_search_tool

        tool = get_web_search_tool()
        result = tool("test query")

        assert result == "No results found."

    @patch("src.infrastructure.tools.get_tavily_client")
    def test_search_handles_error(self, mock_get_client):
        """Test handling when search fails."""
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Network error")
        mock_get_client.return_value = mock_client

        from src.infrastructure.tools import get_web_search_tool

        tool = get_web_search_tool()
        result = tool("test query")

        assert "Search failed" in result

    @patch("src.infrastructure.tools.get_tavily_client")
    def test_search_max_results_parameter(self, mock_get_client):
        """Test that max_results parameter is respected."""
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_get_client.return_value = mock_client

        from src.infrastructure.tools import get_web_search_tool

        tool = get_web_search_tool(max_results=10)
        result = tool("test")

        call_args = mock_client.search.call_args
        assert call_args[1]["max_results"] == 10
