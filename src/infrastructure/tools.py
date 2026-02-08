"""Web search and tool infrastructure using Tavily SDK."""

import logging
from typing import Any

from tavily import TavilyClient

from src.config import settings

logger = logging.getLogger(__name__)


def get_tavily_client() -> TavilyClient:
    """Create a Tavily client.

    Returns:
        Configured TavilyClient instance

    Raises:
        ValueError: If API key is not configured
    """
    api_key = settings.tavily_api_key
    if not api_key:
        logger.warning("Tavily API key not configured. Web search will not work.")
        raise ValueError(
            "TAVILY_API_KEY environment variable is required for web search"
        )

    return TavilyClient(api_key=api_key)


def get_web_search_tool(max_results: int = 5) -> Any:
    """Create a web search tool using Tavily.

    This returns a tool-compatible function that can be used with LangChain agents.

    Args:
        max_results: Maximum number of search results to return

    Returns:
        A tool function for web search
    """
    try:
        client = get_tavily_client()
    except ValueError:
        logger.error("Failed to create Tavily client - API key not configured")
        return None

    def search(query: str) -> str:
        """Search the web for information.

        Args:
            query: The search query

        Returns:
            Search results as formatted string
        """
        try:
            response = client.search(query=query, max_results=max_results)
            # Format the response nicely
            results = response.get("results", [])
            if not results:
                return "No results found."

            formatted = []
            for r in results:
                formatted.append(f"Title: {r.get('title', 'N/A')}")
                formatted.append(f"URL: {r.get('url', 'N/A')}")
                formatted.append(f"Content: {r.get('content', 'N/A')[:300]}...")
                formatted.append("---")

            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return f"Search failed: {str(e)}"

    return search
