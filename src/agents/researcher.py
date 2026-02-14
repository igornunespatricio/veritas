"""Researcher Agent - Collects raw information and key findings using web search."""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.agents.base import BaseAgent
from src.domain.events import ResearchCompleted
from src.domain.interfaces import AgentContext
from src.infrastructure.tools import get_web_search_tool

logger = logging.getLogger(__name__)


# Define web search tool for the agent
@tool
def search_web(query: str) -> str:
    """Search the web for information on a given query.

    Args:
        query: The search query to look up

    Returns:
        Search results with relevant sources
    """
    search_tool_func = get_web_search_tool(max_results=5)
    if search_tool_func is None:
        return "Web search is not configured. Please set TAVILY_API_KEY."

    try:
        result = search_tool_func(query)
        return result
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Web search failed: {str(e)}"


# Define format tool for structured output
@tool
def format_report(sources: list[dict], findings: list[str]) -> str:
    """Format research findings into structured output.

    Args:
        sources: List of source objects with url, title, date
        findings: List of key findings

    Returns:
        Formatted JSON string
    """
    return json.dumps({"sources": sources, "findings": findings})


class ResearcherAgent(BaseAgent[ResearchCompleted]):
    """Researcher Agent implementation.

    Collects raw information, sources, and key findings from the web
    using Tavily search and outputs structured notes with source metadata.
    """

    RESEARCHER_SYSTEM_PROMPT = """You are a professional researcher. Your task is to:
1. Thoroughly research the given topic using the web search tool
2. Find reliable sources and cite them properly
3. Extract key findings and facts from EACH source
4. Organize information in a structured format

IMPORTANT: You must extract AT LEAST 5 distinct findings from the search results.
Each finding should be a unique piece of information from a different source.

When you need current or specific information, use the search_web tool.
Always verify information from multiple sources when possible.
Provide citations for all claims with URLs and publication dates.

Return your findings in JSON format with:
- sources: list of source objects with url, title, date (include ALL sources found)
- findings: list of AT LEAST 5 distinct key findings as strings
"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ):
        """Initialize researcher agent.

        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (None = unlimited)
        """
        super().__init__(
            name="researcher",
            description="Collects raw information, sources, and key findings",
            llm_provider=provider,
            llm_model=model,
            llm_temperature=temperature,
            llm_max_tokens=max_tokens,
        )

        # Initialize the web search tool
        self._search_tool = search_web

        # For ReAct agent pattern
        self._tools = [search_web]

    async def _run(
        self,
        topic: str,
        context: AgentContext,
    ) -> ResearchCompleted:
        """Execute research on the given topic.

        Uses direct invocation with Tavily search to ensure we get
        all search results, then uses LLM to extract findings.

        Args:
            topic: The research topic
            context: Agent context with correlation ID

        Returns:
            ResearchCompleted event with sources and findings
        """
        llm = self.llm.llm

        # Check if LLM supports tool calling
        if not hasattr(llm, "bind_tools"):
            logger.warning("LLM doesn't support tool calling, using direct invocation")
            return await self._run_direct(topic, context)

        # Use direct invocation for more reliable results with small models
        # This bypasses the ReAct pattern issues with Ollama 3.2:3b
        return await self._run_direct(topic, context)

    async def _run_direct(
        self,
        topic: str,
        context: AgentContext,
    ) -> ResearchCompleted:
        """Run research using direct invocation with tool binding.

        Performs web search directly and processes results with LLM.

        Args:
            topic: The research topic
            context: Agent context

        Returns:
            ResearchCompleted event
        """
        llm = self.llm.llm

        # Perform web search directly
        search_result = self._search_tool.invoke(topic)

        # Format the search results nicely
        formatted_results = f"""TOPIC: {topic}

SEARCH RESULTS:
{search_result}

IMPORTANT: Extract AT LEAST 5 distinct findings from the search results above.
For each finding, identify which source it came from.

Provide your findings in EXACTLY this JSON format:
{{
    "sources": [
        {{"url": "source url", "title": "source title", "date": "publication date or N/A"}}
    ],
    "findings": [
        "Finding 1: ...",
        "Finding 2: ...",
        "Finding 3: ...",
        "Finding 4: ...",
        "Finding 5: ..."
    ]
}}

DO NOT include any other text - ONLY the JSON object."""

        # Use LLM with bind_tools for structured output
        if hasattr(llm, "bind_tools"):
            llm_with_tools = llm.bind_tools([format_report])

            response = await llm_with_tools.ainvoke(
                [HumanMessage(content=formatted_results)]
            )

            # Check if tool was called
            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    if tool_name == "format_report":
                        result = format_report.invoke(tool_call.get("args", {}))
                        data = json.loads(result)
                        sources = data.get("sources", [])
                        findings = data.get("findings", [])

                        return ResearchCompleted.create(
                            topic=topic,
                            sources=sources,
                            findings=findings,
                            correlation_id=context.correlation_id,
                        )

            # Fallback to direct response
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
        else:
            response = await self.llm.ainvoke([HumanMessage(content=formatted_results)])
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

        sources, findings = self._parse_response(content)

        return ResearchCompleted.create(
            topic=topic,
            sources=sources,
            findings=findings,
            correlation_id=context.correlation_id,
        )

    def _parse_response(self, content: Any) -> tuple[list[dict], list[str]]:
        """Parse JSON from LLM response.

        Args:
            content: Raw response content (can be str or list)

        Returns:
            Tuple of (sources, findings)
        """
        # Ensure content is a string
        if isinstance(content, list):
            # Handle list responses (e.g., from some Ollama versions)
            content = str(content)
        elif not isinstance(content, str):
            content = str(content)

        try:
            # Try to extract JSON from response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                data = json.loads(json_content)
                sources = data.get("sources", [])
                findings = data.get("findings", [])
            else:
                # If no JSON found, use the entire content
                sources = [{"url": "", "title": "", "date": "", "content": content}]
                findings = [content]
        except json.JSONDecodeError:
            sources = [{"url": "", "title": "", "date": "", "content": content}]
            findings = [content]

        return sources, findings

    async def validate_input(self, input: Any) -> bool:
        """Validate research topic input."""
        if isinstance(input, str):
            return len(input.strip()) > 0
        if isinstance(input, dict) and "topic" in input:
            return len(input["topic"].strip()) > 0
        return False

    async def research(
        self,
        topic: str,
        context: AgentContext,
    ) -> ResearchCompleted:
        """Convenience method to run research."""
        return await self.execute(topic, context)
