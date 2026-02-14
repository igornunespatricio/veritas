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


class ResearcherAgent(BaseAgent[ResearchCompleted]):
    """Researcher Agent implementation.

    Collects raw information, sources, and key findings from the web
    using Tavily search and outputs structured notes with source metadata.
    """

    RESEARCHER_SYSTEM_PROMPT = """You are a professional researcher. Your task is to:
1. Thoroughly research the given topic using the web search tool
2. Find reliable sources and cite them properly
3. Extract key findings and facts
4. Organize information in a structured format

When you need current or specific information, use the search_web tool.
Always verify information from multiple sources when possible.
Provide citations for all claims with URLs and publication dates.

Return your findings in JSON format with:
- sources: list of source objects with url, title, date
- findings: list of key findings as strings
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

        Uses direct tool calling pattern with web search tool to gather
        information and produce structured research output.

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

        # Create LLM with tools bound
        llm_with_tools = llm.bind_tools([search_web])

        # Step 1: Ask the model to research the topic
        research_prompt = (
            f"Research the following topic thoroughly using the web search tool:\n\n"
            f"Topic: {topic}\n\n"
            f"Use the search_web tool to find relevant information, then provide your findings "
            f"in JSON format with 'sources' and 'findings' keys."
        )

        response = await llm_with_tools.ainvoke([HumanMessage(content=research_prompt)])

        # Step 2: Check if the model wants to call a tool
        # Use getattr to safely access tool_calls with proper typing
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            # Execute the tool calls
            tool_outputs = []

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})

                # Call the search tool
                if tool_name == "search_web":
                    query = tool_args.get("query", topic)
                    search_result = search_web.invoke(query)
                    tool_outputs.append(
                        {
                            "tool": tool_name,
                            "output": search_result,
                        }
                    )

            # Step 3: Send tool outputs back to the model for final summary
            if tool_outputs:
                # Create tool response messages
                from langchain_core.messages import ToolMessage

                tool_messages = [
                    ToolMessage(
                        content=tool_output["output"],
                        tool_call_id=f"call_{i}",
                        name=tool_output["tool"],
                    )
                    for i, tool_output in enumerate(tool_outputs)
                ]

                # Ask for final structured output
                final_prompt = (
                    f"Based on the search results above, provide your final research findings "
                    f"in JSON format with:\n"
                    f"- 'sources': list of source objects with url, title, and date\n"
                    f"- 'findings': list of key findings as strings\n\n"
                    f"Topic: {topic}"
                )

                final_response = await llm_with_tools.ainvoke(
                    tool_messages + [HumanMessage(content=final_prompt)]
                )

                # Extract content from final response
                content = (
                    final_response.content
                    if hasattr(final_response, "content")
                    else str(final_response)
                )
                sources, findings = self._parse_response(content)
            else:
                sources, findings = self._parse_response("")
        else:
            # No tool calls, parse direct response
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

    async def _run_direct(
        self,
        topic: str,
        context: AgentContext,
    ) -> ResearchCompleted:
        """Run research using direct invocation (fallback).

        Performs web search directly and processes results.

        Args:
            topic: The research topic
            context: Agent context

        Returns:
            ResearchCompleted event
        """
        # Perform web search
        search_result = self._search_tool.invoke(topic)

        # Use LLM to extract structured findings from search results
        messages = [
            SystemMessage(
                content=self.RESEARCHER_SYSTEM_PROMPT
                + "\n\nBased on the search results provided, extract key findings and sources."
            ),
            HumanMessage(
                content=f"Research topic: {topic}\n\nSearch results:\n{search_result}\n\n"
                "Provide findings in JSON format with sources and findings."
            ),
        ]

        response = await self.llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

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
