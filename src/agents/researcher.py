"""Researcher Agent - Collects raw information and key findings."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.domain.events import ResearchCompleted
from src.domain.interfaces import AgentContext
from src.agents.base import BaseAgent


class ResearcherAgent(BaseAgent[ResearchCompleted]):
    """Researcher Agent implementation.

    Collects raw information, sources, and key findings from the web
    and outputs structured notes with source metadata.
    """

    RESEARCHER_SYSTEM_PROMPT = """You are a professional researcher. Your task is to:
1. Thoroughly research the given topic
2. Find reliable sources and cite them properly
3. Extract key findings and facts
4. Organize information in a structured format

Always verify information from multiple sources when possible.
Provide citations for all claims with URLs and publication dates."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ):
        super().__init__(
            name="researcher",
            description="Collects raw information, sources, and key findings",
            llm_provider=provider,
            llm_model=model,
            llm_temperature=temperature,
        )

    async def _run(
        self,
        topic: str,
        context: AgentContext,
    ) -> ResearchCompleted:
        """Execute research on the given topic.

        Args:
            topic: The research topic
            context: Agent context with correlation ID

        Returns:
            ResearchCompleted event with sources and findings
        """
        messages = [
            SystemMessage(content=self.RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Research the following topic thoroughly:\n\n{topic}\n\n"
                "Provide your findings in JSON format with:\n"
                "- sources: list of source objects with url, title, date\n"
                "- findings: list of key findings as strings"
            ),
        ]

        response = await self.llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # Parse JSON response
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
                sources = [{"url": "", "title": "", "date": "", "content": content}]
                findings = [content]
        except json.JSONDecodeError:
            sources = [{"url": "", "title": "", "date": "", "content": content}]
            findings = [content]

        return ResearchCompleted.create(
            topic=topic,
            sources=sources,
            findings=findings,
            correlation_id=context.correlation_id,
        )

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
