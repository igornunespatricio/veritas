"""Critic Agent - Reviews reports for clarity, logic, and completeness."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.domain.events import ReportReviewed, ReportWritten
from src.domain.interfaces import AgentContext


class CriticAgent(BaseAgent[ReportReviewed]):
    """Critic Agent implementation.

    Reviews reports for clarity, logic gaps, bias, and completeness,
    and suggests revisions. Can approve or request changes.
    """

    CRITIC_SYSTEM_PROMPT = """You are a professional editor and critic. Your task is to:
1. Evaluate the report for clarity and readability
2. Identify logic gaps or unsupported claims
3. Check for bias or one-sided presentation
4. Assess completeness - are all key points covered?
5. Suggest specific, actionable improvements
6. Assign a quality score (0.0 to 1.0)

Be thorough but constructive. Your feedback should help improve the report."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ):
        super().__init__(
            name="critic",
            description="Reviews reports for clarity, logic, and completeness",
            llm_provider=provider,
            llm_model=model,
            llm_temperature=temperature,
            llm_max_tokens=max_tokens,
        )

    async def _run(
        self,
        report_event: ReportWritten,
        context: AgentContext,
    ) -> ReportReviewed:
        """Review a written report.

        Args:
            report_event: ReportWritten event with title and content
            context: Agent context with correlation ID

        Returns:
            ReportReviewed event with suggestions and approval status
        """
        messages = [
            SystemMessage(content=self.CRITIC_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Review the following report:\n\n"
                f"TITLE: {report_event.title}\n\n"
                f"CONTENT:\n{report_event.content}\n\n"
                "Provide your review in JSON format with:\n"
                "- suggestions: list of specific improvement suggestions\n"
                "- score: quality score from 0.0 to 1.0\n"
                "- approved: boolean, true if report is ready for publication"
            ),
        ]

        response = await self.llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # Parse JSON response
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                data = json.loads(json_content)
                suggestions = data.get("suggestions", [])
                score = float(data.get("score", 0.5))
                approved = bool(data.get("approved", False))
            else:
                suggestions = ["Unable to parse review - manual review needed"]
                score = 0.5
                approved = False
        except (json.JSONDecodeError, ValueError):
            suggestions = ["Unable to parse review - manual review needed"]
            score = 0.5
            approved = False

        return ReportReviewed.create(
            suggestions=suggestions,
            score=score,
            approved=approved,
            correlation_id=context.correlation_id,
        )

    async def validate_input(self, input: Any) -> bool:
        """Validate input is a ReportWritten event."""
        return isinstance(input, ReportWritten)

    async def review(
        self,
        report: ReportWritten,
        context: AgentContext,
    ) -> ReportReviewed:
        """Convenience method to review a report."""
        return await self.execute(report, context)
