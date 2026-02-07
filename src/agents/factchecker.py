"""Fact-Checker Agent - Verifies claims and assigns confidence scores."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.domain.events import FactCheckCompleted, ResearchCompleted
from src.domain.interfaces import AgentContext
from src.agents.base import BaseAgent


class FactCheckerAgent(BaseAgent[FactCheckCompleted]):
    """Fact-Checker Agent implementation.

    Verifies claims against sources, flags weak or contradictory evidence,
    and assigns confidence scores.
    """

    FACT_CHECKER_SYSTEM_PROMPT = """You are a professional fact-checker. Your task is to:
1. Extract claims from the research findings
2. Verify each claim against the provided sources
3. Flag claims that are:
   - Unsupported by evidence
   - Contradicted by sources
   - Partially supported
4. Assign confidence scores (0.0 to 1.0) for each claim
5. Provide reasoning for your assessments

Be objective and cite specific evidence from sources."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.3,  # Lower temperature for factual accuracy
    ):
        super().__init__(
            name="fact_checker",
            description="Verifies claims and assigns confidence scores",
            llm_provider=provider,
            llm_model=model,
            llm_temperature=temperature,
        )

    async def _run(
        self,
        research_event: ResearchCompleted,
        context: AgentContext,
    ) -> FactCheckCompleted:
        """Verify claims from research findings.

        Args:
            research_event: ResearchCompleted event with findings
            context: Agent context with correlation ID

        Returns:
            FactCheckCompleted event with verified claims and scores
        """
        findings_text = "\n".join(f"- {finding}" for finding in research_event.findings)
        sources_text = "\n".join(
            f"- {source.get('title', '')}: {source.get('url', '')}"
            for source in research_event.sources
        )

        messages = [
            SystemMessage(content=self.FACT_CHECKER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Fact-check the following research:\n\n"
                f"TOPIC: {research_event.topic}\n\n"
                f"FINDINGS:\n{findings_text}\n\n"
                f"SOURCES:\n{sources_text}\n\n"
                "Provide your analysis in JSON format with:\n"
                "- claims: list of extracted claims\n"
                "- verified_claims: list of claims with verification status\n"
                "- confidence_scores: dict mapping claim to score (0.0-1.0)"
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
                claims = data.get("claims", [])
                verified_claims = data.get("verified_claims", [])
                confidence_scores = data.get("confidence_scores", {})
            else:
                claims = [{"text": content, "status": "unverified"}]
                verified_claims = claims
                confidence_scores = {content: 0.5}
        except json.JSONDecodeError:
            claims = [{"text": content, "status": "unverified"}]
            verified_claims = claims
            confidence_scores = {content: 0.5}

        return FactCheckCompleted.create(
            claims=claims,
            verified_claims=verified_claims,
            confidence_scores=confidence_scores,
            correlation_id=context.correlation_id,
        )

    async def validate_input(self, input: Any) -> bool:
        """Validate input is a ResearchCompleted event."""
        return isinstance(input, ResearchCompleted)

    async def verify_claims(
        self,
        claims: list[str],
        sources: list[dict[str, str]],
        context: AgentContext,
    ) -> FactCheckCompleted:
        """Verify claims against sources."""
        # Create a synthetic research event for compatibility
        research_event = ResearchCompleted.create(
            topic="",
            sources=sources,
            findings=claims,
            correlation_id=context.correlation_id,
        )
        return await self.execute(research_event, context)
