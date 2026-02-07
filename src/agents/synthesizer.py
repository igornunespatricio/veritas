"""Synthesizer Agent - Merges validated research into coherent insights."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.domain.events import FactCheckCompleted, ResearchCompleted, SynthesisCompleted
from src.domain.interfaces import AgentContext
from src.agents.base import BaseAgent


class SynthesizerAgent(BaseAgent[SynthesisCompleted]):
    """Synthesizer Agent implementation.

    Merges validated research into coherent insights, resolves overlaps
    and contradictions between findings.
    """

    SYNTHESIZER_SYSTEM_PROMPT = """You are an expert research synthesizer. Your task is to:
1. Combine research findings with fact-check verification results
2. Identify themes and patterns across sources
3. Resolve contradictions by weighing evidence quality
4. Create coherent insights that integrate multiple sources
5. Highlight areas of consensus and disagreement

Focus on creating actionable insights from the data."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.5,
    ):
        super().__init__(
            name="synthesizer",
            description="Merges validated research into coherent insights",
            llm_provider=provider,
            llm_model=model,
            llm_temperature=temperature,
        )

    async def _run(
        self,
        inputs: dict[str, Any],
        context: AgentContext,
    ) -> SynthesisCompleted:
        """Synthesize research and fact-check into insights.

        Args:
            inputs: Dict with 'research' and 'fact_check' events
            context: Agent context with correlation ID

        Returns:
            SynthesisCompleted event with insights and resolved contradictions
        """
        research: ResearchCompleted = inputs.get("research")
        fact_check: FactCheckCompleted = inputs.get("fact_check")

        findings_text = "\n".join(f"- {finding}" for finding in research.findings)

        # Extract confidence scores for synthesis
        confidence_info = "\n".join(
            f"- {claim}: {score}"
            for claim, score in fact_check.confidence_scores.items()
        )

        messages = [
            SystemMessage(content=self.SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Synthesize the following research and fact-check results:\n\n"
                f"TOPIC: {research.topic}\n\n"
                f"FINDINGS:\n{findings_text}\n\n"
                f"FACT-CHECK CONFIDENCE SCORES:\n{confidence_info}\n\n"
                "Provide your synthesis in JSON format with:\n"
                "- insights: list of coherent insights\n"
                "- resolved_contradictions: list of how contradictions were resolved"
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
                insights = data.get("insights", [])
                resolved_contradictions = data.get("resolved_contradictions", [])
            else:
                insights = [content]
                resolved_contradictions = []
        except json.JSONDecodeError:
            insights = [content]
            resolved_contradictions = []

        return SynthesisCompleted.create(
            insights=insights,
            resolved_contradictions=resolved_contradictions,
            correlation_id=context.correlation_id,
        )

    async def validate_input(self, input: Any) -> bool:
        """Validate input contains research and fact_check events."""
        if isinstance(input, dict):
            return "research" in input and "fact_check" in input
        return False

    async def synthesize(
        self,
        research: ResearchCompleted,
        fact_check: FactCheckCompleted,
        context: AgentContext,
    ) -> SynthesisCompleted:
        """Convenience method to run synthesis."""
        return await self.execute(
            {"research": research, "fact_check": fact_check},
            context,
        )
