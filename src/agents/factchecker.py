"""Fact-Checker Agent - Verifies claims and assigns confidence scores."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.domain.events import FactCheckCompleted, ResearchCompleted
from src.domain.interfaces import AgentContext
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


# Claim verification status constants
class ClaimStatus:
    """Standard claim verification statuses."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    DISPUTED = "disputed"
    UNVERIFIED = "unverified"


class FactCheckerAgent(BaseAgent[FactCheckCompleted]):
    """Fact-Checker Agent implementation.

    Verifies claims against sources, flags weak or contradictory evidence,
    and assigns confidence scores.
    """

    FACT_CHECKER_SYSTEM_PROMPT = f"""You are a professional fact-checker. Your task is to:
1. Extract claims from the research findings
2. Verify each claim against the provided sources
3. Assign ONE of these status values to each claim:
   - "{ClaimStatus.VERIFIED}" - Fully supported by multiple sources
   - "{ClaimStatus.PARTIALLY_VERIFIED}" - Some support but with gaps or nuances
   - "{ClaimStatus.DISPUTED}" - Contradicted or refuted by sources
   - "{ClaimStatus.UNVERIFIED}" - No clear evidence either way
4. Assign confidence scores (0.0 to 1.0) for each claim
5. Provide reasoning for your assessments

Be objective and cite specific evidence from sources. Use ONLY the status
values listed above."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ):
        super().__init__(
            name="fact_checker",
            description="Verifies claims and assigns confidence scores",
            llm_provider=provider,
            llm_model=model,
            llm_temperature=temperature,
            llm_max_tokens=max_tokens,
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
                "IMPORTANT: You MUST create exactly ONE claim for EACH finding above.\n"
                "Do NOT combine, merge, or summarize multiple findings into one claim.\n"
                "Each finding must become a separate claim with its own status.\n\n"
                "Provide your analysis in JSON format with:\n"
                "- claims: list of objects with 'text' and 'status' keys (MUST have one per finding)\n"
                "- verified_claims: list of verified claims with status\n"
                "- confidence_scores: dict mapping claim text to score (0.0-1.0)\n\n"
                "Each claim must have status: verified, partially_verified, disputed, or unverified"
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
                claims = [{"text": content, "status": ClaimStatus.UNVERIFIED}]
                verified_claims = claims
                confidence_scores = {content: 0.5}
        except json.JSONDecodeError:
            claims = [{"text": content, "status": ClaimStatus.UNVERIFIED}]
            verified_claims = claims
            confidence_scores = {content: 0.5}

        # Normalize claim statuses to ensure valid values
        claims = self._normalize_claim_statuses(claims)
        verified_claims = self._normalize_claim_statuses(verified_claims)

        # Ensure we have a claim for each finding (fallback if LLM missed some)
        claims = self._ensure_claims_coverage(claims, research_event.findings)

        return FactCheckCompleted.create(
            claims=claims,
            verified_claims=verified_claims,
            confidence_scores=confidence_scores,
            correlation_id=context.correlation_id,
        )

    def _ensure_claims_coverage(
        self,
        claims: list[dict[str, Any]],
        findings: list[str],
    ) -> list[dict[str, Any]]:
        """Ensure we have a claim for each finding.

        If the LLM extracted fewer claims than findings, create individual claims
        for the missing findings (since LLM may have combined them or missed them).

        Args:
            claims: Claims extracted by LLM
            findings: Original research findings

        Returns:
            Claims with guaranteed coverage for all findings
        """
        # If we have at least as many claims as findings, we're good
        if len(claims) >= len(findings):
            return claims

        # LLM returned fewer claims than findings - add missing ones
        # (This handles cases where LLM combined findings or missed some)
        findings_set = set(f.strip().lower() for f in findings)

        # Find findings not already covered by claim text
        missing_findings = []
        for finding in findings:
            finding_normalized = finding.strip().lower()
            # Check if this finding's key content is represented in any claim
            # Use first 50 chars as a simple fingerprint
            fingerprint = finding_normalized[:50]
            is_covered = any(
                fingerprint in c.get("text", "").lower()[:50] for c in claims
            )
            if not is_covered:
                missing_findings.append(finding)

        # Create claims for missing findings
        missing_claims = [
            {
                "text": finding,
                "status": ClaimStatus.UNVERIFIED,
                "note": "Auto-generated - LLM did not extract this finding",
            }
            for finding in missing_findings
        ]

        if missing_claims:
            logger.warning(
                f"LLM extracted {len(claims)} claims but found {len(findings)} findings. "
                f"Adding {len(missing_claims)} missing claims with UNVERIFIED status."
            )
            return claims + missing_claims

        return claims

    def _normalize_claim_statuses(
        self, claims: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize claim statuses to valid values.

        Args:
            claims: List of claim dictionaries

        Returns:
            List with normalized status values
        """
        valid_statuses = {
            ClaimStatus.VERIFIED,
            ClaimStatus.PARTIALLY_VERIFIED,
            ClaimStatus.DISPUTED,
            ClaimStatus.UNVERIFIED,
        }

        normalized = []
        for claim in claims:
            status = claim.get("status", "")
            # Normalize to valid status (case-insensitive match)
            status_normalized = status.lower().replace(" ", "_")
            if status_normalized in valid_statuses:
                status = status_normalized
            else:
                status = ClaimStatus.UNVERIFIED
            normalized.append({**claim, "status": status})

        return normalized

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
