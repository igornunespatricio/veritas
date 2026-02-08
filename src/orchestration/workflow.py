"""Orchestration layer for multi-agent workflows."""

from dataclasses import dataclass
from enum import Enum

from ..agents import (
    CriticAgent,
    FactCheckerAgent,
    ResearcherAgent,
    SynthesizerAgent,
    WriterAgent,
)
from ..domain.events import (
    FactCheckCompleted,
    ReportReviewed,
    ReportWritten,
    ResearchCompleted,
    SynthesisCompleted,
)
from ..domain.interfaces import AgentContext
from ..infrastructure.logging import get_logger, log_stage

logger = get_logger(__name__)


class WorkflowStage(Enum):
    """Workflow execution stages."""

    RESEARCH = "research"
    FACT_CHECK = "fact_check"
    SYNTHESIS = "synthesis"
    WRITING = "writing"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    status: WorkflowStage
    research: ResearchCompleted | None = None
    fact_check: FactCheckCompleted | None = None
    synthesis: SynthesisCompleted | None = None
    report: ReportWritten | None = None
    review: ReportReviewed | None = None
    error: str | None = None
    iterations: int = 0


class ResearchWorkflow:
    """Orchestrates the multi-agent research workflow.

    Coordinates: Researcher → Fact-Checker → Synthesizer → Writer → Critic
    Supports iterative workflows where Critic feedback triggers revision.
    """

    def __init__(
        self,
        max_iterations: int = 3,
        auto_approve_threshold: float = 0.8,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        max_tokens: int | None = None,
    ):
        """Initialize workflow with agents.

        Args:
            max_iterations: Maximum review iterations
            auto_approve_threshold: Score threshold for auto-approval
            llm_provider: LLM provider to use ("openai" or "anthropic")
            llm_model: Model name to use (e.g., "gpt-4o", "claude-sonnet-4-20250514")
            max_tokens: Maximum tokens per LLM call (None = unlimited)
        """
        self.max_iterations = max_iterations
        self.auto_approve_threshold = auto_approve_threshold
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_tokens = max_tokens

        # Initialize agents with specified LLM provider/model
        self.researcher = ResearcherAgent(
            provider=llm_provider,
            model=llm_model,
            max_tokens=max_tokens,
        )
        self.fact_checker = FactCheckerAgent(
            provider=llm_provider,
            model=llm_model,
            max_tokens=max_tokens,
        )
        self.synthesizer = SynthesizerAgent(
            provider=llm_provider,
            model=llm_model,
            max_tokens=max_tokens,
        )
        self.writer = WriterAgent(
            provider=llm_provider,
            model=llm_model,
            max_tokens=max_tokens,
        )
        self.critic = CriticAgent(
            provider=llm_provider,
            model=llm_model,
            max_tokens=max_tokens,
        )

    async def execute(
        self,
        topic: str,
        correlation_id: str | None = None,
    ) -> WorkflowResult:
        """Execute the complete research workflow.

        Args:
            topic: Research topic
            correlation_id: Optional correlation ID for tracing

        Returns:
            WorkflowResult with all outputs
        """
        context = AgentContext.create(correlation_id=correlation_id)
        result = WorkflowResult(status=WorkflowStage.RESEARCH)

        logger.info(f"🚀 Starting research workflow: {topic}")

        try:
            # Stage 1: Research
            log_stage("RESEARCH", "Gathering sources and findings...")
            result.research = await self.researcher.research(topic, context)
            log_stage(
                "RESEARCH",
                f"✅ Found {len(result.research.sources)} sources, {len(result.research.findings)} findings",
            )
            result.status = WorkflowStage.FACT_CHECK

            # Stage 2: Fact-Check
            log_stage("FACT-CHECK", "Verifying claims against sources...")
            result.fact_check = await self.fact_checker.verify_claims(
                claims=result.research.findings,
                sources=result.research.sources,
                context=context,
            )
            verified = len(
                [c for c in result.fact_check.claims if c.get("status") == "verified"]
            )
            partially = len(
                [
                    c
                    for c in result.fact_check.claims
                    if c.get("status") == "partially_verified"
                ]
            )
            disputed = len(
                [c for c in result.fact_check.claims if c.get("status") == "disputed"]
            )
            unverified = len(
                [c for c in result.fact_check.claims if c.get("status") == "unverified"]
            )
            log_stage(
                "FACT-CHECK",
                f"✅ Verified: {verified} | Partial: {partially} | Disputed: {disputed} | Unverified: {unverified}",
            )
            result.status = WorkflowStage.SYNTHESIS

            # Stage 3: Synthesis
            log_stage("SYNTHESIS", "Merging research into coherent insights...")
            result.synthesis = await self.synthesizer.synthesize(
                research=result.research,
                fact_check=result.fact_check,
                context=context,
            )
            log_stage(
                "SYNTHESIS", f"✅ Generated {len(result.synthesis.insights)} insights"
            )
            result.status = WorkflowStage.WRITING

            # Stage 4: Writing
            log_stage("WRITING", "Drafting structured report...")
            result.report = await self.writer.write_report(
                synthesis=result.synthesis,
                format="markdown",
                context=context,
            )
            log_stage(
                "WRITING", f"✅ Report complete ({len(result.report.content)} chars)"
            )
            result.status = WorkflowStage.REVIEW

            # Stage 5: Review (with iteration)
            for iteration in range(self.max_iterations):
                log_stage(
                    "REVIEW", f"Iteration {iteration + 1}/{self.max_iterations}..."
                )
                result.review = await self.critic.review(result.report, context)
                result.iterations = iteration + 1

                if result.review.approved:
                    log_stage(
                        "REVIEW",
                        f"✅ Report approved (score: {result.review.score:.2f})",
                    )
                    break

                if result.review.score >= self.auto_approve_threshold:
                    log_stage(
                        "REVIEW", f"✅ Auto-approved (score: {result.review.score:.2f})"
                    )
                    break

                # Revision needed - rewrite with feedback
                log_stage(
                    "REVIEW", f"⚠️  Needs revision (score: {result.review.score:.2f})"
                )
                # Create enhanced synthesis with criticism
                # For simplicity, re-run synthesis and writing
                result.synthesis = await self.synthesizer.synthesize(
                    research=result.research,
                    fact_check=result.fact_check,
                    context=context,
                )
                result.report = await self.writer.write_report(
                    synthesis=result.synthesis,
                    format="markdown",
                    context=context,
                )

            result.status = WorkflowStage.COMPLETED
            logger.info(
                "Workflow completed successfully",
                extra={"correlation_id": context.correlation_id},
            )

        except Exception as e:
            result.status = WorkflowStage.FAILED
            result.error = str(e)
            logger.error(
                f"Workflow failed: {e}",
                extra={"correlation_id": context.correlation_id},
            )

        return result

    async def execute_sequential(
        self,
        topic: str,
        correlation_id: str | None = None,
    ) -> WorkflowResult:
        """Execute workflow without iterations (single pass)."""
        context = AgentContext.create(correlation_id=correlation_id)
        result = WorkflowResult(status=WorkflowStage.RESEARCH)

        try:
            result.research = await self.researcher.research(topic, context)
            result.fact_check = await self.fact_checker.verify_claims(
                claims=result.research.findings,
                sources=result.research.sources,
                context=context,
            )
            result.synthesis = await self.synthesizer.synthesize(
                research=result.research,
                fact_check=result.fact_check,
                context=context,
            )
            result.report = await self.writer.write_report(
                synthesis=result.synthesis,
                format="markdown",
                context=context,
            )
            result.status = WorkflowStage.COMPLETED
        except Exception as e:
            result.status = WorkflowStage.FAILED
            result.error = str(e)

        return result
