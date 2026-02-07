"""Orchestration layer for multi-agent workflows."""

import logging
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

logger = logging.getLogger(__name__)


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
    ):
        """Initialize workflow with agents.

        Args:
            max_iterations: Maximum review iterations
            auto_approve_threshold: Score threshold for auto-approval
        """
        self.max_iterations = max_iterations
        self.auto_approve_threshold = auto_approve_threshold

        # Initialize agents
        self.researcher = ResearcherAgent()
        self.fact_checker = FactCheckerAgent()
        self.synthesizer = SynthesizerAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

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

        logger.info(
            f"Starting research workflow for topic: {topic}",
            extra={"correlation_id": context.correlation_id},
        )

        try:
            # Stage 1: Research
            logger.info(
                "Stage: Research", extra={"correlation_id": context.correlation_id}
            )
            result.research = await self.researcher.research(topic, context)
            result.status = WorkflowStage.FACT_CHECK

            # Stage 2: Fact-Check
            logger.info(
                "Stage: Fact-Check", extra={"correlation_id": context.correlation_id}
            )
            result.fact_check = await self.fact_checker.verify_claims(
                claims=result.research.findings,
                sources=result.research.sources,
                context=context,
            )
            result.status = WorkflowStage.SYNTHESIS

            # Stage 3: Synthesis
            logger.info(
                "Stage: Synthesis", extra={"correlation_id": context.correlation_id}
            )
            result.synthesis = await self.synthesizer.synthesize(
                research=result.research,
                fact_check=result.fact_check,
                context=context,
            )
            result.status = WorkflowStage.WRITING

            # Stage 4: Writing
            logger.info(
                "Stage: Writing", extra={"correlation_id": context.correlation_id}
            )
            result.report = await self.writer.write_report(
                synthesis=result.synthesis,
                format="markdown",
                context=context,
            )
            result.status = WorkflowStage.REVIEW

            # Stage 5: Review (with iteration)
            for iteration in range(self.max_iterations):
                logger.info(
                    f"Review iteration {iteration + 1}",
                    extra={"correlation_id": context.correlation_id},
                )
                result.review = await self.critic.review(result.report, context)
                result.iterations = iteration + 1

                if result.review.approved:
                    logger.info(
                        "Report approved",
                        extra={"correlation_id": context.correlation_id},
                    )
                    break

                if result.review.score >= self.auto_approve_threshold:
                    logger.info(
                        f"Report approved (score: {result.review.score})",
                        extra={"correlation_id": context.correlation_id},
                    )
                    break

                # Revision needed - rewrite with feedback
                logger.info(
                    "Report requires revision",
                    extra={"correlation_id": context.correlation_id},
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
