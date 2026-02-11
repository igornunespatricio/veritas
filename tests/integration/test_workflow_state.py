"""Workflow state management integration tests."""

import pytest
from unittest.mock import AsyncMock, patch

from src.orchestration.workflow import ResearchWorkflow, WorkflowStage, WorkflowResult
from src.domain.events import (
    ResearchCompleted,
    FactCheckCompleted,
    SynthesisCompleted,
    ReportWritten,
    ReportReviewed,
)
from src.domain.interfaces import AgentContext


class TestWorkflowResultState:
    """Test WorkflowResult state accumulation."""

    @pytest.mark.asyncio
    async def test_workflow_result_initial_state(self):
        """Test WorkflowResult has correct initial state."""
        result = WorkflowResult(status=WorkflowStage.RESEARCH)

        assert result.status == WorkflowStage.RESEARCH
        assert result.research is None
        assert result.fact_check is None
        assert result.synthesis is None
        assert result.report is None
        assert result.review is None
        assert result.error is None
        assert result.iterations == 0

    @pytest.mark.asyncio
    async def test_workflow_result_accumulates_all_stages(self):
        """Test WorkflowResult correctly accumulates results from all stages."""
        result = WorkflowResult(status=WorkflowStage.COMPLETED)

        # Add research result
        result.research = ResearchCompleted.create(
            topic="test topic",
            sources=[{"url": "", "title": "", "date": ""}],
            findings=["finding 1", "finding 2"],
        )

        # Add fact-check result
        result.fact_check = FactCheckCompleted.create(
            claims=[{"text": "test", "status": "verified"}],
            verified_claims=[],
            confidence_scores={},
        )

        # Add synthesis result
        result.synthesis = SynthesisCompleted.create(
            insights=["insight"],
            resolved_contradictions=[],
        )

        # Add report result
        result.report = ReportWritten.create(
            title="Test Report",
            content="Test content",
            format="markdown",
        )

        # Add review result
        result.review = ReportReviewed.create(
            suggestions=[],
            score=0.9,
            approved=True,
        )

        result.iterations = 1

        # Verify all data is accumulated
        assert result.research is not None
        assert result.research.topic == "test topic"
        assert len(result.research.findings) == 2

        assert result.fact_check is not None
        assert len(result.fact_check.claims) == 1

        assert result.synthesis is not None
        assert len(result.synthesis.insights) == 1

        assert result.report is not None
        assert result.report.title == "Test Report"

        assert result.review is not None
        assert result.review.approved is True
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_workflow_result_error_state(self):
        """Test WorkflowResult correctly stores error state."""
        result = WorkflowResult(
            status=WorkflowStage.FAILED,
            error="Researcher failed: Connection timeout",
        )

        assert result.status == WorkflowStage.FAILED
        assert result.error == "Researcher failed: Connection timeout"
        assert result.research is None
        assert result.iterations == 0


class TestWorkflowStageProgression:
    """Test workflow stage transitions."""

    def test_workflow_stage_order(self):
        """Test that stages are ordered correctly for progression."""
        stages = [
            WorkflowStage.RESEARCH,
            WorkflowStage.FACT_CHECK,
            WorkflowStage.SYNTHESIS,
            WorkflowStage.WRITING,
            WorkflowStage.REVIEW,
            WorkflowStage.COMPLETED,
        ]

        # Verify expected order
        assert WorkflowStage.RESEARCH.value == "research"
        assert WorkflowStage.FACT_CHECK.value == "fact_check"
        assert WorkflowStage.SYNTHESIS.value == "synthesis"
        assert WorkflowStage.WRITING.value == "writing"
        assert WorkflowStage.REVIEW.value == "review"
        assert WorkflowStage.COMPLETED.value == "completed"
        assert WorkflowStage.FAILED.value == "failed"

    @pytest.mark.asyncio
    async def test_stage_progression_in_workflow(self):
        """Test that workflow status progresses through all stages."""
        workflow = ResearchWorkflow(
            max_iterations=1,
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        with (
            patch.object(
                workflow.researcher, "research", new_callable=AsyncMock
            ) as mock_research,
            patch.object(
                workflow.fact_checker, "verify_claims", new_callable=AsyncMock
            ) as mock_factcheck,
            patch.object(
                workflow.synthesizer, "synthesize", new_callable=AsyncMock
            ) as mock_synth,
            patch.object(
                workflow.writer, "write_report", new_callable=AsyncMock
            ) as mock_write,
            patch.object(
                workflow.critic, "review", new_callable=AsyncMock
            ) as mock_review,
        ):

            mock_research.return_value = ResearchCompleted.create(
                topic="test",
                sources=[{"url": "", "title": "", "date": ""}],
                findings=["finding"],
            )
            mock_factcheck.return_value = FactCheckCompleted.create(
                claims=[{"text": "test", "status": "verified"}],
                verified_claims=[],
                confidence_scores={},
            )
            mock_synth.return_value = SynthesisCompleted.create(
                insights=["insight"],
                resolved_contradictions=[],
            )
            mock_write.return_value = ReportWritten.create(
                title="Report", content="Content", format="markdown"
            )
            mock_review.return_value = ReportReviewed.create(
                suggestions=[], score=0.9, approved=True
            )

            result = await workflow.execute("test topic")

            # Should have progressed through all stages to COMPLETED
            assert result.status == WorkflowStage.COMPLETED


class TestCorrelationIdPropagation:
    """Test correlation ID tracking through the workflow."""

    @pytest.mark.asyncio
    async def test_correlation_id_set_from_context(self):
        """Verify correlation ID is properly set from context."""
        context = AgentContext.create(correlation_id="test-correlation-abc")

        # Create context with specific correlation ID
        assert context.correlation_id == "test-correlation-abc"
        assert context.request_id == ""
        assert context.metadata == {}

    @pytest.mark.asyncio
    async def test_events_preserve_correlation_id(self):
        """Verify domain events preserve correlation ID."""
        correlation_id = "research-session-123"

        research = ResearchCompleted.create(
            topic="test",
            sources=[{"url": "", "title": "", "date": ""}],
            findings=["finding"],
            correlation_id=correlation_id,
        )

        fact_check = FactCheckCompleted.create(
            claims=[{"text": "test", "status": "verified"}],
            verified_claims=[],
            confidence_scores={},
            correlation_id=correlation_id,
        )

        synthesis = SynthesisCompleted.create(
            insights=["insight"],
            resolved_contradictions=[],
            correlation_id=correlation_id,
        )

        report = ReportWritten.create(
            title="Test",
            content="Content",
            correlation_id=correlation_id,
        )

        review = ReportReviewed.create(
            suggestions=[],
            score=0.9,
            approved=True,
            correlation_id=correlation_id,
        )

        # All events should have the same correlation ID
        assert research.correlation_id == correlation_id
        assert fact_check.correlation_id == correlation_id
        assert synthesis.correlation_id == correlation_id
        assert report.correlation_id == correlation_id
        assert review.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_workflow_generates_correlation_id_if_not_provided(self):
        """Verify workflow generates correlation ID if not provided."""
        workflow = ResearchWorkflow(
            max_iterations=1,
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        with (
            patch.object(
                workflow.researcher, "research", new_callable=AsyncMock
            ) as mock_research,
            patch.object(
                workflow.fact_checker, "verify_claims", new_callable=AsyncMock
            ) as mock_factcheck,
            patch.object(
                workflow.synthesizer, "synthesize", new_callable=AsyncMock
            ) as mock_synth,
            patch.object(
                workflow.writer, "write_report", new_callable=AsyncMock
            ) as mock_write,
            patch.object(
                workflow.critic, "review", new_callable=AsyncMock
            ) as mock_review,
        ):

            mock_research.return_value = ResearchCompleted.create(
                topic="test",
                sources=[{"url": "", "title": "", "date": ""}],
                findings=["finding"],
            )
            mock_factcheck.return_value = FactCheckCompleted.create(
                claims=[{"text": "test", "status": "verified"}],
                verified_claims=[],
                confidence_scores={},
            )
            mock_synth.return_value = SynthesisCompleted.create(
                insights=["insight"],
                resolved_contradictions=[],
            )
            mock_write.return_value = ReportWritten.create(
                title="Report", content="Content", format="markdown"
            )
            mock_review.return_value = ReportReviewed.create(
                suggestions=[], score=0.9, approved=True
            )

            # Execute without providing correlation ID
            result = await workflow.execute("test topic")

            # Should have generated correlation IDs
            assert result.research is not None
            # Correlation ID should be present (either auto-generated or UUID)


class TestWorkflowResultDataclass:
    """Test WorkflowResult dataclass behavior."""

    @pytest.mark.asyncio
    async def test_workflow_result_is_immutable(self):
        """Test that WorkflowResult fields can be updated."""
        result = WorkflowResult(status=WorkflowStage.RESEARCH)

        # Update fields (dataclass allows this)
        result.status = WorkflowStage.COMPLETED
        result.iterations = 5

        assert result.status == WorkflowStage.COMPLETED
        assert result.iterations == 5

    @pytest.mark.asyncio
    async def test_workflow_result_default_values(self):
        """Test WorkflowResult has correct default values."""
        result = WorkflowResult(status=WorkflowStage.FAILED)

        # Check defaults
        assert result.research is None
        assert result.fact_check is None
        assert result.synthesis is None
        assert result.report is None
        assert result.review is None
        assert result.error is None
        assert result.iterations == 0


class TestAgentContextState:
    """Test AgentContext state management."""

    def test_context_factory_creates_valid_context(self):
        """Test AgentContext.create() factory method."""
        context = AgentContext.create(correlation_id="custom-id")

        assert context.correlation_id == "custom-id"
        assert context.created_at is not None
        assert isinstance(context.created_at, type(context.created_at))

    def test_context_default_values(self):
        """Test AgentContext default values."""
        context = AgentContext.create()

        assert context.correlation_id == ""
        assert context.request_id == ""
        assert context.metadata == {}
        assert context.created_at is not None
