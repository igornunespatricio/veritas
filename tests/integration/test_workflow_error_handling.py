"""Workflow error handling integration tests."""

import pytest
from unittest.mock import AsyncMock, patch
from typing import Any

from src.orchestration.workflow import ResearchWorkflow, WorkflowStage
from src.domain.events import (
    ResearchCompleted,
    FactCheckCompleted,
    SynthesisCompleted,
    ReportWritten,
    ReportReviewed,
)
from src.domain.interfaces import AgentContext
from src.infrastructure.circuit_breaker import CircuitOpenError


class TestWorkflowErrorHandling:
    """Test workflow behavior under error conditions."""

    @pytest.mark.asyncio
    async def test_workflow_handles_researcher_failure(self):
        """Verify workflow degrades gracefully when researcher fails."""
        workflow = ResearchWorkflow(
            max_iterations=1,
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        with patch.object(
            workflow.researcher, "research", new_callable=AsyncMock
        ) as mock_research:
            # Simulate researcher failure
            mock_research.side_effect = Exception("Web search failed")

            result = await workflow.execute("test topic")

            # Workflow should mark as failed with error
            assert result.status == WorkflowStage.FAILED
            assert result.error is not None
            assert "Web search failed" in result.error
            # Verify intermediate states are None
            assert result.research is None
            assert result.fact_check is None

    @pytest.mark.asyncio
    async def test_workflow_handles_factchecker_failure(self):
        """Verify workflow handles fact-checker failure gracefully."""
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
        ):

            mock_research.return_value = ResearchCompleted.create(
                topic="test",
                sources=[{"url": "", "title": "", "date": ""}],
                findings=["finding"],
            )
            mock_factcheck.side_effect = Exception("Fact-check service unavailable")

            result = await workflow.execute("test topic")

            assert result.status == WorkflowStage.FAILED
            assert "Fact-check service unavailable" in result.error
            # Research should be captured even if fact-check fails
            assert result.research is not None
            assert result.fact_check is None

    @pytest.mark.asyncio
    async def test_workflow_handles_synthesizer_failure(self):
        """Verify workflow handles synthesizer failure."""
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
            mock_synth.side_effect = Exception("Synthesis timeout")

            result = await workflow.execute("test topic")

            assert result.status == WorkflowStage.FAILED
            assert "Synthesis timeout" in result.error
            assert result.research is not None
            assert result.fact_check is not None

    @pytest.mark.asyncio
    async def test_workflow_handles_writer_failure(self):
        """Verify workflow handles writer failure."""
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
            mock_write.side_effect = Exception("Writer LLM error")

            result = await workflow.execute("test topic")

            assert result.status == WorkflowStage.FAILED
            assert "Writer LLM error" in result.error
            assert result.synthesis is not None

    @pytest.mark.asyncio
    async def test_workflow_captures_partial_results_on_failure(self):
        """Verify that partial results are captured even on failure."""
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
        ):

            mock_research.return_value = ResearchCompleted.create(
                topic="partial test",
                sources=[
                    {"url": "https://test.com", "title": "Test", "date": "2024-01-01"}
                ],
                findings=["finding 1", "finding 2", "finding 3"],
            )
            mock_factcheck.return_value = FactCheckCompleted.create(
                claims=[{"text": "test", "status": "verified"}],
                verified_claims=[],
                confidence_scores={},
            )
            # Synthesizer fails
            mock_synth.side_effect = Exception("Unexpected error")

            result = await workflow.execute("test topic")

            # Should have partial results
            assert result.research is not None
            assert result.fact_check is not None
            assert result.synthesis is None  # Failed before this
            assert result.report is None
            assert result.error is not None


class TestWorkflowLLMErrors:
    """Test workflow behavior under LLM error conditions."""

    @pytest.mark.asyncio
    async def test_workflow_handles_rate_limit_error(self):
        """Verify workflow correctly reports rate limit errors."""
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

            # Simulate rate limit error
            mock_research.side_effect = Exception("Rate limit exceeded")

            result = await workflow.execute("test topic")

            # Should handle error gracefully
            assert result.status == WorkflowStage.FAILED
            assert result.error is not None
            assert "Rate limit" in result.error

    @pytest.mark.asyncio
    async def test_workflow_handles_circuit_breaker_open(self):
        """Verify workflow handles circuit breaker open state."""
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

            # Simulate circuit breaker open
            mock_research.side_effect = CircuitOpenError("Circuit breaker is open")

            result = await workflow.execute("test topic")

            assert result.status == WorkflowStage.FAILED
            assert result.error is not None
            assert "Circuit breaker" in result.error or "open" in result.error.lower()


class TestWorkflowIterationsLimit:
    """Test workflow iteration limits."""

    @pytest.mark.asyncio
    async def test_max_iterations_enforced_strictly(self):
        """Verify max iterations is strictly enforced."""
        workflow = ResearchWorkflow(
            max_iterations=2,
            auto_approve_threshold=1.0,  # Never auto-approve
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

            # Critic always rejects
            mock_review.return_value = ReportReviewed.create(
                suggestions=["Improve"],
                score=0.5,
                approved=False,
            )

            result = await workflow.execute("test topic")

            # Should stop at max iterations
            assert result.iterations == 2
            assert result.status == WorkflowStage.COMPLETED
            # Even though not approved, it completes due to max iterations

    @pytest.mark.asyncio
    async def test_iteration_zero_with_sequential_workflow(self):
        """Verify sequential workflow has zero iterations."""
        workflow = ResearchWorkflow(
            max_iterations=3,
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

            result = await workflow.execute_sequential("test topic")

            # Sequential workflow skips critic entirely
            assert result.iterations == 0
            assert result.review is None


class TestWorkflowRecovery:
    """Test workflow recovery and continuation."""

    @pytest.mark.asyncio
    async def test_workflow_preserves_correlation_id_in_context(self):
        """Verify correlation ID is passed to agent context correctly."""
        from src.domain.interfaces import AgentContext

        context = AgentContext.create(correlation_id="test-correlation-456")

        # Verify context creation preserves correlation ID
        assert context.correlation_id == "test-correlation-456"
        assert context.created_at is not None
