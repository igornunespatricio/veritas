"""Workflow configuration integration tests."""

from unittest.mock import AsyncMock, patch

import pytest

from src.domain.events import (
    FactCheckCompleted,
    ReportReviewed,
    ReportWritten,
    ResearchCompleted,
    SynthesisCompleted,
)
from src.orchestration.workflow import ResearchWorkflow, WorkflowStage


class TestWorkflowCustomConfiguration:
    """Test workflow with custom configuration parameters."""

    @pytest.mark.asyncio
    async def test_workflow_with_custom_max_iterations(self):
        """Test workflow respects custom max_iterations setting."""
        # Create workflow with 5 iterations
        workflow = ResearchWorkflow(
            max_iterations=5,
            auto_approve_threshold=1.0,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        assert workflow.max_iterations == 5
        assert workflow.auto_approve_threshold == 1.0

    @pytest.mark.asyncio
    async def test_workflow_with_custom_approval_threshold(self):
        """Test workflow respects custom auto_approve_threshold."""
        workflow = ResearchWorkflow(
            max_iterations=3,
            auto_approve_threshold=0.7,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        assert workflow.auto_approve_threshold == 0.7

    @pytest.mark.asyncio
    async def test_workflow_with_zero_iterations(self):
        """Test workflow with max_iterations=0."""
        workflow = ResearchWorkflow(
            max_iterations=0,
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        assert workflow.max_iterations == 0

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

            assert result.status == WorkflowStage.COMPLETED


class TestWorkflowAgentConfiguration:
    """Test workflow agent initialization."""

    @pytest.mark.asyncio
    async def test_workflow_initializes_all_agents(self):
        """Verify all agents are initialized in workflow."""
        workflow = ResearchWorkflow(
            max_iterations=1,
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        # Verify all agents exist
        assert workflow.researcher is not None
        assert workflow.fact_checker is not None
        assert workflow.synthesizer is not None
        assert workflow.writer is not None
        assert workflow.critic is not None

    @pytest.mark.asyncio
    async def test_workflow_agents_have_correct_names(self):
        """Verify agents have expected names."""
        workflow = ResearchWorkflow(
            max_iterations=1,
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        assert workflow.researcher.name == "researcher"
        assert workflow.fact_checker.name == "fact_checker"
        assert workflow.synthesizer.name == "synthesizer"
        assert workflow.writer.name == "writer"
        assert workflow.critic.name == "critic"

    @pytest.mark.asyncio
    async def test_workflow_agents_have_descriptions(self):
        """Verify agents have descriptions."""
        workflow = ResearchWorkflow(
            max_iterations=1,
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        assert len(workflow.researcher.description) > 0
        assert len(workflow.fact_checker.description) > 0
        assert len(workflow.synthesizer.description) > 0
        assert len(workflow.writer.description) > 0
        assert len(workflow.critic.description) > 0


class TestWorkflowModelConfiguration:
    """Test workflow LLM model configuration."""

    @pytest.mark.asyncio
    async def test_workflow_stores_model_name(self):
        """Verify workflow stores the LLM model name."""
        workflow = ResearchWorkflow(
            max_iterations=1,
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        assert workflow.llm_model == "llama3.2:3b"

    @pytest.mark.asyncio
    async def test_workflow_stores_provider_name(self):
        """Verify workflow stores the LLM provider name."""
        workflow = ResearchWorkflow(
            max_iterations=1,
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        assert workflow.llm_provider == "ollama"


class TestWorkflowExecutionValidation:
    """Test workflow execution with various inputs."""

    @pytest.mark.asyncio
    async def test_workflow_accepts_simple_topic(self):
        """Test workflow accepts a simple string topic."""
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

            assert result.status == WorkflowStage.COMPLETED

    @pytest.mark.asyncio
    async def test_workflow_accepts_long_topic(self):
        """Test workflow accepts a longer research topic."""
        workflow = ResearchWorkflow(
            max_iterations=1,
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        long_topic = "What are the latest developments in renewable energy technology, particularly focusing on solar panel efficiency improvements and energy storage solutions in 2024?"

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
                topic=long_topic,
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

            result = await workflow.execute(long_topic)

            assert result.status == WorkflowStage.COMPLETED
            assert result.research.topic == long_topic

    @pytest.mark.asyncio
    async def test_workflow_accepts_optional_correlation_id(self):
        """Test workflow accepts optional correlation_id parameter."""
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

            # Test with custom correlation ID
            result = await workflow.execute(
                "test topic", correlation_id="custom-correlation-123"
            )

            assert result.status == WorkflowStage.COMPLETED

            # Test without correlation ID
            result2 = await workflow.execute("test topic")
            assert result2.status == WorkflowStage.COMPLETED


class TestWorkflowDefaultValues:
    """Test workflow default configuration values."""

    def test_workflow_default_max_iterations(self):
        """Verify default max_iterations is 3."""
        workflow = ResearchWorkflow(
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        assert workflow.max_iterations == 3

    def test_workflow_default_approval_threshold(self):
        """Verify default auto_approve_threshold is 0.8."""
        workflow = ResearchWorkflow(
            max_iterations=3,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        assert workflow.auto_approve_threshold == 0.8

    def test_workflow_default_provider(self):
        """Verify default LLM provider is openai."""
        workflow = ResearchWorkflow(
            max_iterations=3,
            auto_approve_threshold=0.8,
        )

        assert workflow.llm_provider == "openai"

    def test_workflow_default_model(self):
        """Verify default LLM model is gpt-4o."""
        workflow = ResearchWorkflow(
            max_iterations=3,
            auto_approve_threshold=0.8,
            llm_provider="openai",
        )

        assert workflow.llm_model == "gpt-4o"
