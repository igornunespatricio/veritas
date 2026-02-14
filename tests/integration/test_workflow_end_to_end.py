"""End-to-end workflow integration tests."""

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


class TestFullWorkflowExecution:
    """Test the complete multi-agent workflow execution."""

    @pytest.fixture
    def ollama_config(self):
        """Use Ollama local model for testing."""
        return {
            "llm_provider": "ollama",
            "llm_model": "llama3.2:3b",
            "max_iterations": 1,
            "auto_approve_threshold": 0.8,
        }

    @pytest.fixture
    def sample_research_completed(self):
        """Sample research completed event."""
        return ResearchCompleted.create(
            topic="quantum computing applications",
            sources=[
                {
                    "url": "https://example.com/quantum",
                    "title": "Quantum Computing Overview",
                    "date": "2024-01-01",
                }
            ],
            findings=[
                "Quantum computers use qubits instead of classical bits",
                "Quantum superposition allows qubits to represent multiple states",
                "Quantum entanglement enables instant communication between qubits",
            ],
        )

    @pytest.fixture
    def sample_fact_check_completed(self):
        """Sample fact-check completed event."""
        return FactCheckCompleted.create(
            claims=[
                {
                    "text": "Quantum computers use qubits instead of classical bits",
                    "status": "verified",
                },
                {
                    "text": "Quantum superposition allows qubits to represent multiple states",
                    "status": "verified",
                },
                {
                    "text": "Quantum entanglement enables instant communication between qubits",
                    "status": "partially_verified",
                },
            ],
            verified_claims=[
                {
                    "text": "Quantum computers use qubits instead of classical bits",
                    "status": "verified",
                }
            ],
            confidence_scores={
                "Quantum computers use qubits instead of classical bits": 0.95,
                "Quantum superposition allows qubits to represent multiple states": 0.90,
                "Quantum entanglement enables instant communication between qubits": 0.70,
            },
        )

    @pytest.fixture
    def sample_synthesis_completed(self):
        """Sample synthesis completed event."""
        return SynthesisCompleted.create(
            insights=[
                "Quantum computing represents a paradigm shift in computation",
                "Key applications include cryptography, drug discovery, and optimization",
                "Current hardware challenges include error correction and scalability",
            ],
            resolved_contradictions=[
                {
                    "contradiction": "Quantum entanglement and instant communication",
                    "resolution": "While entanglement exists, it cannot be used for faster-than-light communication",
                }
            ],
        )

    @pytest.fixture
    def sample_report_written(self):
        """Sample report written event."""
        return ReportWritten.create(
            title="Quantum Computing: A Comprehensive Overview",
            content="# Quantum Computing\n\n## Introduction\n\nQuantum computing represents...",
            format="markdown",
        )

    @pytest.fixture
    def sample_report_approved(self):
        """Sample report reviewed and approved."""
        return ReportReviewed.create(
            suggestions=["Consider adding more recent developments"],
            score=0.85,
            approved=True,
        )

    @pytest.mark.asyncio
    async def test_workflow_status_progression(self, ollama_config):
        """Test that workflow status progresses through all stages."""
        workflow = ResearchWorkflow(
            max_iterations=1,
            auto_approve_threshold=0.8,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
        )

        # Mock all agent methods to avoid real LLM calls
        with (
            patch.object(
                workflow.researcher, "research", new_callable=AsyncMock
            ) as mock_research,
            patch.object(
                workflow.fact_checker, "verify_claims", new_callable=AsyncMock
            ) as mock_factcheck,
            patch.object(
                workflow.synthesizer, "synthesize", new_callable=AsyncMock
            ) as mock_synthesize,
            patch.object(
                workflow.writer, "write_report", new_callable=AsyncMock
            ) as mock_write,
            patch.object(
                workflow.critic, "review", new_callable=AsyncMock
            ) as mock_review,
        ):

            # Setup mock returns
            mock_research.return_value = ResearchCompleted.create(
                topic="test topic",
                sources=[{"url": "", "title": "", "date": ""}],
                findings=["finding 1", "finding 2"],
            )
            mock_factcheck.return_value = FactCheckCompleted.create(
                claims=[{"text": "test", "status": "verified"}],
                verified_claims=[{"text": "test", "status": "verified"}],
                confidence_scores={"test": 0.9},
            )
            mock_synthesize.return_value = SynthesisCompleted.create(
                insights=["insight 1"],
                resolved_contradictions=[],
            )
            mock_write.return_value = ReportWritten.create(
                title="Test Report", content="Test content", format="markdown"
            )
            mock_review.return_value = ReportReviewed.create(
                suggestions=[], score=0.9, approved=True
            )

            # Execute workflow
            result = await workflow.execute("test topic")

            # Verify status progression
            assert result.status == WorkflowStage.COMPLETED
            assert result.research is not None
            assert result.fact_check is not None
            assert result.synthesis is not None
            assert result.report is not None
            assert result.review is not None
            assert result.error is None

    @pytest.mark.asyncio
    async def test_workflow_researcher_output_structure(self, ollama_config):
        """Test that researcher output has correct structure."""
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
            ) as mock_synthesize,
            patch.object(
                workflow.writer, "write_report", new_callable=AsyncMock
            ) as mock_write,
            patch.object(
                workflow.critic, "review", new_callable=AsyncMock
            ) as mock_review,
        ):

            mock_research.return_value = ResearchCompleted.create(
                topic="machine learning basics",
                sources=[
                    {
                        "url": "https://example.com/ml",
                        "title": "ML Guide",
                        "date": "2024-01-01",
                    }
                ],
                findings=[
                    "Machine learning is a subset of AI",
                    "Neural networks are inspired by biological brains",
                ],
            )
            mock_factcheck.return_value = FactCheckCompleted.create(
                claims=[{"text": "test", "status": "verified"}],
                verified_claims=[{"text": "test", "status": "verified"}],
                confidence_scores={"test": 0.9},
            )
            mock_synthesize.return_value = SynthesisCompleted.create(
                insights=["insight"],
                resolved_contradictions=[],
            )
            mock_write.return_value = ReportWritten.create(
                title="ML Report", content="ML content", format="markdown"
            )
            mock_review.return_value = ReportReviewed.create(
                suggestions=[], score=0.9, approved=True
            )

            result = await workflow.execute("machine learning basics")

            # Verify researcher output structure
            assert result.research.topic == "machine learning basics"
            assert len(result.research.sources) > 0
            assert len(result.research.findings) > 0
            assert "url" in result.research.sources[0]
            assert "title" in result.research.sources[0]

    @pytest.mark.asyncio
    async def test_workflow_accumulates_iterations(self, ollama_config):
        """Test that workflow tracks iteration count."""
        workflow = ResearchWorkflow(
            max_iterations=3,
            auto_approve_threshold=0.9,  # High threshold to trigger iterations
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
            ) as mock_synthesize,
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
                verified_claims=[{"text": "test", "status": "verified"}],
                confidence_scores={"test": 0.9},
            )
            mock_synthesize.return_value = SynthesisCompleted.create(
                insights=["insight"],
                resolved_contradictions=[],
            )
            mock_write.return_value = ReportWritten.create(
                title="Report", content="Content", format="markdown"
            )

            # Critic rejects first two times, approves on third
            call_count = 0

            async def mock_review_func(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    return ReportReviewed.create(
                        suggestions=["Improve clarity"],
                        score=0.6,
                        approved=False,
                    )
                return ReportReviewed.create(
                    suggestions=["Good enough"],
                    score=0.92,
                    approved=True,
                )

            mock_review.side_effect = mock_review_func

            result = await workflow.execute("test topic")

            assert result.iterations == 3
            assert result.status == WorkflowStage.COMPLETED

    @pytest.mark.asyncio
    async def test_sequential_workflow_execution(self, ollama_config):
        """Test sequential workflow without critic iterations."""
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
            ) as mock_synthesize,
            patch.object(
                workflow.writer, "write_report", new_callable=AsyncMock
            ) as mock_write,
        ):

            mock_research.return_value = ResearchCompleted.create(
                topic="climate change",
                sources=[{"url": "", "title": "", "date": ""}],
                findings=["finding 1", "finding 2"],
            )
            mock_factcheck.return_value = FactCheckCompleted.create(
                claims=[{"text": "test", "status": "verified"}],
                verified_claims=[{"text": "test", "status": "verified"}],
                confidence_scores={"test": 0.9},
            )
            mock_synthesize.return_value = SynthesisCompleted.create(
                insights=["insight"],
                resolved_contradictions=[],
            )
            mock_write.return_value = ReportWritten.create(
                title="Climate Report", content="Climate content", format="markdown"
            )

            result = await workflow.execute_sequential("climate change")

            assert result.status == WorkflowStage.COMPLETED
            assert result.research is not None
            assert result.report is not None
            assert result.review is None  # Sequential skips critic

    @pytest.mark.asyncio
    async def test_workflow_with_auto_approval(self, ollama_config):
        """Test workflow auto-approval when score exceeds threshold."""
        workflow = ResearchWorkflow(
            max_iterations=3,
            auto_approve_threshold=0.5,  # Low threshold
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
            ) as mock_synthesize,
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
                verified_claims=[{"text": "test", "status": "verified"}],
                confidence_scores={"test": 0.9},
            )
            mock_synthesize.return_value = SynthesisCompleted.create(
                insights=["insight"],
                resolved_contradictions=[],
            )
            mock_write.return_value = ReportWritten.create(
                title="Report", content="Content", format="markdown"
            )

            # Score below approval but above auto-approve threshold
            mock_review.return_value = ReportReviewed.create(
                suggestions=["Minor suggestions"],
                score=0.6,  # Above 0.5 auto-approve threshold
                approved=False,  # But not explicitly approved
            )

            result = await workflow.execute("test topic")

            # Should auto-approve because score >= threshold
            assert result.status == WorkflowStage.COMPLETED
            assert result.iterations == 1
