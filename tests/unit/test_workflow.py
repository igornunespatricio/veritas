"""Unit tests for workflow orchestration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestWorkflowStage:
    """Tests for WorkflowStage enum."""

    def test_all_stages_defined(self):
        """Test that all workflow stages are defined."""
        from src.orchestration.workflow import WorkflowStage

        stages = [
            WorkflowStage.RESEARCH,
            WorkflowStage.FACT_CHECK,
            WorkflowStage.SYNTHESIS,
            WorkflowStage.WRITING,
            WorkflowStage.REVIEW,
            WorkflowStage.COMPLETED,
            WorkflowStage.FAILED,
        ]
        assert len(stages) == 7

    def test_stage_values(self):
        """Test stage enum values."""
        from src.orchestration.workflow import WorkflowStage

        assert WorkflowStage.RESEARCH.value == "research"
        assert WorkflowStage.FACT_CHECK.value == "fact_check"
        assert WorkflowStage.SYNTHESIS.value == "synthesis"
        assert WorkflowStage.WRITING.value == "writing"
        assert WorkflowStage.REVIEW.value == "review"
        assert WorkflowStage.COMPLETED.value == "completed"
        assert WorkflowStage.FAILED.value == "failed"


class TestWorkflowResult:
    """Tests for WorkflowResult class."""

    def test_default_initialization(self):
        """Test default WorkflowResult initialization."""
        from src.orchestration.workflow import WorkflowResult, WorkflowStage

        result = WorkflowResult(status=WorkflowStage.RESEARCH)

        assert result.status == WorkflowStage.RESEARCH
        assert result.research is None
        assert result.fact_check is None
        assert result.synthesis is None
        assert result.report is None
        assert result.review is None
        assert result.error is None
        assert result.iterations == 0

    def test_with_error(self):
        """Test WorkflowResult with error."""
        from src.orchestration.workflow import WorkflowResult, WorkflowStage

        result = WorkflowResult(
            status=WorkflowStage.FAILED,
            error="Something went wrong",
        )

        assert result.status == WorkflowStage.FAILED
        assert result.error == "Something went wrong"


class TestResearchWorkflow:
    """Tests for ResearchWorkflow class."""

    def test_workflow_initialization_defaults(self):
        """Test ResearchWorkflow with default values."""
        with patch("src.orchestration.workflow.ResearcherAgent"):
            with patch("src.orchestration.workflow.FactCheckerAgent"):
                with patch("src.orchestration.workflow.SynthesizerAgent"):
                    with patch("src.orchestration.workflow.WriterAgent"):
                        with patch("src.orchestration.workflow.CriticAgent"):
                            from src.orchestration.workflow import ResearchWorkflow

                            workflow = ResearchWorkflow()

                            assert workflow.max_iterations == 3
                            assert workflow.auto_approve_threshold == 0.8
                            assert workflow.llm_provider == "openai"
                            assert workflow.llm_model == "gpt-4o"

    def test_workflow_initialization_custom(self):
        """Test ResearchWorkflow with custom values."""
        with patch("src.orchestration.workflow.ResearcherAgent"):
            with patch("src.orchestration.workflow.FactCheckerAgent"):
                with patch("src.orchestration.workflow.SynthesizerAgent"):
                    with patch("src.orchestration.workflow.WriterAgent"):
                        with patch("src.orchestration.workflow.CriticAgent"):
                            from src.orchestration.workflow import ResearchWorkflow

                            workflow = ResearchWorkflow(
                                max_iterations=5,
                                auto_approve_threshold=0.9,
                                llm_provider="anthropic",
                                llm_model="claude-3-opus",
                            )

                            assert workflow.max_iterations == 5
                            assert workflow.auto_approve_threshold == 0.9
                            assert workflow.llm_provider == "anthropic"
                            assert workflow.llm_model == "claude-3-opus"


class TestWorkflowResultProperties:
    """Tests for WorkflowResult properties and methods."""

    def test_result_with_all_outputs(self):
        """Test WorkflowResult with all outputs populated."""
        from src.orchestration.workflow import WorkflowResult, WorkflowStage

        research = MagicMock()
        fact_check = MagicMock()
        synthesis = MagicMock()
        report = MagicMock()
        review = MagicMock()

        result = WorkflowResult(
            status=WorkflowStage.COMPLETED,
            research=research,
            fact_check=fact_check,
            synthesis=synthesis,
            report=report,
            review=review,
            iterations=2,
        )

        assert result.research is research
        assert result.fact_check is fact_check
        assert result.synthesis is synthesis
        assert result.report is report
        assert result.review is review
        assert result.iterations == 2

    def test_result_iteration_counting(self):
        """Test iteration counting in workflow result."""
        from src.orchestration.workflow import WorkflowResult, WorkflowStage

        result = WorkflowResult(status=WorkflowStage.REVIEW, iterations=3)

        assert result.iterations == 3
        assert result.iterations > 0

    def test_result_status_transitions(self):
        """Test that workflow can transition between statuses."""
        from src.orchestration.workflow import WorkflowResult, WorkflowStage

        result = WorkflowResult(status=WorkflowStage.RESEARCH)
        assert result.status == WorkflowStage.RESEARCH

        result.status = WorkflowStage.FACT_CHECK
        assert result.status == WorkflowStage.FACT_CHECK

        result.status = WorkflowStage.COMPLETED
        assert result.status == WorkflowStage.COMPLETED
