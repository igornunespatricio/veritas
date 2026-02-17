"""Unit tests for API route helper functions."""

import pytest

from src.api.routes.research import (
    _map_workflow_stage_to_progress,
)
from src.orchestration.workflow import WorkflowStage


class TestMapWorkflowStageToProgress:
    """Tests for _map_workflow_stage_to_progress helper function."""

    def test_research_stage(self):
        """Test mapping for RESEARCH stage."""
        stage, progress = _map_workflow_stage_to_progress(WorkflowStage.RESEARCH)
        assert stage == "research"
        assert progress == 20

    def test_fact_check_stage(self):
        """Test mapping for FACT_CHECK stage."""
        stage, progress = _map_workflow_stage_to_progress(WorkflowStage.FACT_CHECK)
        assert stage == "fact_check"
        assert progress == 40

    def test_synthesis_stage(self):
        """Test mapping for SYNTHESIS stage."""
        stage, progress = _map_workflow_stage_to_progress(WorkflowStage.SYNTHESIS)
        assert stage == "synthesis"
        assert progress == 60

    def test_writing_stage(self):
        """Test mapping for WRITING stage."""
        stage, progress = _map_workflow_stage_to_progress(WorkflowStage.WRITING)
        assert stage == "writing"
        assert progress == 80

    def test_review_stage(self):
        """Test mapping for REVIEW stage."""
        stage, progress = _map_workflow_stage_to_progress(WorkflowStage.REVIEW)
        assert stage == "review"
        assert progress == 90

    def test_completed_stage(self):
        """Test mapping for COMPLETED stage."""
        stage, progress = _map_workflow_stage_to_progress(WorkflowStage.COMPLETED)
        assert stage == "completed"
        assert progress == 100

    def test_failed_stage(self):
        """Test mapping for FAILED stage."""
        stage, progress = _map_workflow_stage_to_progress(WorkflowStage.FAILED)
        assert stage == "failed"
        assert progress == 100

    def test_unknown_stage(self):
        """Test mapping for unknown stage returns default values."""
        # Using a mock or invalid stage - if the enum handles it gracefully
        # This tests the fallback behavior
        stage, progress = _map_workflow_stage_to_progress(None)  # type: ignore
        assert stage == "unknown"
        assert progress == 0
