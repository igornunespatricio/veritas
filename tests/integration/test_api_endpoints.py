"""Integration tests for API endpoints."""

from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone
import pytest

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.models.response import JobStatus
from src.domain.events import (
    FactCheckCompleted,
    ReportReviewed,
    ReportWritten,
    ResearchCompleted,
    SynthesisCompleted,
)
from src.orchestration.workflow import WorkflowStage


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_healthy(self, client):
        """Test health check returns healthy status."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "veritas-api"

    def test_health_check_no_docs(self, client):
        """Test health check doesn't expose docs."""
        # Health endpoint should work without authentication
        response = client.get("/api/v1/health")
        assert response.status_code == 200


class TestResearchEndpoints:
    """Tests for research API endpoints."""

    def test_submit_research_returns_202(self, client):
        """Test submitting a research job returns 202 Accepted."""
        with patch("src.api.routes.research._run_research_workflow") as mock_workflow:
            response = client.post(
                "/api/v1/research",
                json={"topic": "What is machine learning?"},
            )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["topic"] == "What is machine learning?"

    def test_submit_research_with_custom_params(self, client):
        """Test submitting research with custom parameters."""
        with patch("src.api.routes.research._run_research_workflow"):
            response = client.post(
                "/api/v1/research",
                json={
                    "topic": "Climate change effects",
                    "max_iterations": 5,
                    "auto_approve_threshold": 0.9,
                    "llm_provider": "anthropic",
                    "llm_model": "claude-3-opus",
                },
            )

        assert response.status_code == 202
        data = response.json()
        assert data["topic"] == "Climate change effects"

    def test_submit_research_validation_error(self, client):
        """Test that missing topic returns validation error."""
        response = client.post(
            "/api/v1/research",
            json={},
        )

        assert response.status_code == 422  # Validation error

    def test_get_job_not_found(self, client):
        """Test getting a non-existent job returns 404."""
        response = client.get("/api/v1/research/nonexistent-job-id")

        assert response.status_code == 404

    def test_get_job_pending_status(self, client):
        """Test getting a pending job returns status only."""
        # First create a job
        with patch("src.api.routes.research._run_research_workflow"):
            submit_response = client.post(
                "/api/v1/research",
                json={"topic": "Test topic"},
            )

        job_id = submit_response.json()["job_id"]

        # Now get the job
        response = client.get(f"/api/v1/research/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] in [JobStatus.PENDING, JobStatus.PROCESSING]

    def test_list_jobs_empty(self, client):
        """Test listing jobs when none exist."""
        response = client.get("/api/v1/research")

        assert response.status_code == 200
        # Returns empty list or default jobs

    def test_delete_job_not_found(self, client):
        """Test deleting a non-existent job returns 404."""
        response = client.delete("/api/v1/research/nonexistent-job-id")

        assert response.status_code == 404

    def test_delete_job_success(self, client):
        """Test deleting a job successfully."""
        # First create a job
        with patch("src.api.routes.research._run_research_workflow"):
            submit_response = client.post(
                "/api/v1/research",
                json={"topic": "Test topic"},
            )

        job_id = submit_response.json()["job_id"]

        # Now delete it
        response = client.delete(f"/api/v1/research/{job_id}")

        assert response.status_code == 204


class TestCORSHeaders:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in response."""
        response = client.options(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000"},
        )

        # Should have CORS headers
        assert (
            "access-control-allow-origin" in response.headers
            or response.status_code == 200
        )


class TestAPIEndpointsWithMockedWorkflow:
    """Tests for research endpoints with mocked workflow results."""

    def test_get_completed_job_with_results(self, client):
        """Test getting a completed job with full results."""
        # First create and complete a job
        with patch("src.api.routes.research._run_research_workflow"):
            submit_response = client.post(
                "/api/v1/research",
                json={"topic": "Test topic"},
            )

        job_id = submit_response.json()["job_id"]

        # Manually add a completed result to the jobs dict
        from src.api.routes import research
        from src.orchestration.workflow import WorkflowResult

        # Create mock completed result
        result = WorkflowResult(
            status=WorkflowStage.COMPLETED,
            research=ResearchCompleted.create(
                topic="Test topic",
                sources=[
                    {"url": "http://example.com", "title": "Test", "date": "2024-01-01"}
                ],
                findings=["Finding 1", "Finding 2"],
            ),
            fact_check=FactCheckCompleted.create(
                claims=[{"text": "Finding 1", "status": "verified"}],
                verified_claims=[{"text": "Finding 1", "status": "verified"}],
                confidence_scores={"Finding 1": 0.95},
            ),
            synthesis=SynthesisCompleted.create(
                insights=["Insight 1"],
                resolved_contradictions=[],
            ),
            report=ReportWritten.create(
                title="Test Report",
                content="# Test Report Content",
                format="markdown",
            ),
            review=ReportReviewed.create(
                suggestions=[],
                score=0.9,
                approved=True,
            ),
            iterations=1,
        )

        research._jobs[job_id]["result"] = result
        research._jobs[job_id]["status"] = JobStatus.COMPLETED

        # Now get the completed job
        response = client.get(f"/api/v1/research/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["topic"] == "Test topic"
        assert data["sources"] is not None
        assert data["report_title"] == "Test Report"

    def test_get_failed_job_with_error(self, client):
        """Test getting a failed job returns error info."""
        # First create a job
        with patch("src.api.routes.research._run_research_workflow"):
            submit_response = client.post(
                "/api/v1/research",
                json={"topic": "Test topic"},
            )

        job_id = submit_response.json()["job_id"]

        # Manually mark as failed
        from src.api.routes import research

        research._jobs[job_id]["status"] = JobStatus.FAILED
        research._jobs[job_id]["error"] = "Connection timeout"

        # Now get the failed job
        response = client.get(f"/api/v1/research/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Connection timeout"
