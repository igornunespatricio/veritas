"""Unit tests for API models."""

import pytest
from pydantic import ValidationError

from src.api.models.request import ResearchRequest
from src.api.models.response import (
    FactCheckClaim,
    JobStatus,
    ResearchJobResponse,
    ResearchSource,
    ResearchStatusResponse,
)


class TestResearchRequest:
    """Tests for ResearchRequest model."""

    def test_valid_request_creation(self):
        """Test creating a valid research request."""
        request = ResearchRequest(topic="What is AI?")

        assert request.topic == "What is AI?"
        assert request.max_iterations == 3  # default
        assert request.auto_approve_threshold == 0.8  # default
        assert request.llm_provider == "openai"  # default
        assert request.llm_model == "gpt-4o"  # default

    def test_custom_values(self):
        """Test creating request with custom values."""
        request = ResearchRequest(
            topic="Climate change effects",
            max_iterations=5,
            auto_approve_threshold=0.9,
            llm_provider="anthropic",
            llm_model="claude-3-opus",
            max_tokens=5000,
        )

        assert request.topic == "Climate change effects"
        assert request.max_iterations == 5
        assert request.auto_approve_threshold == 0.9
        assert request.llm_provider == "anthropic"
        assert request.llm_model == "claude-3-opus"
        assert request.max_tokens == 5000

    def test_topic_empty_string_allowed(self):
        """Test that empty string topic is allowed by the model."""
        # Note: Pydantic doesn't validate empty strings by default
        # This is a design choice - empty strings are valid input
        request = ResearchRequest(topic="")
        assert request.topic == ""

    def test_max_iterations_bounds(self):
        """Test max_iterations must be between 1 and 10."""
        # Valid bounds
        request = ResearchRequest(topic="test", max_iterations=1)
        assert request.max_iterations == 1

        request = ResearchRequest(topic="test", max_iterations=10)
        assert request.max_iterations == 10

        # Invalid - too low
        with pytest.raises(ValidationError):
            ResearchRequest(topic="test", max_iterations=0)

        # Invalid - too high
        with pytest.raises(ValidationError):
            ResearchRequest(topic="test", max_iterations=11)

    def test_auto_approve_threshold_bounds(self):
        """Test auto_approve_threshold must be between 0.0 and 1.0."""
        # Valid bounds
        request = ResearchRequest(topic="test", auto_approve_threshold=0.0)
        assert request.auto_approve_threshold == 0.0

        request = ResearchRequest(topic="test", auto_approve_threshold=1.0)
        assert request.auto_approve_threshold == 1.0

        # Invalid - too low
        with pytest.raises(ValidationError):
            ResearchRequest(topic="test", auto_approve_threshold=-0.1)

        # Invalid - too high
        with pytest.raises(ValidationError):
            ResearchRequest(topic="test", auto_approve_threshold=1.1)

    def test_max_tokens_bounds(self):
        """Test max_tokens must be between 100 and 128000."""
        # Valid bounds
        request = ResearchRequest(topic="test", max_tokens=100)
        assert request.max_tokens == 100

        request = ResearchRequest(topic="test", max_tokens=128000)
        assert request.max_tokens == 128000

        # Invalid - too low
        with pytest.raises(ValidationError):
            ResearchRequest(topic="test", max_tokens=99)

    def test_max_tokens_default_none(self):
        """Test max_tokens defaults to None."""
        request = ResearchRequest(topic="test")
        assert request.max_tokens is None

    def test_llm_provider_choices(self):
        """Test valid LLM provider choices."""
        valid_providers = ["openai", "anthropic", "ollama", "openrouter"]

        for provider in valid_providers:
            request = ResearchRequest(topic="test", llm_provider=provider)
            assert request.llm_provider == provider

    def test_model_dump(self):
        """Test model_dump produces expected dict."""
        request = ResearchRequest(
            topic="test",
            max_iterations=2,
            auto_approve_threshold=0.7,
        )

        data = request.model_dump()

        assert data["topic"] == "test"
        assert data["max_iterations"] == 2
        assert data["auto_approve_threshold"] == 0.7


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_all_status_values(self):
        """Test all job status values are defined."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"


class TestResearchSource:
    """Tests for ResearchSource model."""

    def test_valid_source(self):
        """Test creating a valid research source."""
        source = ResearchSource(
            title="Example Article",
            url="https://example.com/article",
        )

        assert source.title == "Example Article"
        assert source.url == "https://example.com/article"

    def test_source_title_required(self):
        """Test that title is required."""
        with pytest.raises(ValidationError):
            ResearchSource(url="https://example.com")

    def test_source_url_required(self):
        """Test that url is required."""
        with pytest.raises(ValidationError):
            ResearchSource(title="Example")


class TestFactCheckClaim:
    """Tests for FactCheckClaim model."""

    def test_valid_claim(self):
        """Test creating a valid fact-check claim."""
        claim = FactCheckClaim(
            claim="The sky is blue",
            status="verified",
            confidence=0.95,
            notes="Confirmed by scientific sources",
        )

        assert claim.claim == "The sky is blue"
        assert claim.status == "verified"
        assert claim.confidence == 0.95
        assert claim.notes == "Confirmed by scientific sources"

    def test_claim_required_fields(self):
        """Test required fields for claim."""
        claim = FactCheckClaim(claim="Test claim", status="verified")

        assert claim.claim == "Test claim"
        assert claim.status == "verified"
        assert claim.confidence is None
        assert claim.notes is None


class TestResearchStatusResponse:
    """Tests for ResearchStatusResponse model."""

    def test_valid_status_response(self):
        """Test creating a valid status response."""
        from datetime import datetime, timezone

        response = ResearchStatusResponse(
            job_id="123e4567-e89b-12d3-a456-426614174000",
            status=JobStatus.PENDING,
            topic="What is machine learning?",
            created_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
        )

        assert response.job_id == "123e4567-e89b-12d3-a456-426614174000"
        assert response.status == JobStatus.PENDING
        assert response.topic == "What is machine learning?"
        assert response.current_stage is None
        assert response.progress_percentage is None

    def test_status_response_with_progress(self):
        """Test status response with progress info."""
        from datetime import datetime, timezone

        response = ResearchStatusResponse(
            job_id="123e4567-e89b-12d3-a456-426614174000",
            status=JobStatus.PROCESSING,
            topic="Test topic",
            created_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 15, 10, 35, tzinfo=timezone.utc),
            current_stage="fact_check",
            progress_percentage=40,
        )

        assert response.current_stage == "fact_check"
        assert response.progress_percentage == 40

    def test_status_response_with_error(self):
        """Test status response with error info."""
        from datetime import datetime, timezone

        response = ResearchStatusResponse(
            job_id="123e4567-e89b-12d3-a456-426614174000",
            status=JobStatus.FAILED,
            topic="Test topic",
            created_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 15, 10, 35, tzinfo=timezone.utc),
            error="Connection timeout",
        )

        assert response.error == "Connection timeout"

    def test_progress_percentage_bounds(self):
        """Test progress_percentage must be between 0 and 100."""
        from datetime import datetime, timezone

        # Valid
        response = ResearchStatusResponse(
            job_id="test",
            status=JobStatus.PENDING,
            topic="test",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            progress_percentage=50,
        )
        assert response.progress_percentage == 50

        # Invalid - too low
        with pytest.raises(ValidationError):
            ResearchStatusResponse(
                job_id="test",
                status=JobStatus.PENDING,
                topic="test",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                progress_percentage=-1,
            )

        # Invalid - too high
        with pytest.raises(ValidationError):
            ResearchStatusResponse(
                job_id="test",
                status=JobStatus.PENDING,
                topic="test",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                progress_percentage=101,
            )


class TestResearchJobResponse:
    """Tests for ResearchJobResponse model (extends ResearchStatusResponse)."""

    def test_full_response_with_all_fields(self):
        """Test creating a full job response with all fields."""
        from datetime import datetime, timezone

        response = ResearchJobResponse(
            job_id="123e4567-e89b-12d3-a456-426614174000",
            status=JobStatus.COMPLETED,
            topic="Environmental impact of EVs",
            created_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 15, 10, 45, tzinfo=timezone.utc),
            current_stage="completed",
            progress_percentage=100,
            sources=[
                ResearchSource(title="EPA Study", url="https://epa.gov/ev"),
            ],
            findings=["EVs reduce emissions by 60%"],
            claims_verified=5,
            claims_partially_verified=2,
            claims_disputed=0,
            claims_unverified=1,
            insights=["Battery production has environmental costs"],
            report_title="Environmental Impact of Electric Vehicles",
            report_content="# Report content...",
            report_format="markdown",
            review_score=0.85,
            review_approved=True,
            review_suggestions=["Add more recent data"],
            review_iterations=2,
        )

        assert response.status == JobStatus.COMPLETED
        assert response.sources is not None and len(response.sources) == 1
        assert response.findings is not None and len(response.findings) == 1
        assert response.claims_verified == 5
        assert response.insights == ["Battery production has environmental costs"]
        assert response.report_title == "Environmental Impact of Electric Vehicles"
        assert response.review_score == 0.85
        assert response.review_approved is True

    def test_review_score_bounds(self):
        """Test review_score must be between 0.0 and 1.0."""
        from datetime import datetime, timezone

        # Valid
        response = ResearchJobResponse(
            job_id="test",
            status=JobStatus.COMPLETED,
            topic="test",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            review_score=0.5,
        )
        assert response.review_score == 0.5

        # Invalid - too low
        with pytest.raises(ValidationError):
            ResearchJobResponse(
                job_id="test",
                status=JobStatus.COMPLETED,
                topic="test",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                review_score=-0.1,
            )

        # Invalid - too high
        with pytest.raises(ValidationError):
            ResearchJobResponse(
                job_id="test",
                status=JobStatus.COMPLETED,
                topic="test",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                review_score=1.1,
            )

    def test_minimal_response(self):
        """Test creating a minimal job response."""
        from datetime import datetime, timezone

        response = ResearchJobResponse(
            job_id="123e4567-e89b-12d3-a456-426614174000",
            status=JobStatus.PENDING,
            topic="Test topic",
            created_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
        )

        # All optional fields should be None
        assert response.sources is None
        assert response.findings is None
        assert response.report_title is None
        assert response.report_content is None
        assert response.review_score is None
