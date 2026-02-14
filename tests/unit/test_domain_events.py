"""Unit tests for domain events."""

from datetime import datetime
from uuid import UUID

from src.domain.events import (
    DomainEvent,
    FactCheckCompleted,
    ReportReviewed,
    ReportWritten,
    ResearchCompleted,
    SynthesisCompleted,
)


class TestDomainEvent:
    """Tests for base DomainEvent class."""

    def test_create_event(self):
        """Test creating a base domain event."""
        event = DomainEvent.create(
            event_type="test.event",
            payload={"key": "value"},
            correlation_id="test-correlation",
        )

        assert event.event_type == "test.event"
        assert event.payload == {"key": "value"}
        assert event.correlation_id == "test-correlation"
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_event_id_is_uuid(self):
        """Test that event_id is a valid UUID."""
        event = DomainEvent.create(event_type="test", payload={})
        # Should not raise
        uuid_obj = UUID(event.event_id)
        assert str(uuid_obj) == event.event_id

    def test_timestamp_is_datetime(self):
        """Test that timestamp is a datetime object."""
        event = DomainEvent.create(event_type="test", payload={})
        assert isinstance(event.timestamp, datetime)

    def test_auto_generate_correlation_id(self):
        """Test that correlation_id is auto-generated if not provided."""
        event = DomainEvent.create(event_type="test", payload={})
        assert event.correlation_id is not None
        assert len(event.correlation_id) > 0


class TestResearchCompleted:
    """Tests for ResearchCompleted event."""

    def test_create_research_completed(self):
        """Test creating a ResearchCompleted event."""
        event = ResearchCompleted.create(
            topic="Climate Change",
            sources=[{"url": "http://example.com", "title": "Source 1"}],
            findings=["Finding 1", "Finding 2"],
            correlation_id="research-123",
        )

        assert event.topic == "Climate Change"
        assert len(event.sources) == 1
        assert len(event.findings) == 2
        assert event.correlation_id == "research-123"
        assert event.event_type == "research.completed"

    def test_research_completed_payload(self):
        """Test that payload contains all fields."""
        event = ResearchCompleted.create(
            topic="AI Research",
            sources=[{"url": "http://ai.com", "title": "AI Source"}],
            findings=["AI is growing"],
        )

        assert event.payload["topic"] == "AI Research"
        assert event.payload["sources"] == [
            {"url": "http://ai.com", "title": "AI Source"}
        ]
        assert event.payload["findings"] == ["AI is growing"]

    def test_research_completed_auto_correlation_id(self):
        """Test auto-generation of correlation_id."""
        event = ResearchCompleted.create(
            topic="Test",
            sources=[],
            findings=[],
        )
        assert event.correlation_id is not None


class TestFactCheckCompleted:
    """Tests for FactCheckCompleted event."""

    def test_create_fact_check_completed(self):
        """Test creating a FactCheckCompleted event."""
        claims = [
            {"text": "Earth is round", "status": "verified"},
            {"text": "Water is wet", "status": "partially_verified"},
        ]
        verified = [{"text": "Earth is round", "status": "verified"}]
        scores = {"Earth is round": 0.95, "Water is wet": 0.7}

        event = FactCheckCompleted.create(
            claims=claims,
            verified_claims=verified,
            confidence_scores=scores,
            correlation_id="factcheck-456",
        )

        assert len(event.claims) == 2
        assert len(event.verified_claims) == 1
        assert event.confidence_scores["Earth is round"] == 0.95
        assert event.event_type == "fact_check.completed"

    def test_fact_check_completed_payload(self):
        """Test that payload contains all fields."""
        event = FactCheckCompleted.create(
            claims=[],
            verified_claims=[],
            confidence_scores={},
        )

        assert "claims" in event.payload
        assert "verified_claims" in event.payload
        assert "confidence_scores" in event.payload


class TestSynthesisCompleted:
    """Tests for SynthesisCompleted event."""

    def test_create_synthesis_completed(self):
        """Test creating a SynthesisCompleted event."""
        insights = ["Insight 1", "Insight 2"]
        resolved = [{"issue": "Contradiction A", "resolution": "Resolved"}]

        event = SynthesisCompleted.create(
            insights=insights,
            resolved_contradictions=resolved,
            correlation_id="synthesis-789",
        )

        assert len(event.insights) == 2
        assert len(event.resolved_contradictions) == 1
        assert event.event_type == "synthesis.completed"

    def test_synthesis_completed_with_empty_lists(self):
        """Test creating event with empty lists."""
        event = SynthesisCompleted.create(
            insights=[],
            resolved_contradictions=[],
        )

        assert event.insights == []
        assert event.resolved_contradictions == []


class TestReportWritten:
    """Tests for ReportWritten event."""

    def test_create_report_written_default_format(self):
        """Test creating a ReportWritten event with default format."""
        event = ReportWritten.create(
            title="Annual Report",
            content="# Annual Report\n\nContent here",
        )

        assert event.title == "Annual Report"
        assert "Annual Report" in event.content
        assert event.format == "markdown"
        assert event.event_type == "report.written"

    def test_create_report_written_custom_format(self):
        """Test creating a ReportWritten event with custom format."""
        event = ReportWritten.create(
            title="Plain Report",
            content="Plain text content",
            format="plain",
        )

        assert event.format == "plain"

    def test_create_report_written_html_format(self):
        """Test creating a ReportWritten event with HTML format."""
        event = ReportWritten.create(
            title="HTML Report",
            content="<h1>HTML Report</h1>",
            format="html",
        )

        assert event.format == "html"

    def test_report_written_payload(self):
        """Test that payload contains all fields."""
        event = ReportWritten.create(
            title="Test",
            content="Content",
            format="markdown",
        )

        assert event.payload["title"] == "Test"
        assert event.payload["content"] == "Content"
        assert event.payload["format"] == "markdown"


class TestReportReviewed:
    """Tests for ReportReviewed event."""

    def test_create_report_reviewed_approved(self):
        """Test creating a ReportReviewed event for approved report."""
        suggestions = ["Add executive summary"]
        event = ReportReviewed.create(
            suggestions=suggestions,
            score=0.9,
            approved=True,
            correlation_id="review-101",
        )

        assert len(event.suggestions) == 1
        assert event.score == 0.9
        assert event.approved is True
        assert event.event_type == "report.reviewed"

    def test_create_report_reviewed_rejected(self):
        """Test creating a ReportReviewed event for rejected report."""
        suggestions = ["Major revisions needed"]
        event = ReportReviewed.create(
            suggestions=suggestions,
            score=0.3,
            approved=False,
        )

        assert event.approved is False
        assert event.score == 0.3

    def test_report_reviewed_payload(self):
        """Test that payload contains all fields."""
        event = ReportReviewed.create(
            suggestions=["Suggestion 1"],
            score=0.8,
            approved=True,
        )

        assert "suggestions" in event.payload
        assert "score" in event.payload
        assert "approved" in event.payload


class TestEventTypes:
    """Tests for event type constants."""

    def test_research_event_type(self):
        """Test research event type."""
        event = ResearchCompleted.create(topic="Test", sources=[], findings=[])
        assert event.event_type == "research.completed"

    def test_fact_check_event_type(self):
        """Test fact check event type."""
        event = FactCheckCompleted.create(
            claims=[], verified_claims=[], confidence_scores={}
        )
        assert event.event_type == "fact_check.completed"

    def test_synthesis_event_type(self):
        """Test synthesis event type."""
        event = SynthesisCompleted.create(insights=[], resolved_contradictions=[])
        assert event.event_type == "synthesis.completed"

    def test_report_written_event_type(self):
        """Test report written event type."""
        event = ReportWritten.create(title="Test", content="Content")
        assert event.event_type == "report.written"

    def test_report_reviewed_event_type(self):
        """Test report reviewed event type."""
        event = ReportReviewed.create(suggestions=[], score=1.0, approved=True)
        assert event.event_type == "report.reviewed"
