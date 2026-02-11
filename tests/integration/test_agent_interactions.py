"""Agent-to-agent interaction integration tests."""

import pytest
from unittest.mock import AsyncMock, patch

from src.agents import (
    ResearcherAgent,
    FactCheckerAgent,
    SynthesizerAgent,
    WriterAgent,
    CriticAgent,
)
from src.domain.events import (
    ResearchCompleted,
    FactCheckCompleted,
    SynthesisCompleted,
    ReportWritten,
    ReportReviewed,
)
from src.domain.interfaces import AgentContext


class TestResearcherToFactCheckerFlow:
    """Test researcher to fact-checker data flow."""

    @pytest.mark.asyncio
    async def test_researcher_output_feeds_factchecker(self):
        """Verify ResearchCompleted event can be processed by fact-checker."""
        researcher = ResearcherAgent(provider="ollama", model="llama3.2:3b")
        fact_checker = FactCheckerAgent(provider="ollama", model="llama3.2:3b")

        # Create sample research output
        research = ResearchCompleted.create(
            topic="artificial intelligence ethics",
            sources=[
                {
                    "url": "https://example.com/ai-ethics",
                    "title": "AI Ethics Guidelines",
                    "date": "2024-01-15",
                },
                {
                    "url": "https://example.com/ai-safety",
                    "title": "AI Safety Principles",
                    "date": "2024-02-01",
                },
            ],
            findings=[
                "AI systems should be transparent and explainable",
                "AI should not discriminate against protected groups",
                "Human oversight should be maintained for critical decisions",
                "AI development should follow ethical guidelines",
            ],
        )

        context = AgentContext.create(correlation_id="test-correlation-001")

        # Mock the LLM call for fact-checker
        with patch.object(
            fact_checker.llm, "ainvoke", new_callable=AsyncMock
        ) as mock_invoke:
            # Simulate a valid JSON response
            mock_response = type(
                "MockResponse",
                (),
                {
                    "content": '{"claims": [{"text": "AI systems should be transparent", "status": "verified"}], "verified_claims": [{"text": "AI systems should be transparent", "status": "verified"}], "confidence_scores": {"AI systems should be transparent": 0.95}}'
                },
            )()
            mock_invoke.return_value = mock_response

            result = await fact_checker.verify_claims(
                claims=research.findings,
                sources=research.sources,
                context=context,
            )

        # Verify output structure
        assert result is not None
        assert hasattr(result, "claims")
        assert hasattr(result, "verified_claims")
        assert hasattr(result, "confidence_scores")
        assert len(result.claims) > 0

    @pytest.mark.asyncio
    async def test_factchecker_preserves_research_topic(self):
        """Verify that fact-checker maintains context of research topic."""
        fact_checker = FactCheckerAgent(provider="ollama", model="llama3.2:3b")

        research = ResearchCompleted.create(
            topic="renewable energy advancements",
            sources=[
                {
                    "url": "https://example.com/renewable",
                    "title": "Renewable Energy",
                    "date": "2024-01-01",
                }
            ],
            findings=[
                "Solar panel efficiency has improved significantly",
                "Wind power costs have decreased by 50%",
            ],
        )

        context = AgentContext.create(correlation_id="test-topic-preservation")

        with patch.object(
            fact_checker.llm, "ainvoke", new_callable=AsyncMock
        ) as mock_invoke:
            mock_response = type(
                "MockResponse",
                (),
                {
                    "content": '{"claims": [{"text": "Solar panel efficiency", "status": "verified"}], "verified_claims": [], "confidence_scores": {}}'
                },
            )()
            mock_invoke.return_value = mock_response

            result = await fact_checker.verify_claims(
                claims=research.findings,
                sources=research.sources,
                context=context,
            )

        # The fact-check result should be traceable to the original research
        assert result.correlation_id == context.correlation_id


class TestFactCheckerToSynthesizerFlow:
    """Test fact-checker to synthesizer data flow."""

    @pytest.mark.asyncio
    async def test_factcheck_output_feeds_synthesizer(self):
        """Verify FactCheckCompleted event flows into synthesizer."""
        synthesizer = SynthesizerAgent(provider="ollama", model="llama3.2:3b")

        fact_check = FactCheckCompleted.create(
            claims=[
                {
                    "text": "Quantum computers use qubits",
                    "status": "verified",
                },
                {
                    "text": "Quantum superposition allows multiple states",
                    "status": "partially_verified",
                },
                {
                    "text": "Quantum entanglement enables instant communication",
                    "status": "disputed",
                },
            ],
            verified_claims=[
                {"text": "Quantum computers use qubits", "status": "verified"}
            ],
            confidence_scores={
                "Quantum computers use qubits": 0.95,
                "Quantum superposition allows multiple states": 0.70,
                "Quantum entanglement enables instant communication": 0.30,
            },
        )

        research = ResearchCompleted.create(
            topic="quantum computing",
            sources=[{"url": "", "title": "", "date": ""}],
            findings=["finding 1", "finding 2"],
        )

        context = AgentContext.create(correlation_id="test-synthesis-flow")

        with patch.object(
            synthesizer.llm, "ainvoke", new_callable=AsyncMock
        ) as mock_invoke:
            mock_response = type(
                "MockResponse",
                (),
                {
                    "content": '{"insights": ["Quantum computing uses fundamentally different computation model"], "resolved_contradictions": []}'
                },
            )()
            mock_invoke.return_value = mock_response

            result = await synthesizer.synthesize(
                research=research,
                fact_check=fact_check,
                context=context,
            )

        assert result is not None
        assert hasattr(result, "insights")
        assert hasattr(result, "resolved_contradictions")
        assert len(result.insights) > 0


class TestSynthesizerToWriterFlow:
    """Test synthesizer to writer data flow."""

    @pytest.mark.asyncio
    async def test_synthesis_output_feeds_writer(self):
        """Verify SynthesisCompleted event flows into writer."""
        writer = WriterAgent(provider="ollama", model="llama3.2:3b")

        synthesis = SynthesisCompleted.create(
            insights=[
                "Blockchain provides decentralized trust mechanisms",
                "Smart contracts enable automated execution of agreements",
                "Cryptography ensures transaction security and privacy",
            ],
            resolved_contradictions=[
                {
                    "contradiction": "Blockchain scalability vs security",
                    "resolution": "Layer 2 solutions address scalability while maintaining security guarantees",
                }
            ],
        )

        context = AgentContext.create(correlation_id="test-writer-flow")

        with patch.object(writer.llm, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_response = type(
                "MockResponse",
                (),
                {
                    "content": '{"title": "Blockchain Technology Overview", "content": "# Blockchain Technology\\n\\n## Introduction\\n\\nBlockchain represents...", "format": "markdown"}'
                },
            )()
            mock_invoke.return_value = mock_response

            result = await writer.write_report(
                synthesis=synthesis,
                format="markdown",
                context=context,
            )

        assert result is not None
        assert hasattr(result, "title")
        assert hasattr(result, "content")
        assert hasattr(result, "format")
        assert "Blockchain" in result.title or result.title != ""

    @pytest.mark.asyncio
    async def test_writer_supports_plain_format(self):
        """Verify writer can produce plain text format."""
        writer = WriterAgent(provider="ollama", model="llama3.2:3b")

        synthesis = SynthesisCompleted.create(
            insights=["Key insight one", "Key insight two"],
            resolved_contradictions=[],
        )

        context = AgentContext.create(correlation_id="test-plain-format")

        with patch.object(writer.llm, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_response = type(
                "MockResponse",
                (),
                {
                    "content": '{"title": "Plain Text Report", "content": "PLAIN TEXT REPORT\\n\\nThis is plain content", "format": "plain"}'
                },
            )()
            mock_invoke.return_value = mock_response

            result = await writer.write_report(
                synthesis=synthesis,
                format="plain",
                context=context,
            )

        assert result.format == "plain"


class TestWriterToCriticFlow:
    """Test writer to critic data flow."""

    @pytest.mark.asyncio
    async def test_report_output_feeds_critic(self):
        """Verify ReportWritten event is reviewed by critic."""
        critic = CriticAgent(provider="ollama", model="llama3.2:3b")

        report = ReportWritten.create(
            title="Climate Change Analysis",
            content="# Climate Change Analysis\n\n## Overview\n\nThis report examines the impacts of climate change on global ecosystems.",
            format="markdown",
        )

        context = AgentContext.create(correlation_id="test-critic-flow")

        with patch.object(critic.llm, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_response = type(
                "MockResponse",
                (),
                {
                    "content": '{"suggestions": ["Add more recent data", "Include economic impact section"], "score": 0.75, "approved": false}'
                },
            )()
            mock_invoke.return_value = mock_response

            result = await critic.review(report=report, context=context)

        assert result is not None
        assert hasattr(result, "suggestions")
        assert hasattr(result, "score")
        assert hasattr(result, "approved")
        assert isinstance(result.suggestions, list)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_critic_can_approve_report(self):
        """Verify critic can approve a high-quality report."""
        critic = CriticAgent(provider="ollama", model="llama3.2:3b")

        report = ReportWritten.create(
            title="Excellent Research Report",
            content="# Excellent Report\n\nThis is a comprehensive and well-written report with proper citations.",
            format="markdown",
        )

        context = AgentContext.create(correlation_id="test-approval")

        with patch.object(critic.llm, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_response = type(
                "MockResponse",
                (),
                {"content": '{"suggestions": [], "score": 0.92, "approved": true}'},
            )()
            mock_invoke.return_value = mock_response

            result = await critic.review(report=report, context=context)

        assert result.approved is True
        assert result.score >= 0.8


class TestMultiAgentDataContract:
    """Test that data contracts are maintained across agents."""

    @pytest.mark.asyncio
    async def test_research_contains_required_fields(self):
        """Verify researcher output contains all required fields."""
        researcher = ResearcherAgent(provider="ollama", model="llama3.2:3b")

        # Create minimal research
        research = ResearchCompleted.create(
            topic="test",
            sources=[{"url": "", "title": "", "date": ""}],
            findings=["finding"],
        )

        assert research.topic == "test"
        assert len(research.sources) == 1
        assert len(research.findings) == 1
        assert "url" in research.sources[0]
        assert research.correlation_id is not None

    @pytest.mark.asyncio
    async def test_factcheck_claim_status_normalization(self):
        """Verify fact-checker normalizes claim statuses correctly."""
        fact_checker = FactCheckerAgent(provider="ollama", model="llama3.2:3b")

        # Test status normalization through the private method
        claims = [
            {"text": "Claim 1", "status": "VERIFIED"},
            {"text": "Claim 2", "status": "Partially_Verified"},
            {"text": "Claim 3", "status": "disputed"},
            {"text": "Claim 4", "status": "unknown_status"},
        ]

        normalized = fact_checker._normalize_claim_statuses(claims)

        assert normalized[0]["status"] == "verified"
        assert normalized[1]["status"] == "partially_verified"
        assert normalized[2]["status"] == "disputed"
        assert normalized[3]["status"] == "unverified"  # Invalid defaults to unverified

    @pytest.mark.asyncio
    async def test_event_correlation_id_tracking(self):
        """Verify correlation ID is preserved through all events."""
        correlation_id = "unique-test-correlation-id"

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
