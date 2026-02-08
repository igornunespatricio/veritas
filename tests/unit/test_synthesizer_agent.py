"""Unit tests for SynthesizerAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.synthesizer import SynthesizerAgent
from src.domain.events import FactCheckCompleted, ResearchCompleted, SynthesisCompleted
from src.domain.interfaces import AgentContext


class TestSynthesizerAgent:
    """Tests for SynthesizerAgent class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper."""
        mock = MagicMock()
        mock.ainvoke = AsyncMock(return_value=MagicMock(content="mock response"))
        return mock

    @pytest.fixture
    def agent_context(self):
        """Create a test agent context."""
        return AgentContext.create(correlation_id="test-correlation-id")

    @pytest.fixture
    def synthesizer_agent(self, mock_llm):
        """Create a SynthesizerAgent with mocked LLM."""
        with patch("src.agents.synthesizer.BaseAgent.__init__", return_value=None):
            agent = SynthesizerAgent()
            agent._llm = mock_llm
            agent._name = "synthesizer"
            agent._description = "Merges validated research into coherent insights"
            agent._correlation_id = None
            return agent

    def test_agent_name(self, synthesizer_agent):
        """Test that agent name is 'synthesizer'."""
        assert synthesizer_agent.name == "synthesizer"

    def test_agent_description(self, synthesizer_agent):
        """Test agent description."""
        assert "merges" in synthesizer_agent.description.lower()
        assert "insights" in synthesizer_agent.description.lower()

    @pytest.mark.asyncio
    async def test_validate_input_accepts_dict_with_required_keys(
        self, synthesizer_agent
    ):
        """Test that validate_input accepts dict with 'research' and 'fact_check'."""
        research = ResearchCompleted.create(
            topic="Test",
            sources=[],
            findings=["Finding"],
        )
        fact_check = FactCheckCompleted.create(
            claims=[],
            verified_claims=[],
            confidence_scores={},
        )
        assert (
            await synthesizer_agent.validate_input(
                {"research": research, "fact_check": fact_check}
            )
            is True
        )

    @pytest.mark.asyncio
    async def test_validate_input_rejects_dict_missing_keys(self, synthesizer_agent):
        """Test that validate_input rejects dict missing required keys."""
        assert await synthesizer_agent.validate_input({"research": {}}) is False
        assert await synthesizer_agent.validate_input({"fact_check": {}}) is False
        assert await synthesizer_agent.validate_input({}) is False

    @pytest.mark.asyncio
    async def test_validate_input_rejects_other_types(self, synthesizer_agent):
        """Test that validate_input rejects other input types."""
        assert await synthesizer_agent.validate_input("string") is False
        assert await synthesizer_agent.validate_input(123) is False
        assert await synthesizer_agent.validate_input(None) is False

    @pytest.mark.asyncio
    async def test_synthesize_method_exists(self, synthesizer_agent):
        """Test that synthesize convenience method exists."""
        assert hasattr(synthesizer_agent, "synthesize")
        assert callable(synthesizer_agent.synthesize)


class TestSynthesizerAgentRun:
    """Tests for SynthesizerAgent._run method."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper with JSON response."""

        def mock_ainvoke(messages):
            return MagicMock(
                content='{"insights": ["Insight 1", "Insight 2"], "resolved_contradictions": [{"issue": "Contradiction A", "resolution": "Both sides have merit"}]}'
            )

        mock = MagicMock()
        mock.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        return mock

    @pytest.fixture
    def agent_context(self):
        """Create a test agent context."""
        return AgentContext.create(correlation_id="test-correlation-id")

    @pytest.fixture
    def synthesizer_agent(self, mock_llm):
        """Create a SynthesizerAgent with mocked LLM."""
        with patch("src.agents.synthesizer.BaseAgent.__init__", return_value=None):
            agent = SynthesizerAgent()
            agent._llm = mock_llm
            agent._name = "synthesizer"
            agent._description = ""
            agent._correlation_id = None
            return agent

    @pytest.mark.asyncio
    async def test_run_parses_valid_json_response(
        self, synthesizer_agent, agent_context
    ):
        """Test that _run correctly parses valid JSON response."""
        research = ResearchCompleted.create(
            topic="Climate Change",
            sources=[{"url": "http://example.com", "title": "Test"}],
            findings=["Finding 1", "Finding 2"],
        )
        fact_check = FactCheckCompleted.create(
            claims=[{"text": "Finding 1", "status": "verified"}],
            verified_claims=[{"text": "Finding 1", "status": "verified"}],
            confidence_scores={"Finding 1": 0.9},
        )

        result = await synthesizer_agent._run(
            {"research": research, "fact_check": fact_check}, agent_context
        )

        assert len(result.insights) == 2
        assert len(result.resolved_contradictions) == 1
        assert result.correlation_id == agent_context.correlation_id

    @pytest.mark.asyncio
    async def test_run_handles_invalid_json_with_fallback(
        self, mock_llm, agent_context
    ):
        """Test that _run handles invalid JSON response gracefully."""
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="No JSON response"))

        with patch("src.agents.synthesizer.BaseAgent.__init__", return_value=None):
            agent = SynthesizerAgent()
            agent._llm = mock_llm
            agent._name = "synthesizer"
            agent._description = ""
            agent._correlation_id = None

            research = ResearchCompleted.create(
                topic="Test",
                sources=[],
                findings=["Finding"],
            )
            fact_check = FactCheckCompleted.create(
                claims=[],
                verified_claims=[],
                confidence_scores={},
            )

            result = await agent._run(
                {"research": research, "fact_check": fact_check}, agent_context
            )

            # Should use fallback handling
            assert len(result.insights) == 1


class TestSynthesizerAgentIntegration:
    """Integration tests for SynthesizerAgent with full execute flow."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper."""

        def mock_ainvoke(messages):
            return MagicMock(
                content='{"insights": ["Climate change is accelerating", "Renewable energy adoption is growing"], "resolved_contradictions": []}'
            )

        mock = MagicMock()
        mock.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        return mock

    @pytest.fixture
    def agent_context(self):
        """Create a test agent context."""
        return AgentContext.create(correlation_id="integration-test-id")

    @pytest.mark.asyncio
    async def test_full_synthesize_flow(self, mock_llm, agent_context):
        """Test complete synthesis flow from execute to result."""
        with patch("src.agents.synthesizer.BaseAgent.__init__", return_value=None):
            agent = SynthesizerAgent()
            agent._llm = mock_llm
            agent._name = "synthesizer"
            agent._description = ""
            agent._correlation_id = None

            research = ResearchCompleted.create(
                topic="Energy Trends",
                sources=[
                    {"url": "http://example.com/1", "title": "Source 1"},
                    {"url": "http://example.com/2", "title": "Source 2"},
                ],
                findings=[
                    "Solar costs decreasing",
                    "Wind adoption increasing",
                ],
            )
            fact_check = FactCheckCompleted.create(
                claims=[
                    {"text": "Solar costs decreasing", "status": "verified"},
                    {
                        "text": "Wind adoption increasing",
                        "status": "partially_verified",
                    },
                ],
                verified_claims=[
                    {"text": "Solar costs decreasing", "status": "verified"}
                ],
                confidence_scores={
                    "Solar costs decreasing": 0.95,
                    "Wind adoption increasing": 0.7,
                },
            )

            result = await agent.synthesize(research, fact_check, agent_context)

            # Verify result
            assert isinstance(result, SynthesisCompleted)
            assert len(result.insights) == 2
            assert result.correlation_id == "integration-test-id"

    @pytest.mark.asyncio
    async def test_synthesize_with_empty_findings(self, mock_llm, agent_context):
        """Test synthesis with empty research findings."""
        with patch("src.agents.synthesizer.BaseAgent.__init__", return_value=None):
            agent = SynthesizerAgent()
            agent._llm = mock_llm
            agent._name = "synthesizer"
            agent._description = ""
            agent._correlation_id = None

            research = ResearchCompleted.create(
                topic="Empty Topic",
                sources=[],
                findings=[],
            )
            fact_check = FactCheckCompleted.create(
                claims=[],
                verified_claims=[],
                confidence_scores={},
            )

            result = await agent.synthesize(research, fact_check, agent_context)

            assert isinstance(result, SynthesisCompleted)
