"""Unit tests for CriticAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.critic import CriticAgent
from src.domain.events import ReportReviewed, ReportWritten
from src.domain.interfaces import AgentContext


class TestCriticAgent:
    """Tests for CriticAgent class."""

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
    def critic_agent(self, mock_llm):
        """Create a CriticAgent with mocked LLM."""
        with patch("src.agents.critic.BaseAgent.__init__", return_value=None):
            agent = CriticAgent()
            agent._llm = mock_llm
            agent._name = "critic"
            agent._description = "Reviews reports for clarity, logic, and completeness"
            agent._correlation_id = None
            return agent

    def test_agent_name(self, critic_agent):
        """Test that agent name is 'critic'."""
        assert critic_agent.name == "critic"

    def test_agent_description(self, critic_agent):
        """Test agent description."""
        assert "clarity" in critic_agent.description.lower()
        assert "logic" in critic_agent.description.lower()

    @pytest.mark.asyncio
    async def test_validate_input_accepts_report_written(self, critic_agent):
        """Test that validate_input accepts ReportWritten events."""
        report = ReportWritten.create(
            title="Test Report",
            content="Test content",
            format="markdown",
        )
        assert await critic_agent.validate_input(report) is True

    @pytest.mark.asyncio
    async def test_validate_input_rejects_other_types(self, critic_agent):
        """Test that validate_input rejects non-ReportWritten inputs."""
        assert await critic_agent.validate_input("string") is False
        assert await critic_agent.validate_input(123) is False
        assert await critic_agent.validate_input(None) is False
        assert await critic_agent.validate_input({}) is False

    @pytest.mark.asyncio
    async def test_review_method_exists(self, critic_agent):
        """Test that review convenience method exists."""
        assert hasattr(critic_agent, "review")
        assert callable(critic_agent.review)


class TestCriticAgentRun:
    """Tests for CriticAgent._run method."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper with JSON response."""

        def mock_ainvoke(messages):
            return MagicMock(
                content='{"suggestions": ["Improve conclusion", "Add more citations"], "score": 0.8, "approved": true}'
            )

        mock = MagicMock()
        mock.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        return mock

    @pytest.fixture
    def agent_context(self):
        """Create a test agent context."""
        return AgentContext.create(correlation_id="test-correlation-id")

    @pytest.fixture
    def critic_agent(self, mock_llm):
        """Create a CriticAgent with mocked LLM."""
        with patch("src.agents.critic.BaseAgent.__init__", return_value=None):
            agent = CriticAgent()
            agent._llm = mock_llm
            agent._name = "critic"
            agent._description = ""
            agent._correlation_id = None
            return agent

    @pytest.mark.asyncio
    async def test_run_parses_valid_json_response(self, critic_agent, agent_context):
        """Test that _run correctly parses valid JSON response."""
        report = ReportWritten.create(
            title="Test Report",
            content="This is test content for the report.",
            format="markdown",
        )

        result = await critic_agent._run(report, agent_context)

        assert result.suggestions == ["Improve conclusion", "Add more citations"]
        assert result.score == 0.8
        assert result.approved is True
        assert result.correlation_id == agent_context.correlation_id

    @pytest.mark.asyncio
    async def test_run_handles_invalid_json_with_fallback(
        self, mock_llm, agent_context
    ):
        """Test that _run handles invalid JSON response gracefully."""
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="No proper JSON format")
        )

        with patch("src.agents.critic.BaseAgent.__init__", return_value=None):
            agent = CriticAgent()
            agent._llm = mock_llm
            agent._name = "critic"
            agent._description = ""
            agent._correlation_id = None

            report = ReportWritten.create(
                title="Test Report",
                content="Test content",
                format="markdown",
            )

            result = await agent._run(report, agent_context)

            assert len(result.suggestions) > 0
            assert result.score == 0.5
            assert result.approved is False


class TestCriticAgentIntegration:
    """Integration tests for CriticAgent with full execute flow."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper."""

        def mock_ainvoke(messages):
            return MagicMock(
                content='{"suggestions": ["Add executive summary"], "score": 0.9, "approved": true}'
            )

        mock = MagicMock()
        mock.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        return mock

    @pytest.fixture
    def agent_context(self):
        """Create a test agent context."""
        return AgentContext.create(correlation_id="integration-test-id")

    @pytest.mark.asyncio
    async def test_full_review_flow(self, mock_llm, agent_context):
        """Test complete review flow from execute to result."""
        with patch("src.agents.critic.BaseAgent.__init__", return_value=None):
            agent = CriticAgent()
            agent._llm = mock_llm
            agent._name = "critic"
            agent._description = ""
            agent._correlation_id = None

            report = ReportWritten.create(
                title="Annual Report 2024",
                content="This report covers our achievements...",
                format="markdown",
            )

            result = await agent.review(report, agent_context)

            assert isinstance(result, ReportReviewed)
            assert result.suggestions == ["Add executive summary"]
            assert result.score == 0.9
            assert result.approved is True
            assert result.correlation_id == "integration-test-id"

    @pytest.mark.asyncio
    async def test_review_with_low_score(self, mock_llm, agent_context):
        """Test review that results in low quality score."""

        def mock_ainvoke(messages):
            return MagicMock(
                content='{"suggestions": ["Major revisions needed"], "score": 0.3, "approved": false}'
            )

        mock_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        with patch("src.agents.critic.BaseAgent.__init__", return_value=None):
            agent = CriticAgent()
            agent._llm = mock_llm
            agent._name = "critic"
            agent._description = ""
            agent._correlation_id = None

            report = ReportWritten.create(
                title="Draft Report",
                content="Content needs work",
                format="markdown",
            )

            result = await agent.review(report, agent_context)

            assert result.score == 0.3
            assert result.approved is False
            assert len(result.suggestions) == 1
