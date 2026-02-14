"""Unit tests for ResearcherAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.researcher import ResearcherAgent
from src.domain.interfaces import AgentContext


class TestResearcherAgent:
    """Tests for ResearcherAgent class."""

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
    def researcher_agent(self, mock_llm):
        """Create a ResearcherAgent with mocked LLM."""
        with patch("src.agents.researcher.BaseAgent.__init__", return_value=None):
            agent = ResearcherAgent()
            agent._llm = mock_llm
            agent._name = "researcher"
            agent._description = "Collects raw information, sources, and key findings"
            agent._correlation_id = None
            agent._tools = []
            return agent

    def test_agent_name(self, researcher_agent):
        """Test that agent name is 'researcher'."""
        assert researcher_agent.name == "researcher"

    def test_agent_description(self, researcher_agent):
        """Test agent description."""
        assert "collects" in researcher_agent.description.lower()
        assert "findings" in researcher_agent.description.lower()

    @pytest.mark.asyncio
    async def test_validate_input_accepts_valid_string(self, researcher_agent):
        """Test that validate_input accepts non-empty strings."""
        assert await researcher_agent.validate_input("Climate Change") is True
        assert await researcher_agent.validate_input("  topic with spaces  ") is True

    @pytest.mark.asyncio
    async def test_validate_input_accepts_dict_with_topic(self, researcher_agent):
        """Test that validate_input accepts dict with 'topic' key."""
        assert await researcher_agent.validate_input({"topic": "Test Topic"}) is True

    @pytest.mark.asyncio
    async def test_validate_input_rejects_empty_string(self, researcher_agent):
        """Test that validate_input rejects empty strings."""
        assert await researcher_agent.validate_input("") is False
        assert await researcher_agent.validate_input("   ") is False

    @pytest.mark.asyncio
    async def test_validate_input_rejects_dict_with_empty_topic(self, researcher_agent):
        """Test that validate_input rejects dict with empty topic."""
        assert await researcher_agent.validate_input({"topic": ""}) is False
        assert await researcher_agent.validate_input({"topic": "   "}) is False

    @pytest.mark.asyncio
    async def test_validate_input_rejects_other_types(self, researcher_agent):
        """Test that validate_input rejects other input types."""
        assert await researcher_agent.validate_input(123) is False
        assert await researcher_agent.validate_input(None) is False
        assert await researcher_agent.validate_input({}) is False
        assert await researcher_agent.validate_input([]) is False

    @pytest.mark.asyncio
    async def test_research_method_exists(self, researcher_agent):
        """Test that research convenience method exists."""
        assert hasattr(researcher_agent, "research")
        assert callable(researcher_agent.research)


class TestResearcherAgentRun:
    """Tests for ResearcherAgent._run method with fallback path."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper with JSON response."""

        def mock_ainvoke(messages):
            return MagicMock(
                content='{"sources": [{"url": "http://example.com", "title": "Test Source", "date": "2024-01-01"}], "findings": ["Finding 1", "Finding 2"]}'
            )

        mock = MagicMock()
        mock.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        return mock

    @pytest.fixture
    def mock_search_tool(self):
        """Create a mock search tool."""
        mock = MagicMock()
        mock.invoke = MagicMock(return_value="Search results about the topic")
        mock.ainvoke = AsyncMock(return_value="Search results about the topic")
        return mock

    @pytest.fixture
    def agent_context(self):
        """Create a test agent context."""
        return AgentContext.create(correlation_id="test-correlation-id")

    @pytest.fixture
    def researcher_agent(self, mock_llm, mock_search_tool):
        """Create a ResearcherAgent with mocked LLM and search tool."""
        with patch("src.agents.researcher.BaseAgent.__init__", return_value=None):
            agent = ResearcherAgent()
            agent._llm = mock_llm
            agent._name = "researcher"
            agent._description = ""
            agent._correlation_id = None
            agent._search_tool = mock_search_tool
            agent._tools = []
            return agent

    @pytest.mark.asyncio
    async def test_run_direct_parses_valid_json_response(
        self, researcher_agent, agent_context
    ):
        """Test that _run correctly parses valid JSON response in direct mode."""
        result = await researcher_agent._run_direct("Test Topic", agent_context)

        assert result.topic == "Test Topic"
        assert len(result.sources) > 0
        assert len(result.findings) > 0
        assert result.correlation_id == agent_context.correlation_id


class TestParseResponse:
    """Tests for _parse_response method."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper."""
        mock = MagicMock()
        mock.ainvoke = AsyncMock(return_value=MagicMock(content="{}"))
        return mock

    @pytest.fixture
    def researcher_agent(self, mock_llm):
        """Create a ResearcherAgent for testing parse method."""
        with patch("src.agents.researcher.BaseAgent.__init__", return_value=None):
            agent = ResearcherAgent()
            agent._llm = mock_llm
            agent._name = "researcher"
            agent._description = ""
            return agent

    def test_parse_valid_json(self, researcher_agent):
        """Test parsing valid JSON response."""
        content = '{"sources": [{"url": "http://test.com"}], "findings": ["Finding 1"]}'
        sources, findings = researcher_agent._parse_response(content)

        assert len(sources) == 1
        assert len(findings) == 1

    def test_parse_json_with_extra_content(self, researcher_agent):
        """Test parsing JSON with surrounding text."""
        content = 'Here is the response: {"sources": [], "findings": []}'
        sources, findings = researcher_agent._parse_response(content)

        assert sources == []
        assert findings == []

    def test_parse_invalid_json_uses_fallback(self, researcher_agent):
        """Test that invalid JSON falls back to treating content as finding."""
        content = "This is not JSON at all"
        sources, findings = researcher_agent._parse_response(content)

        assert len(sources) == 1
        assert len(findings) == 1
        assert findings[0] == "This is not JSON at all"

    def test_parse_empty_content(self, researcher_agent):
        """Test parsing empty content."""
        sources, findings = researcher_agent._parse_response("")
        assert sources == [{"url": "", "title": "", "date": "", "content": ""}]
        assert findings == [""]
