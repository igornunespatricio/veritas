"""Unit tests for WriterAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.writer import WriterAgent
from src.domain.events import ReportWritten, SynthesisCompleted
from src.domain.interfaces import AgentContext


class TestWriterAgent:
    """Tests for WriterAgent class."""

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
    def writer_agent(self, mock_llm):
        """Create a WriterAgent with mocked LLM."""
        with patch("src.agents.writer.BaseAgent.__init__", return_value=None):
            agent = WriterAgent()
            agent._llm = mock_llm
            agent._name = "writer"
            agent._description = "Produces polished, structured reports"
            agent._correlation_id = None
            return agent

    def test_agent_name(self, writer_agent):
        """Test that agent name is 'writer'."""
        assert writer_agent.name == "writer"

    def test_agent_description(self, writer_agent):
        """Test agent description."""
        assert "produces" in writer_agent.description.lower()
        assert "reports" in writer_agent.description.lower()

    @pytest.mark.asyncio
    async def test_validate_input_accepts_dict_with_synthesis(self, writer_agent):
        """Test that validate_input accepts dict with 'synthesis' key."""
        synthesis = SynthesisCompleted.create(
            insights=["Insight 1"],
            resolved_contradictions=[],
        )
        assert await writer_agent.validate_input({"synthesis": synthesis}) is True

    @pytest.mark.asyncio
    async def test_validate_input_rejects_dict_missing_synthesis(self, writer_agent):
        """Test that validate_input rejects dict missing 'synthesis' key."""
        assert await writer_agent.validate_input({}) is False
        assert await writer_agent.validate_input({"format": "markdown"}) is False

    @pytest.mark.asyncio
    async def test_validate_input_rejects_other_types(self, writer_agent):
        """Test that validate_input rejects other input types."""
        assert await writer_agent.validate_input("string") is False
        assert await writer_agent.validate_input(123) is False
        assert await writer_agent.validate_input(None) is False

    @pytest.mark.asyncio
    async def test_write_report_method_exists(self, writer_agent):
        """Test that write_report convenience method exists."""
        assert hasattr(writer_agent, "write_report")
        assert callable(writer_agent.write_report)


class TestWriterAgentRun:
    """Tests for WriterAgent._run method."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper with JSON response."""
        # Create inner LLM mock that supports bind_tools
        inner_llm = MagicMock()
        inner_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                tool_calls=[
                    {
                        "name": "format_report",
                        "args": {
                            "title": "Annual Technology Report 2024",
                            "content": "# Report\n\nThis is the report content.",
                            "format": "markdown",
                        },
                    }
                ]
            )
        )
        inner_llm.bind_tools = MagicMock(return_value=inner_llm)

        # Create outer wrapper mock (ResilientLLMWrapper structure)
        mock = MagicMock()
        mock.llm = inner_llm
        mock.ainvoke = inner_llm.ainvoke
        return mock

    @pytest.fixture
    def agent_context(self):
        """Create a test agent context."""
        return AgentContext.create(correlation_id="test-correlation-id")

    @pytest.fixture
    def writer_agent(self, mock_llm):
        """Create a WriterAgent with mocked LLM."""
        with patch("src.agents.writer.BaseAgent.__init__", return_value=None):
            agent = WriterAgent()
            agent._llm = mock_llm
            agent._name = "writer"
            agent._description = ""
            agent._correlation_id = None
            return agent

    @pytest.mark.asyncio
    async def test_run_parses_valid_json_response(self, writer_agent, agent_context):
        """Test that _run correctly parses valid JSON response."""
        synthesis = SynthesisCompleted.create(
            insights=["AI is advancing rapidly", "Automation is growing"],
            resolved_contradictions=[
                {"issue": "Job impact", "resolution": "Balanced view"}
            ],
        )

        result = await writer_agent._run(
            {"synthesis": synthesis, "format": "markdown"}, agent_context
        )

        assert result.title == "Annual Technology Report 2024"
        assert "Report" in result.content
        assert result.format == "markdown"
        assert result.correlation_id == agent_context.correlation_id

    @pytest.mark.asyncio
    async def test_run_handles_invalid_json_with_fallback(self, agent_context):
        """Test that _run handles invalid JSON response gracefully."""
        # Create fresh mock with invalid JSON response
        inner_llm = MagicMock()
        inner_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="No JSON response")
        )
        inner_llm.bind_tools = MagicMock(return_value=inner_llm)

        mock_llm = MagicMock()
        mock_llm.llm = inner_llm
        mock_llm.ainvoke = inner_llm.ainvoke

        with patch("src.agents.writer.BaseAgent.__init__", return_value=None):
            agent = WriterAgent()
            agent._llm = mock_llm
            agent._name = "writer"
            agent._description = ""
            agent._correlation_id = None

            synthesis = SynthesisCompleted.create(
                insights=["Insight"],
                resolved_contradictions=[],
            )

            result = await agent._run(
                {"synthesis": synthesis, "format": "markdown"}, agent_context
            )

            # Should use fallback handling
            assert result.title == "Research Report"

    @pytest.mark.asyncio
    async def test_run_with_plain_format(self, agent_context):
        """Test that _run handles plain text format correctly."""
        # Create fresh mock with plain text response
        inner_llm = MagicMock()
        inner_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                tool_calls=[
                    {
                        "name": "format_report",
                        "args": {
                            "title": "Plain Text Report",
                            "content": "Report content here",
                            "format": "plain",
                        },
                    }
                ]
            )
        )
        inner_llm.bind_tools = MagicMock(return_value=inner_llm)

        mock_llm = MagicMock()
        mock_llm.llm = inner_llm
        mock_llm.ainvoke = inner_llm.ainvoke

        with patch("src.agents.writer.BaseAgent.__init__", return_value=None):
            agent = WriterAgent()
            agent._llm = mock_llm
            agent._name = "writer"
            agent._description = ""
            agent._correlation_id = None

            synthesis = SynthesisCompleted.create(
                insights=["Key finding"],
                resolved_contradictions=[],
            )

            result = await agent._run(
                {"synthesis": synthesis, "format": "plain"}, agent_context
            )

            assert result.format == "plain"


class TestWriterAgentIntegration:
    """Integration tests for WriterAgent with full execute flow."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper with proper nested structure."""
        # Create inner LLM mock that supports bind_tools
        inner_llm = MagicMock()
        inner_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                tool_calls=[
                    {
                        "name": "format_report",
                        "args": {
                            "title": "Q4 Market Analysis",
                            "content": "# Q4 Market Analysis\n\n## Executive Summary\n\nStrong performance in key sectors.",
                            "format": "markdown",
                        },
                    }
                ]
            )
        )

        # bind_tools returns the same mock (for chaining)
        inner_llm.bind_tools = MagicMock(return_value=inner_llm)

        # Create outer wrapper mock (ResilientLLMWrapper structure)
        mock = MagicMock()
        mock.llm = inner_llm
        mock.ainvoke = inner_llm.ainvoke  # Also support direct access
        # bind_tools on outer mock should delegate to inner_llm
        mock.bind_tools = MagicMock(return_value=inner_llm)
        return mock

    @pytest.fixture
    def agent_context(self):
        """Create a test agent context."""
        return AgentContext.create(correlation_id="integration-test-id")

    @pytest.mark.asyncio
    async def test_full_write_report_flow(self, mock_llm, agent_context):
        """Test complete write report flow from execute to result."""
        with patch("src.agents.writer.BaseAgent.__init__", return_value=None):
            agent = WriterAgent()
            agent._llm = mock_llm
            agent._name = "writer"
            agent._description = ""
            agent._correlation_id = None

            synthesis = SynthesisCompleted.create(
                insights=[
                    "Tech sector grew 15%",
                    "AI adoption increased 40%",
                    "Remote work became permanent",
                ],
                resolved_contradictions=[
                    {
                        "issue": "Work-life balance concerns",
                        "resolution": "Both productivity and wellbeing improved",
                    }
                ],
            )

            result = await agent.write_report(
                synthesis, format="markdown", context=agent_context
            )

            # Verify result
            assert isinstance(result, ReportWritten)
            assert result.title == "Q4 Market Analysis"
            assert "Executive Summary" in result.content
            assert result.format == "markdown"
            assert result.correlation_id == "integration-test-id"

    @pytest.mark.asyncio
    async def test_write_report_with_default_format(self, mock_llm, agent_context):
        """Test write report uses markdown as default format."""
        with patch("src.agents.writer.BaseAgent.__init__", return_value=None):
            agent = WriterAgent()
            agent._llm = mock_llm
            agent._name = "writer"
            agent._description = ""
            agent._correlation_id = None

            synthesis = SynthesisCompleted.create(
                insights=["Finding"],
                resolved_contradictions=[],
            )

            # Call without format parameter
            result = await agent.execute({"synthesis": synthesis}, agent_context)

            assert isinstance(result, ReportWritten)
            assert result.format == "markdown"  # Default

    @pytest.mark.asyncio
    async def test_write_report_with_html_format(self, agent_context):
        """Test write report with HTML format."""
        # Create a fresh mock for this test with HTML response
        inner_llm = MagicMock()
        inner_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                tool_calls=[
                    {
                        "name": "format_report",
                        "args": {
                            "title": "HTML Report",
                            "content": "<h1>HTML Report</h1>",
                            "format": "html",
                        },
                    }
                ]
            )
        )
        inner_llm.bind_tools = MagicMock(return_value=inner_llm)

        mock_llm = MagicMock()
        mock_llm.llm = inner_llm
        mock_llm.ainvoke = inner_llm.ainvoke

        with patch("src.agents.writer.BaseAgent.__init__", return_value=None):
            agent = WriterAgent()
            agent._llm = mock_llm
            agent._name = "writer"
            agent._description = ""
            agent._correlation_id = None

            synthesis = SynthesisCompleted.create(
                insights=["Insight"],
                resolved_contradictions=[],
            )

            result = await agent.write_report(
                synthesis, format="html", context=agent_context
            )

            assert result.format == "html"
            assert "<h1>" in result.content


class TestWriterPrompt:
    """Tests for WriterAgent system prompt."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper."""
        mock = MagicMock()
        mock.ainvoke = AsyncMock(return_value=MagicMock(content="{}"))
        return mock

    @pytest.fixture
    def writer_agent(self, mock_llm):
        """Create a WriterAgent for testing prompt."""
        with patch("src.agents.writer.BaseAgent.__init__", return_value=None):
            agent = WriterAgent()
            agent._llm = mock_llm
            agent._name = "writer"
            agent._description = ""
            return agent

    def test_system_prompt_contains_writing_instructions(self, writer_agent):
        """Test that system prompt contains writing instructions."""
        prompt = writer_agent.WRITER_SYSTEM_PROMPT
        assert "expert technical writer" in prompt.lower()
        assert "well-structured report" in prompt.lower()
        assert "clear headings" in prompt.lower()

    def test_system_prompt_mentions_citations(self, writer_agent):
        """Test that system prompt mentions citations."""
        prompt = writer_agent.WRITER_SYSTEM_PROMPT
        assert "citations" in prompt.lower() or "sources" in prompt.lower()
