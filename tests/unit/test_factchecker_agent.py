"""Unit tests for FactCheckerAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.factchecker import ClaimStatus, FactCheckerAgent
from src.domain.events import FactCheckCompleted, ResearchCompleted
from src.domain.interfaces import AgentContext


class TestFactCheckerAgent:
    """Tests for FactCheckerAgent class."""

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
    def fact_check_agent(self, mock_llm):
        """Create a FactCheckerAgent with mocked LLM."""
        with patch("src.agents.factchecker.BaseAgent.__init__", return_value=None):
            agent = FactCheckerAgent()
            agent._llm = mock_llm
            agent._name = "fact_checker"
            agent._description = "Verifies claims and assigns confidence scores"
            agent._correlation_id = None
            return agent

    def test_agent_name(self, fact_check_agent):
        """Test that agent name is 'fact_checker'."""
        assert fact_check_agent.name == "fact_checker"

    def test_agent_description(self, fact_check_agent):
        """Test agent description."""
        assert "verifies" in fact_check_agent.description.lower()
        assert "confidence" in fact_check_agent.description.lower()

    @pytest.mark.asyncio
    async def test_validate_input_accepts_research_completed(self, fact_check_agent):
        """Test that validate_input accepts ResearchCompleted events."""
        research = ResearchCompleted.create(
            topic="Test Topic",
            sources=[{"url": "http://example.com", "title": "Test"}],
            findings=["Finding 1", "Finding 2"],
        )
        assert await fact_check_agent.validate_input(research) is True

    @pytest.mark.asyncio
    async def test_validate_input_rejects_other_types(self, fact_check_agent):
        """Test that validate_input rejects non-ResearchCompleted inputs."""
        assert await fact_check_agent.validate_input("string") is False
        assert await fact_check_agent.validate_input(123) is False
        assert await fact_check_agent.validate_input(None) is False

    @pytest.mark.asyncio
    async def test_verify_claims_method_exists(self, fact_check_agent):
        """Test that verify_claims convenience method exists."""
        assert hasattr(fact_check_agent, "verify_claims")
        assert callable(fact_check_agent.verify_claims)


class TestFactCheckerAgentRun:
    """Tests for FactCheckerAgent._run method."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper with JSON response."""

        def mock_ainvoke(messages):
            # Simulate LLM response with JSON
            return MagicMock(
                content='{"claims": [{"text": "Claim 1", "status": "verified"}], "verified_claims": [{"text": "Claim 1", "status": "verified"}], "confidence_scores": {"Claim 1": 0.9}}'
            )

        mock = MagicMock()
        mock.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        return mock

    @pytest.fixture
    def agent_context(self):
        """Create a test agent context."""
        return AgentContext.create(correlation_id="test-correlation-id")

    @pytest.fixture
    def fact_check_agent(self, mock_llm):
        """Create a FactCheckerAgent with mocked LLM."""
        with patch("src.agents.factchecker.BaseAgent.__init__", return_value=None):
            agent = FactCheckerAgent()
            agent._llm = mock_llm
            agent._name = "fact_checker"
            agent._description = ""
            agent._correlation_id = None
            return agent

    @pytest.mark.asyncio
    async def test_run_parses_valid_json_response(
        self, fact_check_agent, agent_context
    ):
        """Test that _run correctly parses valid JSON response."""
        research = ResearchCompleted.create(
            topic="Climate Change",
            sources=[{"url": "http://example.com", "title": "Test"}],
            findings=["Global temperatures are rising", "CO2 levels increasing"],
        )

        result = await fact_check_agent._run(research, agent_context)

        assert len(result.claims) == 1
        assert result.claims[0]["status"] == "verified"
        assert result.confidence_scores.get("Claim 1") == 0.9
        assert result.correlation_id == agent_context.correlation_id

    @pytest.mark.asyncio
    async def test_run_handles_invalid_json_with_fallback(
        self, mock_llm, agent_context
    ):
        """Test that _run handles invalid JSON response gracefully."""
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="No JSON response from LLM")
        )

        with patch("src.agents.factchecker.BaseAgent.__init__", return_value=None):
            agent = FactCheckerAgent()
            agent._llm = mock_llm
            agent._name = "fact_checker"
            agent._description = ""
            agent._correlation_id = None

            research = ResearchCompleted.create(
                topic="Test",
                sources=[{"url": "http://example.com", "title": "Test"}],
                findings=["Finding 1"],
            )

            result = await agent._run(research, agent_context)

            # Should use fallback handling
            assert len(result.claims) == 1
            assert result.claims[0]["status"] == "unverified"


class TestNormalizeClaimStatuses:
    """Tests for claim status normalization."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper."""
        mock = MagicMock()
        mock.ainvoke = AsyncMock(return_value=MagicMock(content="{}"))
        return mock

    @pytest.fixture
    def fact_check_agent(self, mock_llm):
        """Create a FactCheckerAgent for testing normalization."""
        with patch("src.agents.factchecker.BaseAgent.__init__", return_value=None):
            agent = FactCheckerAgent()
            agent._llm = mock_llm
            agent._name = "fact_checker"
            agent._description = ""
            return agent

    def test_normalize_verified_status(self, fact_check_agent):
        """Test normalization of verified status."""
        claims = [{"text": "Test", "status": "VERIFIED"}]
        result = fact_check_agent._normalize_claim_statuses(claims)
        assert result[0]["status"] == "verified"

    def test_normalize_partially_verified_status(self, fact_check_agent):
        """Test normalization of partially verified status."""
        claims = [{"text": "Test", "status": "Partially Verified"}]
        result = fact_check_agent._normalize_claim_statuses(claims)
        assert result[0]["status"] == "partially_verified"

    def test_normalize_disputed_status(self, fact_check_agent):
        """Test normalization of disputed status."""
        claims = [{"text": "Test", "status": "DISPUTED"}]
        result = fact_check_agent._normalize_claim_statuses(claims)
        assert result[0]["status"] == "disputed"

    def test_normalize_unverified_status(self, fact_check_agent):
        """Test normalization of unverified status."""
        claims = [{"text": "Test", "status": "UNVERIFIED"}]
        result = fact_check_agent._normalize_claim_statuses(claims)
        assert result[0]["status"] == "unverified"

    def test_normalize_invalid_status_defaults_to_unverified(self, fact_check_agent):
        """Test that invalid status defaults to unverified."""
        claims = [{"text": "Test", "status": "INVALID_STATUS"}]
        result = fact_check_agent._normalize_claim_statuses(claims)
        assert result[0]["status"] == "unverified"

    def test_normalize_empty_status_defaults_to_unverified(self, fact_check_agent):
        """Test that empty status defaults to unverified."""
        claims = [{"text": "Test", "status": ""}]
        result = fact_check_agent._normalize_claim_statuses(claims)
        assert result[0]["status"] == "unverified"

    def test_normalize_preserves_other_fields(self, fact_check_agent):
        """Test that normalization preserves other claim fields."""
        claims = [
            {
                "text": "Test",
                "status": "verified",
                "source": "http://example.com",
                "confidence": 0.9,
            }
        ]
        result = fact_check_agent._normalize_claim_statuses(claims)
        assert result[0]["source"] == "http://example.com"
        assert result[0]["confidence"] == 0.9


class TestClaimStatus:
    """Tests for ClaimStatus constants."""

    def test_claim_status_constants(self):
        """Test that claim status constants are defined correctly."""
        assert ClaimStatus.VERIFIED == "verified"
        assert ClaimStatus.PARTIALLY_VERIFIED == "partially_verified"
        assert ClaimStatus.DISPUTED == "disputed"
        assert ClaimStatus.UNVERIFIED == "unverified"

    def test_all_statuses_defined(self):
        """Test that all required statuses are defined."""
        expected_statuses = {"verified", "partially_verified", "disputed", "unverified"}
        actual_statuses = {
            ClaimStatus.VERIFIED,
            ClaimStatus.PARTIALLY_VERIFIED,
            ClaimStatus.DISPUTED,
            ClaimStatus.UNVERIFIED,
        }
        assert actual_statuses == expected_statuses


class TestFactCheckerAgentIntegration:
    """Integration tests for FactCheckerAgent with full execute flow."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock resilient LLM wrapper."""

        def mock_ainvoke(messages):
            return MagicMock(
                content='{"claims": [{"text": "Earth is round", "status": "verified"}, {"text": "Water is wet", "status": "partially_verified"}], "verified_claims": [{"text": "Earth is round", "status": "verified"}], "confidence_scores": {"Earth is round": 0.95, "Water is wet": 0.7}}'
            )

        mock = MagicMock()
        mock.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        return mock

    @pytest.fixture
    def agent_context(self):
        """Create a test agent context."""
        return AgentContext.create(correlation_id="integration-test-id")

    @pytest.mark.asyncio
    async def test_full_verify_claims_flow(self, mock_llm, agent_context):
        """Test complete verification flow from execute to result."""
        with patch("src.agents.factchecker.BaseAgent.__init__", return_value=None):
            agent = FactCheckerAgent()
            agent._llm = mock_llm
            agent._name = "fact_checker"
            agent._description = ""
            agent._correlation_id = None

            research = ResearchCompleted.create(
                topic="Basic Facts",
                sources=[
                    {"url": "http://example.com/1", "title": "Source 1"},
                    {"url": "http://example.com/2", "title": "Source 2"},
                ],
                findings=["Earth is round", "Water is wet"],
            )

            result = await agent.execute(research, agent_context)

            # Verify result
            assert isinstance(result, FactCheckCompleted)
            assert len(result.claims) == 2
            assert result.correlation_id == "integration-test-id"

    @pytest.mark.asyncio
    async def test_verify_claims_with_custom_claims(self, mock_llm, agent_context):
        """Test verify_claims convenience method with custom inputs."""
        with patch("src.agents.factchecker.BaseAgent.__init__", return_value=None):
            agent = FactCheckerAgent()
            agent._llm = mock_llm
            agent._name = "fact_checker"
            agent._description = ""
            agent._correlation_id = None

            claims = ["Claim 1", "Claim 2"]
            sources = [{"url": "http://test.com", "title": "Test"}]

            result = await agent.verify_claims(claims, sources, agent_context)

            assert isinstance(result, FactCheckCompleted)
            assert len(result.claims) == 2
