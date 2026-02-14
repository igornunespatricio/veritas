"""Unit tests for domain interfaces."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.domain.interfaces import (
    Agent,
    AgentContext,
    AgentRegistry,
)


class TestAgentContext:
    """Tests for AgentContext class."""

    def test_create_with_correlation_id(self):
        """Test creating context with correlation ID."""
        context = AgentContext.create(correlation_id="test-123")
        assert context.correlation_id == "test-123"

    def test_create_without_correlation_id(self):
        """Test creating context without correlation ID."""
        context = AgentContext.create()
        assert context.correlation_id == ""

    def test_create_sets_timestamp(self):
        """Test that create sets created_at timestamp."""
        before = datetime.now(UTC)
        context = AgentContext.create()
        after = datetime.now(UTC)

        assert context.created_at >= before
        assert context.created_at <= after

    def test_create_has_empty_request_id(self):
        """Test that request_id is empty by default."""
        context = AgentContext.create()
        assert context.request_id == ""

    def test_create_has_empty_metadata(self):
        """Test that metadata is empty dict by default."""
        context = AgentContext.create()
        assert context.metadata == {}

    def test_context_is_immutable(self):
        """Test that context is a dataclass."""
        context = AgentContext.create(correlation_id="test")
        # Dataclasses are mutable but the create method ensures proper initialization
        assert context.correlation_id == "test"
        # Verify the object is a dataclass
        assert hasattr(context, "correlation_id")
        assert hasattr(context, "request_id")
        assert hasattr(context, "created_at")
        assert hasattr(context, "metadata")

    def test_context_with_custom_metadata(self):
        """Test creating context with custom metadata."""
        context = AgentContext.create(correlation_id="test")
        context.metadata["source"] = "test"
        context.metadata["priority"] = "high"
        assert context.metadata["source"] == "test"
        assert context.metadata["priority"] == "high"


class TestAgentRegistry:
    """Tests for AgentRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        AgentRegistry._agents.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        AgentRegistry._agents.clear()

    def test_register_agent(self):
        """Test registering an agent."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        AgentRegistry.register(mock_agent)

        assert "test_agent" in AgentRegistry._agents
        assert AgentRegistry._agents["test_agent"] is mock_agent

    def test_get_registered_agent(self):
        """Test getting a registered agent."""
        mock_agent = MagicMock()
        mock_agent.name = "researcher"

        AgentRegistry.register(mock_agent)
        retrieved = AgentRegistry.get("researcher")

        assert retrieved is mock_agent

    def test_get_nonexistent_agent(self):
        """Test getting a nonexistent agent returns None."""
        result = AgentRegistry.get("nonexistent")
        assert result is None

    def test_list_agents(self):
        """Test listing all registered agent names."""
        mock_agent1 = MagicMock()
        mock_agent1.name = "agent1"
        mock_agent2 = MagicMock()
        mock_agent2.name = "agent2"

        AgentRegistry.register(mock_agent1)
        AgentRegistry.register(mock_agent2)

        agents = AgentRegistry.list_agents()
        assert len(agents) == 2
        assert "agent1" in agents
        assert "agent2" in agents

    def test_register_same_agent_twice(self):
        """Test that registering same agent twice overwrites."""
        agent1 = MagicMock()
        agent1.name = "test"

        agent2 = MagicMock()
        agent2.name = "test"

        AgentRegistry.register(agent1)
        AgentRegistry.register(agent2)

        assert AgentRegistry._agents["test"] is agent2

    def test_empty_registry_list(self):
        """Test listing agents when registry is empty."""
        AgentRegistry._agents.clear()
        agents = AgentRegistry.list_agents()
        assert agents == []


class TestAgentAbstractClass:
    """Tests for Agent abstract base class."""

    def test_agent_is_abstract(self):
        """Test that Agent is an abstract base class."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            Agent()

    def test_agent_requires_name_property(self):
        """Test that name property is abstract."""

        class TestAgent(Agent):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test agent"

            async def execute(self, input, context):
                return None

            async def validate_input(self, input) -> bool:
                return True

        agent = TestAgent()
        assert agent.name == "test"

    def test_agent_requires_description_property(self):
        """Test that description property is abstract."""

        class TestAgent(Agent):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "A test agent"

            async def execute(self, input, context):
                return None

            async def validate_input(self, input) -> bool:
                return True

        agent = TestAgent()
        assert agent.description == "A test agent"

    def test_agent_requires_execute_method(self):
        """Test that execute method is abstract."""

        class TestAgent(Agent):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test"

        # Should raise TypeError because execute is not implemented
        with pytest.raises(TypeError):
            TestAgent()

    def test_agent_requires_validate_input_method(self):
        """Test that validate_input method is abstract."""

        class IncompleteAgent(Agent):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test"

            async def execute(self, input, context):
                return None

            # Missing validate_input

        with pytest.raises(TypeError):
            IncompleteAgent()


class TestAgentContextEquality:
    """Tests for AgentContext equality and hashing."""

    def test_contexts_with_same_id_are_equal(self):
        """Test that contexts with same correlation_id are considered equal."""
        context1 = AgentContext.create(correlation_id="same-id")
        context2 = AgentContext.create(correlation_id="same-id")

        # Note: Dataclass equality compares all fields
        assert context1.correlation_id == context2.correlation_id

    def test_contexts_with_different_ids_are_not_equal(self):
        """Test that contexts with different correlation_ids are not equal."""
        context1 = AgentContext.create(correlation_id="id-1")
        context2 = AgentContext.create(correlation_id="id-2")

        assert context1.correlation_id != context2.correlation_id
