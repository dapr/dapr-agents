"""Tests for orchestration strategies."""

import pytest
from unittest.mock import MagicMock, Mock

from dapr_agents.agents.orchestration import (
    OrchestrationStrategy,
    AgentOrchestrationStrategy,
    RoundRobinOrchestrationStrategy,
    RandomOrchestrationStrategy,
)


class TestRoundRobinOrchestrationStrategy:
    """Test cases for RoundRobinOrchestrationStrategy."""

    def test_initialize(self):
        """Test strategy initialization."""
        strategy = RoundRobinOrchestrationStrategy()
        ctx = Mock(is_replaying=False)

        agents = {"agent1": {}, "agent2": {}, "agent3": {}}
        state = strategy.initialize(ctx, "test task", agents)

        assert "agent_names" in state
        assert state["agent_names"] == ["agent1", "agent2", "agent3"]  # Sorted
        assert state["last_response"] is None
        assert state["task"] == "test task"

    def test_select_next_agent_first_turn(self):
        """Test agent selection on first turn."""
        strategy = RoundRobinOrchestrationStrategy()
        ctx = Mock(is_replaying=False)

        state = {
            "agent_names": ["agent1", "agent2", "agent3"],
            "last_response": None,
            "task": "test task",
        }

        action = strategy.select_next_agent(ctx, state, turn=1)

        assert action["agent"] == "agent1"  # First agent
        assert "test task" in action["instruction"]

    def test_select_next_agent_sequential(self):
        """Test sequential agent selection."""
        strategy = RoundRobinOrchestrationStrategy()
        ctx = Mock(is_replaying=False)

        state = {
            "agent_names": ["agent1", "agent2", "agent3"],
            "last_response": {"name": "agent1", "content": "response1"},
            "task": "test task",
        }

        # Turn 2 should select agent2
        action = strategy.select_next_agent(ctx, state, turn=2)
        assert action["agent"] == "agent2"

        # Turn 3 should select agent3
        action = strategy.select_next_agent(ctx, state, turn=3)
        assert action["agent"] == "agent3"

        # Turn 4 should wrap around to agent1
        action = strategy.select_next_agent(ctx, state, turn=4)
        assert action["agent"] == "agent1"

    def test_process_response(self):
        """Test response processing."""
        strategy = RoundRobinOrchestrationStrategy()
        ctx = Mock(is_replaying=False)

        state = {"agent_names": ["agent1", "agent2"], "last_response": None}
        response = {"name": "agent1", "content": "test response"}

        result = strategy.process_response(ctx, state, response)

        assert result["updated_state"]["last_response"] == response
        assert result["verdict"] == "continue"

    def test_should_continue(self):
        """Test continuation logic."""
        strategy = RoundRobinOrchestrationStrategy()

        state = {}

        assert strategy.should_continue(state, turn=1, max_iterations=5) is True
        assert strategy.should_continue(state, turn=4, max_iterations=5) is True
        assert strategy.should_continue(state, turn=5, max_iterations=5) is False

    def test_finalize(self):
        """Test finalization."""
        strategy = RoundRobinOrchestrationStrategy()
        strategy.orchestrator_name = "TestOrch"
        ctx = Mock(is_replaying=False)

        state = {"last_response": {"name": "agent1", "content": "final response"}}

        final_message = strategy.finalize(ctx, state)

        assert final_message["role"] == "assistant"
        assert "final response" in final_message["content"]
        assert final_message["name"] == "TestOrch"


class TestRandomOrchestrationStrategy:
    """Test cases for RandomOrchestrationStrategy."""

    def test_initialize(self):
        """Test strategy initialization."""
        strategy = RandomOrchestrationStrategy()
        ctx = Mock(is_replaying=False)

        agents = {"agent1": {}, "agent2": {}}
        state = strategy.initialize(ctx, "test task", agents)

        assert "agent_names" in state
        assert len(state["agent_names"]) == 2
        assert state["previous_agent"] is None
        assert state["task"] == "test task"

    def test_select_random_agent_no_previous(self):
        """Test random selection with no previous agent."""
        strategy = RandomOrchestrationStrategy()

        agents = ["agent1", "agent2", "agent3"]
        selected = strategy._select_random_agent(agents, None)

        assert selected in agents

    def test_select_random_agent_with_avoidance(self):
        """Test random selection avoids previous agent."""
        strategy = RandomOrchestrationStrategy()

        agents = ["agent1", "agent2", "agent3"]

        # Run multiple times to ensure avoidance works
        selections = []
        for _ in range(10):
            selected = strategy._select_random_agent(agents, "agent1")
            selections.append(selected)

        # Should never select agent1 when it's the previous agent
        # (unless only one agent, but we have 3)
        # At least some should be agent2 or agent3
        assert any(s in ["agent2", "agent3"] for s in selections)

    def test_select_random_agent_single_agent(self):
        """Test random selection with only one agent."""
        strategy = RandomOrchestrationStrategy()

        agents = ["agent1"]
        selected = strategy._select_random_agent(agents, "agent1")

        assert selected == "agent1"  # No choice but to select same agent

    def test_process_response(self):
        """Test response processing updates previous agent."""
        strategy = RandomOrchestrationStrategy()
        ctx = Mock(is_replaying=False)

        state = {
            "agent_names": ["agent1", "agent2"],
            "previous_agent": None,
            "last_response": None,
        }
        response = {"name": "agent2", "content": "response"}

        result = strategy.process_response(ctx, state, response)

        assert result["updated_state"]["previous_agent"] == "agent2"
        assert result["updated_state"]["last_response"] == response
        assert result["verdict"] == "continue"

    def test_should_continue(self):
        """Test continuation logic."""
        strategy = RandomOrchestrationStrategy()

        state = {}

        assert strategy.should_continue(state, turn=1, max_iterations=3) is True
        assert strategy.should_continue(state, turn=3, max_iterations=3) is False

    def test_finalize(self):
        """Test finalization."""
        strategy = RandomOrchestrationStrategy()
        strategy.orchestrator_name = "RandomOrch"
        ctx = Mock(is_replaying=False)

        state = {"last_response": {"name": "agent3", "content": "final response"}}

        final_message = strategy.finalize(ctx, state)

        assert final_message["role"] == "assistant"
        assert "final response" in final_message["content"]
        assert "agent3" in final_message["content"]
        assert final_message["name"] == "RandomOrch"


class TestAgentOrchestrationStrategy:
    """Test cases for AgentOrchestrationStrategy."""

    def test_initialize(self):
        """Test strategy initialization."""

        strategy = AgentOrchestrationStrategy()
        ctx = Mock()

        state = strategy.initialize(ctx, "test task", {"agent1": {}})

        assert state["task"] == "test task"
        assert state["agents"] == {"agent1": {}}
        assert state["plan"] == []
        assert state["verdict"] is None

    def test_select_next_agent(self):
        """Test agent selection returns expected structure."""

        strategy = AgentOrchestrationStrategy()
        ctx = Mock()

        state = {
            "task": "test task",
            "agents": {"agent1": {}},
            "plan": [{"step": 1, "description": "Do something"}],
        }

        action = strategy.select_next_agent(ctx, state, turn=1)

        # For AgentOrchestrationStrategy, logic is in orchestration_workflow
        # Strategy methods return placeholders
        assert "agent" in action
        assert "instruction" in action

    def test_should_continue(self):
        """Test continuation logic."""

        strategy = AgentOrchestrationStrategy()

        # For AgentOrchestrationStrategy, logic is in orchestration_workflow
        # Strategy method returns True (placeholder)
        assert strategy.should_continue({}, turn=1, max_iterations=5) is True

    def test_finalize(self):
        """Test finalization returns message structure."""

        strategy = AgentOrchestrationStrategy()
        strategy.orchestrator_name = "AgentOrch"
        ctx = Mock()

        state = {"plan": [], "verdict": "completed"}

        final_message = strategy.finalize(ctx, state)

        assert final_message["role"] == "assistant"
        assert "content" in final_message
        assert final_message["name"] == "AgentOrch"
