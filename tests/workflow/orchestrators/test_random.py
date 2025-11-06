"""Tests for the RandomOrchestrator."""
import pytest
from unittest.mock import MagicMock, patch
from dapr_agents.agents.orchestrators import RandomOrchestrator
from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentStateConfig,
    AgentRegistryConfig,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService


@pytest.fixture
def orchestrator_config():
    """Fixture to provide common orchestrator configuration."""
    return {
        "name": "test_orchestrator",
        "pubsub": AgentPubSubConfig(pubsub_name="test-message-bus"),
        "state": AgentStateConfig(
            store=StateStoreService(store_name="test-state-store")
        ),
        "registry": AgentRegistryConfig(
            store=StateStoreService(store_name="test-registry-store")
        ),
    }


def test_random_orchestrator_initialization(orchestrator_config):
    """Test that RandomOrchestrator can be initialized."""
    with patch("dapr.ext.workflow.WorkflowRuntime") as mock_runtime:
        mock_runtime.return_value = MagicMock()
        orchestrator = RandomOrchestrator(**orchestrator_config)
        assert orchestrator.name == "test_orchestrator"


@pytest.mark.asyncio
async def test_process_input(orchestrator_config):
    """Test the _process_input_activity task."""
    with patch("dapr.ext.workflow.WorkflowRuntime") as mock_runtime:
        mock_runtime.return_value = MagicMock()
        orchestrator = RandomOrchestrator(**orchestrator_config)

        # Mock the activity context
        mock_ctx = MagicMock()
        task = "test task"
        result = orchestrator._process_input_activity(mock_ctx, {"task": task})

        assert result["role"] == "user"
        assert result["name"] == "user"
        assert result["content"] == task


def test_select_random_speaker(orchestrator_config):
    """Test the _select_random_speaker_activity task."""
    with patch("dapr.ext.workflow.WorkflowRuntime") as mock_runtime, patch.object(
        RandomOrchestrator,
        "list_team_agents",
        return_value={"agent1": {"name": "agent1"}, "agent2": {"name": "agent2"}},
    ):
        mock_runtime.return_value = MagicMock()
        orchestrator = RandomOrchestrator(**orchestrator_config)

        # Mock the activity context
        mock_ctx = MagicMock()
        speaker = orchestrator._select_random_speaker_activity(mock_ctx)

        assert speaker in ["agent1", "agent2"]
