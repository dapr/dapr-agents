"""Tests for the RandomOrchestrator."""
import pytest
from unittest.mock import MagicMock, patch
from dapr_agents.workflow.orchestrators import RandomOrchestrator


@pytest.fixture
def orchestrator_config():
    """Fixture to provide common orchestrator configuration."""
    return {
        "name": "test_orchestrator",
        "message_bus_name": "test-message-bus",
        "state_store_name": "test-state-store",
        "agents_registry_store_name": "test-registry-store",
    }


@pytest.mark.asyncio
async def test_random_orchestrator_initialization(orchestrator_config):
    """Test that RandomOrchestrator can be initialized."""
    with patch(
        "dapr_agents.workflow.orchestrators.random.OrchestratorWorkflowBase.model_post_init"
    ) as mock_init, patch(
        "dapr_agents.workflow.agentic.AgenticWorkflow.model_post_init"
    ), patch(
        "dapr_agents.workflow.agentic.AgenticWorkflow._is_dapr_available"
    ) as mock_dapr_check, patch(
        "dapr_agents.workflow.agentic.AgenticWorkflow._state_store_client"
    ) as mock_state_store, patch(
        "dapr_agents.workflow.agentic.AgenticWorkflow._dapr_client"
    ) as mock_dapr_client:
        mock_dapr_check.return_value = True
        mock_state_store.return_value = MagicMock()
        mock_dapr_client.return_value = MagicMock()
        orchestrator = RandomOrchestrator(**orchestrator_config)
        assert orchestrator.name == "test_orchestrator"
        assert orchestrator._workflow_name == "RandomWorkflow"
        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_process_input(orchestrator_config):
    """Test the process_input task."""
    with patch(
        "dapr_agents.workflow.orchestrators.random.OrchestratorWorkflowBase.model_post_init"
    ), patch("dapr_agents.workflow.agentic.AgenticWorkflow.model_post_init"), patch(
        "dapr_agents.workflow.agentic.AgenticWorkflow._is_dapr_available"
    ) as mock_dapr_check, patch(
        "dapr_agents.workflow.agentic.AgenticWorkflow._state_store_client"
    ) as mock_state_store, patch(
        "dapr_agents.workflow.agentic.AgenticWorkflow._dapr_client"
    ) as mock_dapr_client:
        mock_dapr_check.return_value = True
        mock_state_store.return_value = MagicMock()
        mock_dapr_client.return_value = MagicMock()
        orchestrator = RandomOrchestrator(**orchestrator_config)
        task = "test task"
        result = await orchestrator.process_input(task)

        assert result["role"] == "user"
        assert result["name"] == "test_orchestrator"
        assert result["content"] == task


@pytest.mark.asyncio
async def test_select_random_speaker(orchestrator_config):
    """Test the select_random_speaker task."""
    with patch(
        "dapr_agents.workflow.orchestrators.random.OrchestratorWorkflowBase.model_post_init"
    ), patch("dapr_agents.workflow.agentic.AgenticWorkflow.model_post_init"), patch(
        "dapr_agents.workflow.agentic.AgenticWorkflow._is_dapr_available"
    ) as mock_dapr_check, patch(
        "dapr_agents.workflow.agentic.AgenticWorkflow._state_store_client"
    ) as mock_state_store, patch(
        "dapr_agents.workflow.agentic.AgenticWorkflow._dapr_client"
    ) as mock_dapr_client, patch.object(
        RandomOrchestrator,
        "get_agents_metadata",
        return_value={"agent1": {"name": "agent1"}, "agent2": {"name": "agent2"}},
    ):
        mock_dapr_check.return_value = True
        mock_state_store.return_value = MagicMock()
        mock_dapr_client.return_value = MagicMock()
        orchestrator = RandomOrchestrator(**orchestrator_config)

        speaker = orchestrator.select_random_speaker(iteration=1)
        assert speaker in ["agent1", "agent2"]
        assert orchestrator.current_speaker == speaker
