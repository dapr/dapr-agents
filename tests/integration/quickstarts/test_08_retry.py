import pytest
from guide_rapide.08_durable_agent_retry import perform_task, agent

@pytest.mark.asyncio
async def test_durable_agent_retry_logic():
    """
    Tests the retry logic implementation in a functional way.
    Verifies that the agent and task execution are correctly configured.
    """
    # Execute the task from the quickstart
    result = await perform_task()
    
    # Assertions to ensure everything is wired correctly
    assert result == "Task Success"
    assert agent.name == "ResilientAgent"
    assert agent.retry_policy.max_attempts == 3
