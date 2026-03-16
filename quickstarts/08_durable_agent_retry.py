import pytest
import importlib

# We use import_module because the filename starts with a number (08), 
# which is not allowed in standard Python import syntax.
retry_module = importlib.import_module("guide_rapide.08_durable_agent_retry")
perform_task = retry_module.perform_task
agent = retry_module.agent

@pytest.mark.asyncio
async def test_durable_agent_retry_logic():
    """
    Tests the retry logic implementation in a functional way.
    """
    result = await perform_task()
    
    assert result == "Task Success"
    assert agent.name == "ResilientAgent"
    assert agent.retry_policy.max_attempts == 3
