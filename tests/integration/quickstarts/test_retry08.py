import importlib
import pytest

# Since the module name starts with a number ('08_...'), it's not a valid 
# Python identifier. We must use importlib to import it dynamically.
# Change "quick_starts" to "quickstarts"
module_name = "quickstarts.08_durable_agent_retry"

# Extract the required objects from the dynamically imported module
perform_task = quickstart_module.perform_task
agent = quickstart_module.agent

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
