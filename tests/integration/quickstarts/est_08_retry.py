import pytest
from dapr_agents.agents import DurableAgent
from quickstarts.08_durable_agent_retry import ResilientAgent

@pytest.mark.asyncio
async def test_durable_agent_retry_config():
    """Verifica la configurazione della policy di retry."""
    agent = ResilientAgent()
    assert agent.retry_policy.max_attempts == 3
    assert agent.retry_policy.initial_interval_ms == 1000
    assert agent.retry_policy.backoff_coefficient == 2.0
