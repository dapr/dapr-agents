import importlib
import os
import sys

import pytest


@pytest.mark.asyncio
async def test_durable_agent_retry_config():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    if root not in sys.path:
        sys.path.insert(0, root)

    module = importlib.import_module("quickstarts.08_durable_agent_retry")
    agent = module.ResilientAgent()

    assert agent.retry_policy.max_attempts == 3
    assert agent.retry_policy.initial_interval_ms == 1000
    assert agent.retry_policy.backoff_coefficient == 2.0
