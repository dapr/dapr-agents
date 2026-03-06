"""Quickstart 12 — Durable Agent with Retry Policy.

Demonstrates how to configure a WorkflowRetryPolicy on a DurableAgent so that
every workflow activity (LLM calls, tool calls, state writes) is automatically
retried with exponential backoff on transient failures.

The five knobs exposed by WorkflowRetryPolicy are:

    max_attempts             – total number of attempts (1 = no retries)
    initial_backoff_seconds  – wait before the first retry
    max_backoff_seconds      – upper bound on the backoff interval
    backoff_multiplier       – multiplier applied after each retry
    retry_timeout            – optional overall timeout across all retries

You can also override max_attempts at deploy time by setting the environment
variable DAPR_API_MAX_RETRIES, which takes precedence over the value in code.
"""

import asyncio
import logging

from dapr_agents.llm import DaprChatClient

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentStateConfig,
    WorkflowRetryPolicy,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner
from function_tools import slow_weather_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    # Define an explicit retry policy.
    # These values are intentionally non-default so every field is visible in
    # the quickstart.
    retry = WorkflowRetryPolicy(
        max_attempts=3,                # retry up to 3 times per activity
        initial_backoff_seconds=2,     # first retry after 2 s
        max_backoff_seconds=30,        # cap backoff at 30 s
        backoff_multiplier=2.0,        # double the wait each retry
        retry_timeout=120,             # give up after 120 s total
    )

    weather_agent = DurableAgent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[slow_weather_func],
        # Configure this agent to use the Dapr Conversation API.
        llm=DaprChatClient(component_name="llm-provider"),
        # Configure the agent to use Dapr State Store for conversation history.
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="agent-memory",
            )
        ),
        # This is where the execution state is stored.
        state=AgentStateConfig(
            store=StateStoreService(store_name="agent-workflow"),
        ),
        # Attach the retry policy — every workflow activity now uses it.
        retry_policy=retry,
    )

    logger.info(
        "RetryPolicy configured: max_attempts=%s, backoff=%s→%ss (×%.1f), timeout=%ss",
        retry.max_attempts,
        retry.initial_backoff_seconds,
        retry.max_backoff_seconds,
        retry.backoff_multiplier,
        retry.retry_timeout,
    )

    runner = AgentRunner()
    try:
        prompt = "What is the weather in London?"
        await runner.run(weather_agent, payload={"task": prompt})
    finally:
        runner.shutdown(weather_agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
