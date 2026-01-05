#!/usr/bin/env python3
import logging

from agent_tools import tools
from dotenv import load_dotenv

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    pubsub = AgentPubSubConfig(
        pubsub_name="messagepubsub",
        agent_topic="weather.requests",
        broadcast_topic="agents.broadcast",
    )
    state = AgentStateConfig(
        store=StateStoreService(store_name="workflowstatestore"),
    )
    registry = AgentRegistryConfig(
        store=StateStoreService(store_name="registrystatestore"),
        team_name="default",
    )
    execution = AgentExecutionConfig(max_iterations=3)
    memory = AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="conversationstore",
            session_id="weather-session",
        )
    )

    weather_agent = DurableAgent(
        role="Weather Assistant",
        name="Stevie",
        goal="Help humans get weather and location info using smart tools.",
        instructions=[
            "Respond clearly and helpfully to weather-related questions.",
            "Use tools when appropriate to fetch weather data.",
        ],
        tools=tools,
        pubsub=pubsub,
        registry=registry,
        execution=execution,
        memory=memory,
        state=state,
    )

    runner = AgentRunner()
    try:
        runner.serve(weather_agent, port=8001)
    finally:
        runner.shutdown(weather_agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
