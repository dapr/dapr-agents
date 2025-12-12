import asyncio

from dapr_agents.llm import DaprChatClient

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentStateConfig,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.workflow.utils.core import wait_for_shutdown
from function_tools import slow_weather_func


def main() -> None:
    weather_agent = DurableAgent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[slow_weather_func],
        llm=DaprChatClient(component_name="llm-provider"),
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="conversation-statestore",
                session_id="05-durable-agent-sub",
            )
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflow-statestore"),
        ),
        pubsub=AgentPubSubConfig(
            pubsub_name="message-pubsub",
            agent_topic="weather.requests",
            broadcast_topic="agents.broadcast",
        ),
    )

    runner = AgentRunner()
    try:
        runner.subscribe(weather_agent)
        asyncio.run(wait_for_shutdown())
    finally:
        runner.shutdown(weather_agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
