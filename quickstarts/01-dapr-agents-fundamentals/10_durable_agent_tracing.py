import asyncio
import logging

from dapr_agents.llm import DaprChatClient

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentStateConfig,
    AgentObservabilityConfig,
    AgentTracingExporter,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner
from function_tools import slow_weather_func


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    weather_agent = DurableAgent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[slow_weather_func],
        llm=DaprChatClient(component_name="llm-provider"),
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="conversation-statestore",
                session_id="08-durable-agent-trace",
            )
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflow-statestore"),
        ),
        agent_observability=AgentObservabilityConfig(
            enabled=True,
            tracing_enabled=True,
            tracing_exporter=AgentTracingExporter.ZIPKIN,
            endpoint="http://localhost:9411/api/v2/spans",
        ),
    )

    runner = AgentRunner()
    try:
        prompt = "What is the weather in London?"
        await runner.run(weather_agent, payload={"task": prompt})
    finally:
        runner.shutdown(weather_agent)

        exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
