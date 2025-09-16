import asyncio
from weather_tools import tools
from dapr_agents import DurableAgent
from dotenv import load_dotenv

load_dotenv()


# Wrap your async call
async def main():
    from phoenix.otel import register
    from dapr_agents.observability import DaprAgentsInstrumentor

    # Register Dapr Agents with Phoenix OpenTelemetry
    tracer_provider = register(
        project_name="dapr-weather-durable-agent",
        protocol="http/protobuf",
    )

    # Initialize Dapr Agents OpenTelemetry instrumentor
    try:
        instrumentor = DaprAgentsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    except Exception as e:
        raise

    AIAgent = DurableAgent(
        name="Steviee",
        role="Weather Assistant",
        goal="Assist Humans with weather related tasks.",
        instructions=[
            "Always answer the user's main weather question directly and clearly.",
            "If you perform any additional actions (like jumping), summarize those actions and their results.",
            "At the end, provide a concise summary that combines the weather information for all requested locations and any other actions you performed.",
        ],
        tools=tools,
        message_bus_name="messagepubsub",
        state_store_name="workflowstatestore",
        agents_registry_store_name="agentstatestore",
        history_store_name="historystore",
    )

    await AIAgent.run("What is the weather in Virginia, New York and Washington DC?")


if __name__ == "__main__":
    asyncio.run(main())
