from dapr_agents import LLMOrchestrator
from dapr_agents.llm import DaprChatClient
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    from phoenix.otel import register
    from dapr_agents.observability import DaprAgentsInstrumentor

    # Register Dapr Agents with Phoenix OpenTelemetry
    tracer_provider = register(
        project_name="dapr-multi-agent-workflows",
        protocol="http/protobuf",
    )

    # Initialize Dapr Agents OpenTelemetry instrumentor
    try:
        instrumentor = DaprAgentsInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    except Exception as e:
        raise

    llm = DaprChatClient(component_name="openai")

    try:
        workflow_service = LLMOrchestrator(
            name="LLMOrchestrator",
            llm=llm,
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="workflow_state",
            agents_registry_store_name="agentstatestore",
            agents_registry_key="agents_registry",
            broadcast_topic_name="beacon_channel",
            max_iterations=3,
        ).as_service(port=8004)

        await workflow_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
