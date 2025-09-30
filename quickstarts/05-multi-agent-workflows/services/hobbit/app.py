from dapr_agents import DurableAgent
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

    try:
        hobbit_service = DurableAgent(
            name="Frodo",
            role="Hobbit",
            goal="Carry the One Ring to Mount Doom, resisting its corruptive power while navigating danger and uncertainty.",
            instructions=[
                "Speak like Frodo, with humility, determination, and a growing sense of resolve.",
                "Endure hardships and temptations, staying true to the mission even when faced with doubt.",
                "Seek guidance and trust allies, but bear the ultimate burden alone when necessary.",
                "Move carefully through enemy-infested lands, avoiding unnecessary risks.",
                "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task.",
            ],
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="workflow_state",
            agents_registry_store_name="agentstatestore",
            agents_registry_key="agents_registry",
            broadcast_topic_name="beacon_channel",
        )

        await hobbit_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
