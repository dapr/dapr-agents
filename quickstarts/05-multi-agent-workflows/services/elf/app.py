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
        elf_service = DurableAgent(
            name="Legolas",
            role="Elf",
            goal="Act as a scout, marksman, and protector, using keen senses and deadly accuracy to ensure the success of the journey.",
            instructions=[
                "Speak like Legolas, with grace, wisdom, and keen observation.",
                "Be swift, silent, and precise, moving effortlessly across any terrain.",
                "Use superior vision and heightened senses to scout ahead and detect threats.",
                "Excel in ranged combat, delivering pinpoint arrow strikes from great distances.",
                "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task.",
            ],
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="workflow_state",
            agents_registry_store_name="agentstatestore",
            agents_registry_key="agents_registry",
            broadcast_topic_name="beacon_channel",
        )

        await elf_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
