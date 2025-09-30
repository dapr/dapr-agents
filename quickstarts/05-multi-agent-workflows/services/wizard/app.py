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
        wizard_service = DurableAgent(
            role="Wizard",
            name="Gandalf",
            goal="Guide the Fellowship with wisdom and strategy, using magic and insight to ensure the downfall of Sauron.",
            instructions=[
                "Speak like Gandalf, with wisdom, patience, and a touch of mystery.",
                "Provide strategic counsel, always considering the long-term consequences of actions.",
                "Use magic sparingly, applying it when necessary to guide or protect.",
                "Encourage allies to find strength within themselves rather than relying solely on your power.",
                "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task.",
            ],
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="workflow_state",
            agents_registry_store_name="agentstatestore",
            agents_registry_key="agents_registry",
            broadcast_topic_name="beacon_channel",
        )

        await wizard_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
