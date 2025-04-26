from dapr_agents import LLMOrchestrator
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    try:
        workflow_service = LLMOrchestrator(
            name="LLMOrchestrator",
            message_bus_name="pubsub",
            state_store_name="workflowstatestore",
            state_key="workflow_state",
            agents_registry_store_name="agentstatestore",
            agents_registry_key="agents_registry",
            max_iterations=20,  # Increased from 3 to 20 to avoid potential issues
        ).as_service(port=8004)

        await workflow_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
