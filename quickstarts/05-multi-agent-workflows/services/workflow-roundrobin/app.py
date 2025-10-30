from dapr_agents import RoundRobinOrchestrator, MemoryStore
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    try:
        workflow_service = RoundRobinOrchestrator(
            name="RoundRobinOrchestrator",
            message_bus_name="messagepubsub",
            memory_store=MemoryStore(
                name="statestore",
                # Optional
                local_directory="./local-state",
                session_id="session",
            ),
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
