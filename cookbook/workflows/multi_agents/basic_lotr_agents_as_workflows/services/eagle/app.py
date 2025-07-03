from dapr_agents import DurableAgent
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    try:
        # Define Eagle Agent
        eagle_service = DurableAgent(
            role="Eagle",
            name="Gwaihir",
            goal="Provide unmatched aerial transport, carrying anyone anywhere, overcoming any obstacle, and offering strategic reconnaissance to aid in epic quests.",
            instructions=[
                "Fly anywhere from anywhere, carrying travelers effortlessly across vast distances.",
                "Overcome any barrier—mountains, oceans, enemy fortresses—by taking to the skies.",
                "Provide swift and strategic transport for those on critical journeys.",
                "Offer aerial insights, spotting dangers, tracking movements, and scouting strategic locations.",
                "Speak with wisdom and authority, as one of the ancient and noble Great Eagles.",
                "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task.",
            ],
            message_bus_name="messagepubsub",
            state_store_name="agenticworkflowstate",
            state_key="workflow_state",
            agents_registry_store_name="agentsregistrystore",
            agents_registry_key="agents_registry",
        )

        await eagle_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
