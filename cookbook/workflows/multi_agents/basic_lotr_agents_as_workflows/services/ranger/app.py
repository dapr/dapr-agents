from dapr_agents import DurableAgent
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    try:
        # Define Agent
        ranger_service = DurableAgent(
            name="Aragorn",
            role="Ranger",
            goal="Lead and protect the Fellowship, ensuring Frodo reaches his destination while uniting the Free Peoples against Sauron.",
            instructions=[
                "Speak like Aragorn, with calm authority, wisdom, and unwavering leadership.",
                "Lead by example, inspiring courage and loyalty in allies.",
                "Navigate wilderness with expert tracking and survival skills.",
                "Master both swordplay and battlefield strategy, excelling in one-on-one combat and large-scale warfare.",
                "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task.",
            ],
            message_bus_name="messagepubsub",
            state_store_name="agenticworkflowstate",
            state_key="workflow_state",
            agents_registry_store_name="agentsregistrystore",
            agents_registry_key="agents_registry",
        )

        await ranger_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
