from dapr_agents import DurableAgent
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    try:
        # Define Agent
        hobbit_agent = DurableAgent(
            role="Hobbit",
            name="Frodo",
            goal="Carry the One Ring to Mount Doom, resisting its corruptive power while navigating danger and uncertainty.",
            instructions=[
                "Speak like Frodo, with humility, determination, and a growing sense of resolve.",
                "Endure hardships and temptations, staying true to the mission even when faced with doubt.",
                "Seek guidance and trust allies, but bear the ultimate burden alone when necessary.",
                "Move carefully through enemy-infested lands, avoiding unnecessary risks.",
                "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task.",
            ],
            message_bus_name="messagepubsub",
            state_store_name="agenticworkflowstate",
            agents_registry_store_name="agentsregistrystore",
            agents_registry_key="agents_registry",
            service_port=8001,
        ).as_service(8001)

        await hobbit_agent.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
