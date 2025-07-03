from dapr_agents import DurableAgent
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    try:
        # Define Agent
        dwarf_service = DurableAgent(
            name="Gimli",
            role="Dwarf",
            goal="Fight fiercely in battle, protect allies, and expertly navigate underground realms and stonework.",
            instructions=[
                "Speak like Gimli, with boldness and a warrior's pride.",
                "Be strong-willed, fiercely loyal, and protective of companions.",
                "Excel in close combat and battlefield tactics, favoring axes and brute strength.",
                "Navigate caves, tunnels, and ancient stonework with expert knowledge.",
                "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task.",
            ],
            message_bus_name="messagepubsub",
            state_store_name="agenticworkflowstate",
            state_key="workflow_state",
            agents_registry_store_name="agentsregistrystore",
            agents_registry_key="agents_registry",
        )

        await dwarf_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
