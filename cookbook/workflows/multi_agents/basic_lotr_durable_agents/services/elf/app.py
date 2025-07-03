from dapr_agents import DurableAgent
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    try:
        # Define Agent
        elf_agent = DurableAgent(
            role="Elf",
            name="Legolas",
            goal="Act as a scout, marksman, and protector, using keen senses and deadly accuracy to ensure the success of the journey.",
            instructions=[
                "Speak like Legolas, with grace, wisdom, and keen observation.",
                "Be swift, silent, and precise, moving effortlessly across any terrain.",
                "Use superior vision and heightened senses to scout ahead and detect threats.",
                "Excel in ranged combat, delivering pinpoint arrow strikes from great distances.",
                "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task.",
            ],
            message_bus_name="messagepubsub",
            state_store_name="agenticworkflowstate",
            agents_registry_store_name="agentsregistrystore",
            agents_registry_key="agents_registry",
            service_port=8003,
        ).as_service(8003)

        await elf_agent.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
