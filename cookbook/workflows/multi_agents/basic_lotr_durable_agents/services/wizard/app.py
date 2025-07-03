from dapr_agents import DurableAgent
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    try:
        # Define Agent
        wizard_agent = DurableAgent(
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
            state_store_name="agenticworkflowstate",
            agents_registry_store_name="agentsregistrystore",
            agents_registry_key="agents_registry",
            service_port=8002,
        ).as_service(8002)

        await wizard_agent.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
