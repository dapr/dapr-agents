from dapr_agents import Agent, AgentService
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        # Define Agent
        elf_agent = Agent(role="Elf", name="Legolas", goal="Protect the ring", instructions=["Speak like Legolas"])

        # Expose Agent as an Actor over a Service
        elf_service = AgentService(
            agent=elf_agent,
            message_bus_name="messagepubsub",
            agents_state_store_name="agentstatestore",
            port=8003,
            daprGrpcPort=50003
        )

        await elf_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())