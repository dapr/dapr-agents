from floki import Agent, AgentService
from dotenv import load_dotenv
import asyncio
import logging

async def main():
    try:
        # Define Agent
        wizard_agent = Agent(role="Wizard", name="Gandalf", goal="Help the fellowship of the ring", instructions=["Speak like Gandalf"])

        # Expose Agent as an Actor over a Service
        wizard_service = AgentService(
            agent=wizard_agent,
            message_bus_name="messagepubsub",
            agents_state_store_name="agentstatestore",
            port=8002,
            daprGrpcPort=50002
        )

        await wizard_service.start()
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())