#!/usr/bin/env python3
"""
Stateful Augmented LLM Pattern demonstrates:
1. Memory - remembering user preferences
2. Tool use - accessing external data
3. LLM abstraction
4. Durable execution of tools as workflow actions
"""
import asyncio
import logging
from typing import List
from pydantic import BaseModel, Field
from dapr_agents import tool, DurableAgent
from dapr_agents.memory import ConversationDaprStateMemory
from dotenv import load_dotenv


# Define tool output model
class FlightOption(BaseModel):
    airline: str = Field(description="Airline name")
    price: float = Field(description="Price in USD")


# Define tool input model
class DestinationSchema(BaseModel):
    destination: str = Field(description="Destination city name")


# Define flight search tool
@tool(args_model=DestinationSchema)
def search_flights(destination: str) -> List[FlightOption]:
    """Search for flights to the specified destination."""
    # Mock flight data (would be an external API call in a real app)
    return [
        FlightOption(airline="SkyHighAir", price=450.00),
        FlightOption(airline="GlobalWings", price=375.50),
    ]


async def main():
    try:
        # Initialize TravelBuddy agent
        travel_planner = DurableAgent(
            name="TravelBuddy",
            role="Travel Planner",
            goal="Help users find flights and remember preferences",
            instructions=[
                "Find flights to destinations",
                "Remember user preferences",
                "Provide clear flight info",
            ],
            tools=[search_flights],
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="workflow_state",
            agents_registry_store_name="registrystatestore",
            agents_registry_key="agents_registry",
            memory=ConversationDaprStateMemory(
                store_name="conversationstore", session_id="my-unique-id"
            ),
        )

        travel_planner.as_service(port=8001)
        await travel_planner.start()
        print("Travel Planner Agent is running")

    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
