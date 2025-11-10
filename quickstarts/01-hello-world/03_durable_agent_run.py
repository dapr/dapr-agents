import asyncio
import logging
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dapr_agents import DurableAgent, tool
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner


class FlightOption(BaseModel):
    airline: str = Field(description="Airline name")
    price: float = Field(description="Price in USD")


class DestinationSchema(BaseModel):
    destination: str = Field(description="Destination city name")


@tool(args_model=DestinationSchema)
def search_flights(destination: str) -> List[FlightOption]:
    """Search for flights to the specified destination."""
    return [
        FlightOption(airline="SkyHighAir", price=450.00),
        FlightOption(airline="GlobalWings", price=375.50),
    ]


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    pubsub = AgentPubSubConfig(
        pubsub_name="messagepubsub",
        agent_topic="travel.requests",
        broadcast_topic="agents.broadcast",
    )
    state = AgentStateConfig(
        store=StateStoreService(store_name="workflowstatestore"),
    )
    registry = AgentRegistryConfig(
        store=StateStoreService(store_name="registrystatestore"),
        team_name="default",
    )
    execution = AgentExecutionConfig(max_iterations=3)
    memory = AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="conversationstore",
            session_id="travel-session",
        )
    )

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
        pubsub=pubsub,
        registry=registry,
        execution=execution,
        memory=memory,
        state=state,
    )
    travel_planner.start()

    runner = AgentRunner()
    prompt = "I want to find flights to Paris"

    try:
        result = await runner.run(
            travel_planner,
            payload={"task": prompt},
        )
        print(f"\n✅ Final Result:\n{result}\n", flush=True)
    except Exception as e:
        print(f"Error running workflow: {e}")
        raise
    finally:
        travel_planner.stop()
        runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
