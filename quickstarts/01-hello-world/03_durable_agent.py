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
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    AgentExecutionConfig,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner


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


# ----------------------------------------------------------------------------------------------------------------------
# There are three ways to set the LLM component with DaprChatClient:
#
# 1. Explicitly instantiate the DaprChatClient with the component name
# llm = DaprChatClient(component_name="openai")
#
# 2. Use the environment variable DAPR_LLM_COMPONENT_DEFAULT
# os.environ.setdefault("DAPR_LLM_COMPONENT_DEFAULT", "openai")
#
# 3. If there is only one conversation component in the resources folder, it will be used by default
# ----------------------------------------------------------------------------------------------------------------------


async def main():
    try:
        pubsub = AgentPubSubConfig(
            pubsub_name="messagepubsub",
        )
        state = AgentStateConfig(
            store=StateStoreService(
                store_name="workflowstatestore",
            )
        )
        registry = AgentRegistryConfig(
            store=StateStoreService(store_name="registrystatestore"),
            team_name="default",
        )
        execution = AgentExecutionConfig(max_iterations=3)
        dapr_store = ConversationDaprStateMemory(
            store_name="conversationstore", session_id="my-unique-id"
        )
        memory = AgentMemoryConfig(store=dapr_store)

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
            pubsub=pubsub,
            registry=registry,
            execution=execution,
            memory=memory,
            state=state,
            # llm=llm, # if you don't set the llm attribute, it will be by default set to DaprChatClient()
        )

        # Start the agent (registers workflows with the runtime)
        travel_planner.start()

        # Create an AgentRunner to execute the workflow
        runner = AgentRunner()

        # Define the prompt
        prompt = "I want to find flights to Paris"

        # Run the workflow and wait for completion
        result = await runner.run(
            travel_planner,
            payload={"task": prompt},
        )
        print(f"\n✅ Final Result:\n{result}\n", flush=True)

    except Exception as e:
        print(f"Error running workflow: {e}")
        raise
    finally:
        # Stop agent first (tears down durabletask runtime)
        travel_planner.stop()
        # Then shut down runner (unwire/close clients)
        runner.shutdown()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
