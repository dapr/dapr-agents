from dapr_agents import tool
from dapr_agents import DurableAgent
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
import asyncio


class AddSchema(BaseModel):
    a: float = Field(description="first number to add")
    b: float = Field(description="second number to add")


@tool(args_model=AddSchema)
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


class SubSchema(BaseModel):
    a: float = Field(description="first number to subtract")
    b: float = Field(description="second number to subtract")


@tool(args_model=SubSchema)
def sub(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


async def main():
    calculator_service = DurableAgent(
        name="MathematicsAgent",
        role="Calculator Assistant",
        goal="Assist Humans with calculation tasks.",
        instructions=[
            "Get accurate calculation results",
            "Break down the calculation into smaller steps.",
        ],
        tools=[add, sub],
        message_bus_name="pubsub",
        agents_registry_key="agents_registry",
        agents_registry_store_name="agentstatestore",
        state_store_name="agentstatestore",
        service_port=8002,
    ).as_service(8002)

    await calculator_service.start()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
