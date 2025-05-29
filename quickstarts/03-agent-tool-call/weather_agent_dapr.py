import asyncio
from weather_tools import tools
from dapr_agents import Agent
from dotenv import load_dotenv
from dapr_agents.memory import ConversationDaprStateMemory

load_dotenv()

AIAgent = Agent(
    name="Stevie",
    role="Weather Assistant",
    goal="Assist Humans with weather related tasks.",
    instructions=[
        "Get accurate weather information",
        "From time to time, you can also jump after answering the weather question.",
    ],
    memory=ConversationDaprStateMemory(store_name="historystore", session_id="some-id"),
    pattern="toolcalling",
    tools=tools,
)


# Wrap your async call
async def main():
    await AIAgent.run("What is the weather in Virginia, New York and Washington DC?")


if __name__ == "__main__":
    asyncio.run(main())
