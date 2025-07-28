import asyncio
from weather_tools import tools
from dapr_agents import Agent, HFHubChatClient
from dotenv import load_dotenv

load_dotenv()

llm = HFHubChatClient(model="HuggingFaceTB/SmolLM3-3B")

AIAgent = Agent(
    name="Stevie",
    role="Weather Assistant",
    goal="Assist Humans with weather related tasks.",
    instructions=[
        "Always answer the user's main weather question directly and clearly.",
        "If you perform any additional actions (like jumping), summarize those actions and their results.",
        "At the end, provide a concise summary that combines the weather information for all requested locations and any other actions you performed.",
    ],
    llm=llm,
    tools=tools,
)


# Wrap your async call
async def main():
    await AIAgent.run("What is the weather in Virginia, New York and Washington DC?")


if __name__ == "__main__":
    asyncio.run(main())
