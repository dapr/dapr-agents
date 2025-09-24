from dapr_agents import DurableAgent
from dapr_agents.llm.dapr import DaprChatClient
from dotenv import load_dotenv
from multi_tools import tools
import asyncio
import logging
import os


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Ensure default Dapr LLM component is set (e.g., "openai" or "google")
    os.environ.setdefault("DAPR_LLM_COMPONENT_DEFAULT", "openai")

    # Initialize the llm provider using the DaprChatClient
    # By default, the LLM provider is DaprChatClient("openai")
    # Uncomment the line below to use it:
    # llm_provider = DaprChatClient()

    # Setting to None has the same effect as the default above
    llm_provider = None

    # Instantiate the agent with the llm provider
    agent = DurableAgent(
        role="Research And Weather Assistant",
        name="Alex",
        goal=(
            "Help humans get weather and general information; when needed, use tools like"
            " weather lookup, calculator, and web search to answer multi-part queries."
        ),
        instructions=[
            "Be concise and accurate.",
            "Use the calculator for numeric expressions.",
            "Use web search for general facts when asked.",
            "Use the weather tool for location-based weather.",
        ],
        message_bus_name="messagepubsub",
        state_store_name="workflowstatestore",
        state_key="workflow_state",
        agents_registry_store_name="workflowstatestore",
        agents_registry_key="agents_registry",
        tools=tools,
        llm=llm_provider,
    )

    # An example prompt that can require multiple tool calls using the llm provider
    prompt = (
        "What's the current weather in Boston, MA, then compute (14*7)+23, and finally "
        "search for the official tourism site for Boston?"
    )

    await agent.run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
