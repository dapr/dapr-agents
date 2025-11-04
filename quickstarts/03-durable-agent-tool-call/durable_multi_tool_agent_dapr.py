import asyncio
import logging
import os

from dotenv import load_dotenv
from agent_tools import tools

from dapr_agents import DurableAgent
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.runners import AgentRunner


async def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    logger = logging.getLogger(__name__)

    # Ensure default Dapr LLM component is set (e.g., "openai" or "google")
    os.environ.setdefault("DAPR_LLM_COMPONENT_DEFAULT", "openai")

    # Initialize the llm provider using the DaprChatClient
    # By default, the LLM provider is DaprChatClient("openai")
    # Uncomment the line below to use it:
    # llm_provider = DaprChatClient()

    # Setting to None has the same effect as the default above
    llm_provider = None

    logger.info("Creating DurableAgent...")
    
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
        tools=tools,
        llm=llm_provider,
        # pubsub is omitted - agent runs standalone via AgentRunner
    )
    # Start the agent (registers workflows with the runtime)
    agent.start()

    # Create an AgentRunner to execute the workflow
    runner = AgentRunner()

    try:
        prompt = (
            "What's the current weather in Boston, MA, then compute (14*7)+23, and finally "
            "search for the official tourism site for Boston?"
        )

        logger.info(f"Running workflow with prompt: {prompt}")
        
        # Run the workflow and wait for completion
        result = await runner.run(
            agent, 
            payload={"task": prompt},
        )

        print(f"\n✅ Final Result:\n{result}\n", flush=True)

    except Exception as e:
        logger.error(f"Error running workflow: {e}", exc_info=True)
        raise
    finally:
        # Stop agent first (tears down durabletask runtime)
        agent.stop()
        # Then shut down runner (unwire/close clients)
        runner.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

