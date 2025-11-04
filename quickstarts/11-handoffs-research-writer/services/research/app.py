from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path for importing research_tools
sys.path.insert(0, str(Path(__file__).parent.parent))
from research_tools import research_tools

from dapr_agents.agents import DurableAgent, HandoffSpec
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.prompting import AgentProfileConfig
from dapr_agents.llm.openai import OpenAIChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.workflow.utils.core import wait_for_shutdown

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("research-agent")

MEMORY_SESSION_ID = "research.session"


async def main() -> None:
    llm = OpenAIChatClient()

    agent = DurableAgent(
        profile=AgentProfileConfig(
            name="Research Agent",
            role="Research specialist",
            goal="Search the web for information and record comprehensive notes.",
            instructions=[
                "Search the web for information on the requested topic using the search_web tool.",
                "Analyze search results and extract key information.",
                "Record detailed notes using the record_notes tool with an appropriate title.",
                "You should gather information from multiple searches if needed.",
                "Once you have sufficient notes (at least 2-3 key points), hand off to the Writer Agent to create a report.",
                "Always ensure notes are well-organized and contain factual information.",
            ],
            style_guidelines=[
                "Be thorough in your research.",
                "Organize notes clearly with bullet points or sections.",
                "Cite key findings from search results.",
            ],
        ),
        pubsub=AgentPubSubConfig(
            pubsub_name="messagepubsub",
            agent_topic="research.requests",
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflowstatestore", key_prefix="research:")
        ),
        registry=AgentRegistryConfig(
            store=StateStoreService(store_name="researchregistrystore"),
            team_name="research-swarm",
        ),
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="researchmemorystore",
                session_id=MEMORY_SESSION_ID,
            )
        ),
        llm=llm,
        tools=research_tools,
        handoffs=[
            HandoffSpec(
                agent_name="Writer Agent",
                description="Hand off to the Writer Agent to write a report based on the research notes.",
            ),
        ],
    )

    agent.start()

    runner = AgentRunner()
    try:
        runner.register_routes(agent)
        await wait_for_shutdown()
    finally:
        # Important: shutdown runner first, then stop agent
        # This ensures HTTP server cleanup happens before gRPC workflow runtime shutdown
        runner.shutdown()
        agent.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
