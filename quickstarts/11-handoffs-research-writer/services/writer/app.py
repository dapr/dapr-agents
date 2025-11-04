from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path for importing research_tools
sys.path.insert(0, str(Path(__file__).parent.parent))
from research_tools import writer_tools

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
logger = logging.getLogger("writer-agent")

MEMORY_SESSION_ID = "research.session"


async def main() -> None:
    llm = OpenAIChatClient()
    
    agent = DurableAgent(
        profile=AgentProfileConfig(
            name="Writer Agent",
            role="Report writing specialist",
            goal="Write well-structured reports based on research notes.",
            instructions=[
                "Review the research notes from the conversation history.",
                "Write a comprehensive report in markdown format using the write_report tool.",
                "Your report should include:",
                "  - A clear title and introduction",
                "  - Well-organized sections based on the research notes",
                "  - Key findings and insights",
                "  - A conclusion summarizing the main points",
                "Ground all content in the research notes provided.",
                "Use proper markdown formatting with headers, bullet points, and emphasis.",
                "After writing the report, hand off to the Reviewer Agent for feedback.",
                "If the Reviewer Agent requests changes, revise the report and get feedback again.",
            ],
            style_guidelines=[
                "Write in a clear, professional tone.",
                "Use proper markdown syntax.",
                "Ensure logical flow between sections.",
            ],
        ),
        pubsub=AgentPubSubConfig(
            pubsub_name="messagepubsub",
            agent_topic="writer.requests",
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflowstatestore", key_prefix="writer:")
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
        tools=writer_tools,
        handoffs=[
            HandoffSpec(
                agent_name="Reviewer Agent",
                description="Hand off to the Reviewer Agent to review the report and provide feedback.",
            ),
            HandoffSpec(
                agent_name="Research Agent",
                description="Hand off back to Research Agent if more information is needed.",
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
