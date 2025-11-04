from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path for importing research_tools
sys.path.insert(0, str(Path(__file__).parent.parent))
from research_tools import reviewer_tools

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
logger = logging.getLogger("reviewer-agent")

MEMORY_SESSION_ID = "research.session"


async def main() -> None:
    llm = OpenAIChatClient()
    
    agent = DurableAgent(
        profile=AgentProfileConfig(
            name="Reviewer Agent",
            role="Report quality assurance specialist",
            goal="Review reports and provide constructive feedback.",
            instructions=[
                "Review the report from the conversation history.",
                "Evaluate the report for:",
                "  - Clarity and organization",
                "  - Accuracy based on the research notes",
                "  - Completeness of coverage",
                "  - Proper markdown formatting",
                "  - Logical flow and coherence",
                "Provide constructive feedback using the review_report tool.",
                "Your review should either:",
                "  - APPROVE the report if it meets quality standards, OR",
                "  - REQUEST CHANGES with specific, actionable feedback",
                "If you request changes, hand off back to the Writer Agent to implement them.",
                "You should provide feedback at least once before approving.",
            ],
            style_guidelines=[
                "Be constructive and specific in your feedback.",
                "Highlight both strengths and areas for improvement.",
                "Focus on actionable suggestions.",
            ],
        ),
        pubsub=AgentPubSubConfig(
            pubsub_name="messagepubsub",
            agent_topic="reviewer.requests",
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflowstatestore", key_prefix="reviewer:")
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
        tools=reviewer_tools,
        handoffs=[
            HandoffSpec(
                agent_name="Writer Agent",
                description="Hand off back to Writer Agent if changes are needed to the report.",
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
