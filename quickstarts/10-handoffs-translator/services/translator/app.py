from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

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
logger = logging.getLogger("translator-intake")

MEMORY_SESSION_ID = "translator.session"


async def main() -> None:

    llm = OpenAIChatClient()

    agent = DurableAgent(
        profile=AgentProfileConfig(
            name="Translation Intake",
            role="Multilingual translation coordinator",
            goal="Understand the user request and hand off to the appropriate translator.",
            instructions=[
                "Identify the language requested by the user.",
                "Hand off to the appropriate translator based on the language.",
                "If Spanish is requested, hand off to Spanish Translator.",
                "If Italian is requested, hand off to Italian Translator.",
            ],
            style_guidelines=[
                "Be polite and concise when addressing the user.",
            ],
        ),
        pubsub=AgentPubSubConfig(
            pubsub_name="messagepubsub",
            agent_topic="translator.intake.requests",
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflowstatestore", key_prefix="translator-intake:")
        ),
        registry = AgentRegistryConfig(
            store=StateStoreService(store_name="translatorregistrystore"),
            team_name="translator-swarm",
        ),
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="translatormemorystore",
                session_id=MEMORY_SESSION_ID,
            )
        ),
        llm=llm,
        handoffs=[
            HandoffSpec(
                agent_name="Spanish Translator",
                description="Hand off to the Spanish translator for Spanish translation requests.",
            ),
            HandoffSpec(
                agent_name="Italian Translator",
                description="Hand off to the Italian translator for Italian translation requests.",
            ),
        ],
    )

    agent.start()

    runner = AgentRunner()
    try:
        runner.register_routes(agent)
        await wait_for_shutdown()
    finally:
        runner.shutdown()
        agent.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
