from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from pydantic import BaseModel

from dapr_agents.agents import DurableAgent
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
logger = logging.getLogger("italian-translator")

MEMORY_SESSION_ID = "translator.session"


class ItalianTranslationOutput(BaseModel):
    """Italian translation output schema."""
    italian: str


async def main() -> None:

    llm = OpenAIChatClient()

    agent = DurableAgent(
        profile=AgentProfileConfig(
            name="Italian Translator",
            role="Italian language specialist",
            goal="Provide high-quality Italian translations.",
            instructions=[
                "Translate the provided text into standard Italian.",
                "Provide only the translation in your response.",
                "Keep the translation fluent and natural.",
            ],
            style_guidelines=[
                "Produce fluid, natural Italian sentences.",
            ],
            output_type=ItalianTranslationOutput,
        ),
        pubsub=AgentPubSubConfig(
            pubsub_name="messagepubsub",
            agent_topic="translator.italian.requests",
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflowstatestore", key_prefix="italian:")
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
