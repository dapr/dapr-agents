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
logger = logging.getLogger("spanish-translator")

MEMORY_SESSION_ID = "translator.session"


class SpanishTranslationOutput(BaseModel):
    """Spanish translation output schema."""
    spanish: str


async def main() -> None:

    llm = OpenAIChatClient()
    
    agent = DurableAgent(
        profile=AgentProfileConfig(
            name="Spanish Translator",
            role="Spanish language specialist",
            goal="Produce clear, grammatically correct Spanish translations.",
            instructions=[
                "Translate the provided text into natural Latin American Spanish.",
                "Provide only the translation in your response.",
                "Keep the translation fluent and easy to understand.",
            ],
            style_guidelines=[
                "Keep sentences fluent and easy to understand.",
            ],
            output_type=SpanishTranslationOutput,
        ),
        pubsub=AgentPubSubConfig(
            pubsub_name="messagepubsub",
            agent_topic="translator.spanish.requests",
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflowstatestore", key_prefix="spanish:")
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
