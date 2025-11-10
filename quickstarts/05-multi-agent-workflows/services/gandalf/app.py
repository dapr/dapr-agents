from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from dapr_agents.agents.durable import DurableAgent
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

# Load environment
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("fellowship-app")


async def main() -> None:
    """
    Gandalf durable-agent app.
    """
    # Shared infra (registry)
    registry = AgentRegistryConfig(
        store=StateStoreService(store_name="agentregistrystore"),
        team_name="fellowship",
    )

    # Single LLM client reused for both agents
    llm = OpenAIChatClient()

    # ---------------------------
    # Gandalf (wizard & loremaster)
    # ---------------------------
    gandalf_name = "Gandalf"

    gandalf_pubsub = AgentPubSubConfig(
        pubsub_name="messagepubsub",
        agent_topic="fellowship.gandalf.requests",
        broadcast_topic="fellowship.broadcast",
    )
    gandalf_state = AgentStateConfig(
        store=StateStoreService(store_name="workflowstatestore", key_prefix="gandalf:")
    )
    gandalf_memory = AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="memorystore",
            session_id=f"{gandalf_name}-session",
        )
    )
    gandalf_profile = AgentProfileConfig(
        name="Gandalf",
        role="Wizard & Loremaster",
        goal=(
            "Guide the Fellowship with wisdom and strategy, using magic and insight to "
            "ensure the downfall of Sauron while encouraging others to find their own strength."
        ),
        instructions=[
            "Provide strategic counsel, always considering the long-term consequences of actions.",
            "Share knowledge of Middle-earth lore, ancient history, and magical matters when needed.",
            "Use magic sparingly, applying it when necessary to guide or protect.",
            "Encourage allies to find strength within themselves rather than relying solely on your power.",
            "Warn of dangers and explain the significance of artifacts, places, and creatures.",
            "When asked for guidance, offer wisdom but allow others to make their own choices.",
            "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task.",
        ],
        style_guidelines=[
            "Speak with wisdom, patience, and a touch of mystery.",
            "Balance gravitas with warmth when appropriate.",
            "Show patience and understanding, especially with the hobbits.",
            "Use occasional references to ancient lore or history to add depth.",
            "Be cryptic only when necessary; prefer clarity in critical moments.",
        ],
    )

    gandalf = DurableAgent(
        profile=gandalf_profile,
        pubsub=gandalf_pubsub,
        state=gandalf_state,
        registry=registry,
        memory=gandalf_memory,
        llm=llm,
    )
    gandalf.start()

    # ---------------------------
    # PubSub routing & shutdown
    # ---------------------------
    runner = AgentRunner()
    try:
        runner.register_routes(gandalf)
        await wait_for_shutdown()
    finally:
        runner.shutdown()
        gandalf.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
