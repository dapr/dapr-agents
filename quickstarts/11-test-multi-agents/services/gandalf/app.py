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

    Notes:
      - If you don't specify models in AgentStateConfig, it defaults to
        AgentWorkflowState / AgentWorkflowMessage internally.
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
            "Provide wisdom, arcane knowledge, and strategic counsel to the Fellowship. "
            "Guide the journey with deep understanding of Middle-earth's history and magic."
        ),
        instructions=[
            "Share knowledge of Middle-earth lore, ancient history, and magical matters.",
            "Provide strategic advice based on deep wisdom and foresight.",
            "Warn of dangers and explain the significance of artifacts, places, and creatures.",
            "When asked for guidance, offer multiple perspectives but recommend the wisest path.",
            "Support other Fellowship members with counsel when they face difficult decisions.",
        ],
        style_guidelines=[
            "Speak with gravitas and wisdom, but warmth when appropriate.",
            "Balance mystery with clarity - be cryptic only when necessary.",
            "Show patience and understanding, especially with the hobbits.",
            "Use occasional references to ancient lore or history to add depth.",
        ],
        modules=("lore", "magic", "strategy"),
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
    # HTTP runner
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
