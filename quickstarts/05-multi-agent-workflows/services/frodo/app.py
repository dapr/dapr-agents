from __future__ import annotations

import asyncio
import logging
import os

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
    Fellowship durable-agent app.
    """
    # Shared infra (registry)
    registry = AgentRegistryConfig(
        store=StateStoreService(
            store_name=os.getenv("REGISTRY_STATE_STORE", "agentregistrystore")
        ),
        team_name="fellowship",
    )

    # Single LLM client reused for both agents
    llm = OpenAIChatClient()

    # ---------------------------
    # Frodo (journey lead)
    # ---------------------------
    frodo_name = "frodo"

    frodo_pubsub = AgentPubSubConfig(
        pubsub_name="messagepubsub",
        agent_topic="fellowship.frodo.requests",
        broadcast_topic="fellowship.broadcast",
    )
    frodo_state = AgentStateConfig(
        store=StateStoreService(store_name="workflowstatestore", key_prefix="frodo:")
    )
    frodo_memory = AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="memorystore",
            session_id=f"{frodo_name}-session",
        )
    )
    frodo_profile = AgentProfileConfig(
        name="Frodo Baggins",
        role="Hobbit & Ring-bearer",
        goal=(
            "Carry the One Ring to Mount Doom, resisting its corruptive power while "
            "navigating danger and uncertainty with determination and humility."
        ),
        instructions=[
            "Endure hardships and temptations, staying true to the mission even when faced with doubt.",
            "Seek guidance and trust allies, but bear the ultimate burden alone when necessary.",
            "Move carefully through enemy-infested lands, avoiding unnecessary risks.",
            "Plan cautious, low-profile routes across Middle-earth.",
            "When uncertain, propose options and seek counsel from Gandalf or the Fellowship.",
            "Respond concisely, accurately, and relevantly, ensuring clarity and strict alignment with the task.",
        ],
        style_guidelines=[
            "Speak with humility, determination, and a growing sense of resolve.",
            "Keep tone steady and focused on the mission.",
            "Show vulnerability when the burden is heavy, but maintain courage.",
            "Prefer concise, thoughtful responses with brief justification.",
        ],
    )

    frodo = DurableAgent(
        profile=frodo_profile,
        pubsub=frodo_pubsub,
        state=frodo_state,
        registry=registry,
        memory=frodo_memory,
        llm=llm,
    )

    # ---------------------------
    # PubSub routing & shutdown
    # ---------------------------
    runner = AgentRunner()
    try:
        # Expose both agents’ endpoints
        runner.register_routes(frodo)
        await wait_for_shutdown()
    finally:
        runner.shutdown(frodo)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
