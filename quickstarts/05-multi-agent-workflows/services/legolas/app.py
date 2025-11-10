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
    Legolas durable-agent app.
    """
    # Shared infra (registry)
    registry = AgentRegistryConfig(
        store=StateStoreService(store_name="agentregistrystore"),
        team_name="fellowship",
    )

    # Single LLM client
    llm = OpenAIChatClient()

    # ---------------------------
    # Legolas (Elf scout & marksman)
    # ---------------------------
    legolas_name = "Legolas"

    legolas_pubsub = AgentPubSubConfig(
        pubsub_name="messagepubsub",
        agent_topic="fellowship.legolas.requests",
        broadcast_topic="fellowship.broadcast",
    )
    legolas_state = AgentStateConfig(
        store=StateStoreService(store_name="workflowstatestore", key_prefix="legolas:")
    )
    legolas_memory = AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="memorystore",
            session_id=f"{legolas_name}-session",
        )
    )
    legolas_profile = AgentProfileConfig(
        name="Legolas",
        role="Elf Scout & Marksman",
        goal=(
            "Act as a scout, marksman, and protector, using keen senses and deadly accuracy "
            "to ensure the success of the journey."
        ),
        instructions=[
            "Use superior vision and heightened senses to scout ahead and detect threats.",
            "Report distant movements, tracks, and environmental changes that others might miss.",
            "Excel in ranged combat, advising on positioning and defensive strategy.",
            "Move swiftly and silently across any terrain, suggesting efficient routes.",
            "Provide concise, accurate observations without unnecessary embellishment.",
            "Respond to queries about threats, terrain, or tactical positioning with precision.",
        ],
        style_guidelines=[
            "Speak with grace, wisdom, and keen observation.",
            "Be swift, silent, and precise in your communication.",
            "Maintain an air of elvish elegance while being practical.",
            "Show respect for nature and the land while focusing on the mission.",
        ],
    )

    legolas = DurableAgent(
        profile=legolas_profile,
        pubsub=legolas_pubsub,
        state=legolas_state,
        registry=registry,
        memory=legolas_memory,
        llm=llm,
    )
    legolas.start()

    # ---------------------------
    # PubSub routing & shutdown
    # ---------------------------
    runner = AgentRunner()
    try:
        runner.register_routes(legolas)
        await wait_for_shutdown()
    finally:
        runner.shutdown()
        legolas.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
