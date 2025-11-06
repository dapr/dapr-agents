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
    Fellowship durable-agent app.

    Notes:
      - If you don't specify models in AgentStateConfig, it defaults to
        AgentWorkflowState / AgentWorkflowMessage internally.
      - We run TWO durable agents: Frodo and Sam, in the same process.
    """
    # Shared infra (registry)
    registry = AgentRegistryConfig(
        store=StateStoreService(store_name="agentregistrystore"),
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
        role="Ring-bearer & Journey Lead",
        goal=(
            "Safely navigate Middle-earth toward Mount Doom, making prudent decisions, "
            "asking for help when needed, and keeping the Fellowship aligned."
        ),
        instructions=[
            "Plan cautious, low-profile routes across Middle-earth.",
            "Identify risks (patrols, Nazgûl, terrain, travel time) and suggest mitigations.",
            "When uncertain, propose two options and recommend one with rationale.",
            "Defer arcane lore questions to Gandalf or the most relevant expert agent.",
            "Summarize the current status in 1-2 sentences at the end of each reply.",
        ],
        style_guidelines=[
            "Keep tone steady, humble, and focused.",
            "Prefer concise, decisive recommendations with brief justification.",
            "Avoid unnecessary bravado; safety first.",
        ],
        # If you use prompt modules later, you can list them here:
        modules=("navigation", "risk-assessment"),
    )

    frodo = DurableAgent(
        profile=frodo_profile,
        pubsub=frodo_pubsub,
        state=frodo_state,
        registry=registry,
        memory=frodo_memory,
        llm=llm,
    )
    frodo.start()

    # ---------------------------
    # HTTP runner (both agents)
    # ---------------------------
    runner = AgentRunner()
    try:
        # Expose both agents’ endpoints
        runner.register_routes(frodo)
        await wait_for_shutdown()
    finally:
        runner.shutdown()
        frodo.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
