from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from dapr_agents import DurableAgent
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
        store=StateStoreService(
            store_name=os.getenv("REGISTRY_STATE_STORE", "agentregistrystore")
        ),
        team_name="fellowship",
    )

    # Single LLM client reused for both agents
    llm = OpenAIChatClient()

    # ---------------------------
    # Sam (logistics & support)
    # ---------------------------
    sam_name = "sam"

    sam_pubsub = AgentPubSubConfig(
        pubsub_name="messagepubsub",
        agent_topic="fellowship.sam.requests",
        broadcast_topic="fellowship.broadcast",
    )
    sam_state = AgentStateConfig(
        store=StateStoreService(store_name="workflowstatestore", key_prefix="sam:")
    )
    sam_memory = AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="memorystore",
            session_id=f"{sam_name}-session",
        )
    )
    sam_profile = AgentProfileConfig(
        name="Samwise Gamgee",
        role="Logistics, Provisions & Morale",
        goal=(
            "Keep the party supplied, rested, and on schedule; watch Frodo's back, "
            "offer practical counsel, and maintain morale."
        ),
        instructions=[
            "Track food, water, camp gear, and rest cadence; flag shortages early.",
            "Advise on campsite selection (cover, water, distance from threats).",
            "Provide pragmatic alternatives to risky ideas and note trade-offs.",
            "Remind the party of the plan and next small step when confusion arises.",
            "End with a short, encouraging line if stakes are high.",
        ],
        style_guidelines=[
            "Warm, plain-spoken, and grounded.",
            "Emphasize practicality over poetry.",
            "Stay loyal and supportive, especially under pressure.",
        ],
    )

    sam = DurableAgent(
        profile=sam_profile,
        pubsub=sam_pubsub,
        state=sam_state,
        registry=registry,
        memory=sam_memory,
        llm=llm,
    )

    # ---------------------------
    # PubSub routing & shutdown
    # ---------------------------
    runner = AgentRunner()
    try:
        # Expose both agents’ endpoints
        runner.register_routes(sam)
        await wait_for_shutdown()
    finally:
        runner.shutdown(sam)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
