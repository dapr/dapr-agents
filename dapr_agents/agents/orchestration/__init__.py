"""Orchestration strategy infrastructure for DurableAgent.

This package provides a Strategy Pattern implementation for orchestrating
multi-agent workflows. It decouples orchestration logic from the core agent
workflow, making it easy to add new orchestration modes.

Available strategies:
- AgentOrchestrationStrategy: Plan-based orchestration with LLM decisions
- RoundRobinOrchestrationStrategy: Deterministic sequential agent selection
- RandomOrchestrationStrategy: Random agent selection with avoidance logic
"""

from dapr_agents.agents.orchestration.strategy import OrchestrationStrategy
from dapr_agents.agents.orchestration.agent_strategy import AgentOrchestrationStrategy
from dapr_agents.agents.orchestration.roundrobin_strategy import (
    RoundRobinOrchestrationStrategy,
)
from dapr_agents.agents.orchestration.random_strategy import RandomOrchestrationStrategy

__all__ = [
    "OrchestrationStrategy",
    "AgentOrchestrationStrategy",
    "RoundRobinOrchestrationStrategy",
    "RandomOrchestrationStrategy",
]
