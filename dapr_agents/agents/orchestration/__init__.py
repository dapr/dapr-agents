#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Orchestration strategy for DurableAgent.

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
