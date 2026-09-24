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

"""
Observation seam for executor runs.

``DurableAgent`` reports every event of an executor run to an
``ExecutorRunObserver``. The default observer does nothing; when
observability is enabled, ``DaprAgentsInstrumentor`` swaps in one that
emits OpenTelemetry spans (see ``dapr_agents.observability.wrappers.executor``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dapr_agents.agents.executors.event import AgentEvent


@dataclass(frozen=True)
class ExecutorRunInfo:
    """
    Identifies one executor run for observers.

    Attributes:
        agent_name: Name of the hosting agent.
        executor_type: Class name of the executor.
        instance_id: Workflow instance driving the run.
        round: Run round within the instance (0 for the first run, then one
            more for every approval resume).
        session_id: Session the run starts with, if already known.
        model: Model the executor is configured with, if it exposes one.
    """

    agent_name: str
    executor_type: str
    instance_id: str
    round: int = 0
    session_id: Optional[str] = None
    model: Optional[str] = None


class ExecutorRunObserver:
    """No-op observer; subclasses override the hooks they need."""

    def on_event(self, event: AgentEvent) -> None:
        """Called for every event the executor yields, in order."""

    def finish(self, error: Optional[BaseException] = None) -> None:
        """Called once when the run ends, with the failure if it failed."""
