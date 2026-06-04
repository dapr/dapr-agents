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

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional, Self

from dapr.clients import DaprClient
from dapr.ext.workflow import DaprWorkflowClient
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from pydantic.dataclasses import dataclass

from dapr_agents.agents.durable import DurableAgent
from dapr_agents.workflow.runners.agent import AgentRunner


_DRASI_OP_TO_DESCRIPTION = {
    "i": "insert",
    "u": "update",
    "d": "delete",
    "x": "control",
}


class DrasiUnpackedSource(BaseModel):
    """Source information for a Drasi event."""
    queryId: str = Field(description="The query ID that generated this event")
    ts_ms: int = Field(description="Source timestamp in milliseconds")


class DrasiUnpackedPayload(BaseModel):
    """Payload containing the event data."""
    source: DrasiUnpackedSource = Field(description="Source information")
    after: Optional[Dict[str, Any]] = Field(
        default=None, description="Record state after the change"
    )
    before: Optional[Dict[str, Any]] = Field(
        default=None, description="Record state before the change"
    )


class DrasiUnpackedEvent(BaseModel):
    """Drasi unpacked event model for CDC (Change Data Capture) events."""
    op: Literal["i", "u", "d", "x"] = Field(
        description="Operation type: i (insert), u (update), d (delete), x (control)"
    )
    ts_ms: int = Field(description="Event timestamp in milliseconds")
    seq: int = Field(description="Event sequence number")
    payload: DrasiUnpackedPayload = Field(description="Event payload containing source and data")

    # ``task`` is required by the agent workflow, but we set it to optional for validation to work since the in-flight Drasi event doesn't have this field
    task: Optional[str] = Field(
        default=None,
        description="Task string for the agent workflow",
    )

    @model_validator(mode="after")
    def ensure_task(self) -> Self:
        """Create a ``task`` string from the Drasi change event data."""

        if self.task is None:
            # Hardcode task for now
            self.task = (
                f"Create a summary of the changes in this {_DRASI_OP_TO_DESCRIPTION[self.op]} event for the '{self.payload.source.queryId}' query.\n"
                f"This is the state before the change: {self.payload.before}.\n"
                f"This is the state after the change: {self.payload.after}.\n"
            )

        return self
        

# TODO: Remove these once activation PR is merged
@dataclass(frozen=True, kw_only=True)
class ActivationContext:
    """Immutable snapshot handed to each activation callback at hosting time.

    Treat every field as read-only. The runner builds one ``ActivationContext``
    per agent the first time the agent is attached, and passes it to each
    callback registered via :meth:`DurableAgent.add_activation`.

    Attributes:
        agent: The agent being hosted.
        runner: The ``AgentRunner`` hosting the agent. Use
            ``runner.run(agent, payload={"task": ...}, wait=False)`` to schedule
            a workflow run from inside an event handler.
        dapr_client: A live Dapr client, guaranteed non-``None`` — the runner
            ensures one exists before activating, even under ``workflow()`` /
            ``run()`` which otherwise never create one. Use this to open a
            streaming subscription when no FastAPI ``app`` is available.
        wf_client: The runner's Dapr workflow client.
        app: The FastAPI app, present only when the agent is hosted via
            ``serve()`` or ``register_routes(fastapi_app=...)``. It is ``None``
            under ``subscribe()``, ``workflow()`` and ``run()`` — extensions
            must branch on ``app is None`` and fall back to ``dapr_client``
            rather than mounting an HTTP route.
    """

    agent: "DurableAgent"
    runner: "AgentRunner"
    dapr_client: "DaprClient"
    wf_client: "DaprWorkflowClient"
    app: Optional["FastAPI"] = None


# A callback registered via ``DurableAgent.add_activation``. It receives the
# ``ActivationContext`` and may return a zero-arg teardown closer that the
# runner invokes on shutdown (return ``None`` when there is nothing to close).
ActivationCallback = Callable[["ActivationContext"], Optional[Callable[[], None]]]