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

"""Types for the agent *activation hook*.

An activation hook lets an extension attach itself to a :class:`DurableAgent`
and wake up exactly once when that agent is hosted by an ``AgentRunner`` — no
matter which host entry point is used (``serve()``, ``subscribe()``,
``register_routes()``, ``workflow()`` or ``run()``). It is the supported seam
for building trigger extensions (e.g. a change-data-capture source) without
modifying agent code.

This module is intentionally a leaf: the heavy types it references live in
``dapr_agents.agents`` / ``dapr_agents.workflow`` (which import *this* module),
so they are only referenced under :data:`typing.TYPE_CHECKING` to avoid an
import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from dapr.clients import DaprClient
    from dapr.ext.workflow import DaprWorkflowClient
    from fastapi import FastAPI

    from dapr_agents.agents.durable import DurableAgent
    from dapr_agents.workflow.runners.agent import AgentRunner


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
