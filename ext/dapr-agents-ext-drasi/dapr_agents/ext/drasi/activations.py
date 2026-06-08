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

import logging
from typing import Callable

from dapr_agents.types.activation import ActivationCallback, ActivationContext
from dapr_agents.workflow.decorators import message_router
from dapr_agents.ext.drasi.utils.types import (
    DrasiUnpackedEvent,
)

logger = logging.getLogger(__name__)


def drasi_trigger() -> ActivationCallback:
    """
    Overrides the agent workflow's pub/sub -> workflow routing behavior to accept Drasi events.
    The agent MUST be initialized with a pub/sub config, otherwise the activation is skipped.

    Returns:
        The activation callback to be invoked when the agent is hosted.
    """

    def activation_callback(ctx: ActivationContext) -> Callable[[], None] | None:
        if ctx.agent.pubsub is None:
            logger.warning("No pubsub config found on agent. Skipping activation.")
            return None

        # Get the underlying function object from the bound method
        agent_workflow = ctx.agent.agent_workflow.__func__  # type: ignore[attr-defined]
        pubsub = ctx.agent.pubsub.pubsub_name
        topic = ctx.agent.pubsub.agent_topic

        message_router(
            agent_workflow,
            pubsub=pubsub,
            topic=topic,
            message_model=DrasiUnpackedEvent,  # NOTE: only unpacked Drasi events are currently supported
        )

        # No closer needed for now
        return None

    return activation_callback
