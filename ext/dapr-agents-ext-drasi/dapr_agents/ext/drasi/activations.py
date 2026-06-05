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
from typing import Optional

from dapr_agents.workflow.decorators import message_router
from .utils.types import (
    ActivationCallback,
    ActivationContext,
    DrasiUnpackedEvent,
)

logger = logging.getLogger(__name__)


def drasi_trigger(
    pubsub: Optional[str] = None,
    topic: Optional[str] = None,
) -> ActivationCallback:
    """
    TODO: Add docstring
    """

    def activation_callback(ctx: ActivationContext) -> None:
        # Fallback to the agent's pubsub/topic if not provided
        resolved_pubsub = pubsub
        if not resolved_pubsub:
            if ctx.agent.pubsub and ctx.agent.pubsub.pubsub_name:
                resolved_pubsub = ctx.agent.pubsub.pubsub_name
            else:
                logger.warning("No pubsub could be resolved. Skipping activation.")
                return None

        resolved_topic = topic
        if not resolved_topic:
            if ctx.agent.pubsub and ctx.agent.pubsub.topic:
                resolved_topic = ctx.agent.pubsub.topic
            else:
                logger.warning("No topic could be resolved. Skipping activation.")
                return None

        # Override the agent workflow's pubsub -> workflow routing behavior
        message_router(
            ctx.agent.agent_workflow,
            pubsub=resolved_pubsub,
            topic=resolved_topic,
            message_model=DrasiUnpackedEvent,  # NOTE: only unpacked Drasi events are currently supported
        )

        # No closer needed for now
        return None

    return activation_callback
