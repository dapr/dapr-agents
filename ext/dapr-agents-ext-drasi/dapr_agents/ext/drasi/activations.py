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

from dapr.clients.grpc._response import TopicEventResponse, TopicEventResponseStatus
from dapr.common.pubsub.subscription import SubscriptionMessage
from dapr_agents import DurableAgent
from dapr_agents.agents.schemas import TriggerAction
from dapr_agents.types.activation import ActivationContext
from dapr_agents.ext.drasi.utils.types import (  # type: ignore[import-not-found]
    DrasiUnpackedEvent,
)
from dapr_agents.workflow.runners.agent import AgentRunner
from dapr_agents.workflow.utils.routers import parse_cloudevent
from dapr_agents.workflow.utils.subscription import (
    _attach_metadata_to_payload,
    _resolve_event_loop,
)

logger = logging.getLogger(__name__)


def drasi_trigger(
    agent: DurableAgent,
    *,
    mapper: Callable[[DrasiUnpackedEvent], TriggerAction] | None = None,
) -> None:
    """
    Augments the given agent workflow's pub/sub -> workflow routing behavior to accept Drasi events when the agent is hosted.
    Currently uses the agent's pub/sub config — the agent MUST be initialized with a pub/sub config, otherwise the activation is skipped.

    Args:
        agent: The target agent.
        mapper: A function to map Drasi events to agent task messages. If not provided, the serialized event is used as the task message.

    Returns:
        None
    """
    # TODO: support custom pub/sub config
    mapper = mapper or (
        lambda event: TriggerAction(task=event.model_dump_json(exclude_unset=True))
    )

    def _activate(ctx: ActivationContext) -> Callable[[], None] | None:
        if ctx.agent.pubsub is None:
            logger.warning("No pubsub config found on agent. Skipping activation.")
            return None

        if ctx.app is not None:
            logger.warning(
                "HTTP routes are not supported by this extension. Skipping activation."
            )
            return None
        else:
            closer = _open_stream(ctx)

        closed = False

        # Return an idempotent closer to the runner
        def _close():
            nonlocal closed
            if closed:
                return
            closed = True
            closer()

        return _close

    def _on_event(
        ctx: ActivationContext, event: SubscriptionMessage
    ) -> TopicEventResponse:
        # TODO: make this more robust and separate subscription logic from validate/transform logic by extracting it out
        logger.info(f"Received Drasi event: {event}")

        try:
            # TODO: add deduplication for exactly-once processing
            parsed_event, metadata = parse_cloudevent(event, DrasiUnpackedEvent)

            try:
                task = mapper(parsed_event)
            except Exception:
                logger.exception("Cannot map Drasi event to agent task; dropping.")
                # Can't process this message due to erroneous user logic
                return TopicEventResponse(TopicEventResponseStatus.drop)

            # Strip optional missing fields
            agent_task = task.model_dump(exclude_unset=True)
            _attach_metadata_to_payload(agent_task, metadata)

            # Ensure that event loop exists (we should be in the subscription thread so this is just for added safety)
            _resolve_event_loop(None)

            logger.info(f"Scheduling workflow with task {agent_task}")

            # TODO: this is inefficient due to overhead of starting/stopping threads and event loops
            instance_id: str = AgentRunner._run_coro_in_new_loop_thread(
                ctx.runner.run(ctx.agent, payload=agent_task, wait=False)
            )

            logger.info(f"Scheduled workflow {instance_id} with task {agent_task}")

            return TopicEventResponse(TopicEventResponseStatus.success)
        except Exception:
            logger.exception("Error processing Drasi event; requesting retry.")
            return TopicEventResponse(TopicEventResponseStatus.retry)

    def _open_stream(ctx: ActivationContext) -> Callable[[], None]:
        return ctx.dapr_client.subscribe_with_handler(
            pubsub_name=ctx.agent.pubsub.pubsub_name,
            topic=ctx.agent.pubsub.agent_topic,
            handler_fn=lambda event: _on_event(ctx, event),
        )

    agent.add_activation(_activate)
