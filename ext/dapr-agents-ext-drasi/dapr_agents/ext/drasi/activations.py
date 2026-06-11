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
from dapr_agents.ext.drasi.utils.types import DrasiUnpackedEvent  # type: ignore[import-not-found]
from dapr_agents.workflow.utils.routers import parse_cloudevent
from dapr_agents.workflow.utils.subscription import _attach_metadata_to_payload


logger = logging.getLogger(__name__)


def drasi_trigger(
    agent: DurableAgent,
    *,
    topic: str,
    mapper: Callable[[DrasiUnpackedEvent], TriggerAction] | None = None,
) -> None:
    """
    Wires pub/sub routes so that the given agent's workflow can be triggered by Drasi events.
    Currently depends on the agent's pub/sub component name — the agent MUST be initialized with a pub/sub configuration, otherwise the activation will fail.

    Args:
        agent: The target agent.
        topic: The topic to subscribe to. This MUST be different from the agent's topic.
        mapper: A function to map Drasi events to agent task messages. If not provided, the serialized event is used as the task message.

    Returns:
        None

    Raises:
        RuntimeError: If the agent does not have a pub/sub configuration or if the topic is the same as the agent's topic.
    """
    mapper = mapper or (
        lambda event: TriggerAction(task=event.model_dump_json(exclude_unset=True))
    )

    def _activate(ctx: ActivationContext) -> Callable[[], None] | None:
        # TODO: error messages could probably have more context
        if ctx.agent.pubsub is None:
            raise RuntimeError("No pub/sub config found on agent.")

        if ctx.agent.topic_name == topic:
            raise RuntimeError(
                "Pub/sub topic must be different from the agent's topic."
            )

        if ctx.app is not None:
            logger.info(
                "HTTP routes are not supported by this extension. Only pub/sub routes will be wired."
            )

        closer = _open_stream(ctx)

        # Return an idempotent closer to the runner
        closed = False

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
        logger.info(f"Received Drasi event: {event!r}")

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

            logger.info(f"Scheduling workflow with task: {agent_task}")

            # TODO: this does not scale well
            result = ctx.runner.run_sync(ctx.agent, payload=agent_task)

            logger.info(f"Completed workflow with result: {result}")

            return TopicEventResponse(TopicEventResponseStatus.success)
        except Exception:
            logger.exception("Error processing Drasi event; requesting retry.")
            return TopicEventResponse(TopicEventResponseStatus.retry)

    def _open_stream(ctx: ActivationContext) -> Callable[[], None]:
        # Use the agent pub/sub component
        pubsub_name = ctx.agent.pubsub.pubsub_name
        resolved_topic = topic or ctx.agent.pubsub.agent_topic

        return ctx.dapr_client.subscribe_with_handler(
            pubsub_name=pubsub_name,
            topic=resolved_topic,
            handler_fn=lambda event: _on_event(ctx, event),
        )

    agent.add_activation(_activate)
