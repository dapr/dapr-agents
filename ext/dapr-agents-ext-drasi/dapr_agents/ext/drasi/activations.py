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
from typing import Any, Callable, Literal, Sequence

from dapr.clients.grpc._response import TopicEventResponse, TopicEventResponseStatus
from dapr.common.pubsub.subscription import SubscriptionMessage
from dapr_agents import DurableAgent
from dapr_agents.agents.schemas import TriggerAction
from dapr_agents.types.activation import ActivationContext
from dapr_agents.ext.drasi.utils.types import DrasiUnpackedEvent  # type: ignore[import-not-found]
from dapr_agents.ext.drasi.utils.filters import normalize_to_list  # type: ignore[import-not-found]
from dapr_agents.workflow.utils.registration import _validate_pubsub_components
from dapr_agents.workflow.utils.routers import parse_cloudevent, validate_message_model
from dapr_agents.workflow.utils.subscription import _attach_metadata_to_payload

logger = logging.getLogger(__name__)

DRASI_TRIGGER_DEFAULT_TASK = "Return the following payload exactly as-is"
DRASI_TRIGGER_DEFAULT_TOPIC_PREFIX = "drasi-events"


def drasi_trigger(
    agent: DurableAgent,
    *,
    query_id: str,
    pubsub: str | None = None,
    topic: str | None = None,
    dead_letter_topic: str | None = None,
    mapper: Callable[[DrasiUnpackedEvent], TriggerAction] | None = None,
    operation: Literal["i", "u", "d"] | Sequence[Literal["i", "u", "d"]] | None = None,
    result_model: type[Any] | None = None,
) -> None:
    """
    Wires pub/sub routes so that the target agent's workflow can be triggered by Drasi change events.
    If the agent's pub/sub component is used and a topic is provided, the topic MUST be different from the agent's topic, otherwise the activation will fail to avoid indeterministic behavior.

    Args:
        agent: The agent to target.
        query_id: The Drasi query ID whose events should trigger the agent.
        pubsub: Optional name of the Dapr pub/sub component to use. If `None`, the agent's pub/sub component is used.
        topic: Optional topic to subscribe to. If `None`, the topic is derived from the query ID as `"drasi-events-<query_id>"`.
        dead_letter_topic: Optional dead-letter topic to publish failed messages to. Defaults to `None`.
        mapper: Optional function to map Drasi events to agent task messages. If `None`, the task message will instruct the agent to return the serialized Drasi event as-is (no-op).
        operation: Optional Drasi operation(s) to filter events by (`"i"` for insert, `"u"` for update, `"d"` for delete). Defaults to `None` (no filtering).
        result_model: Optional model to use to validate the individual changes within Drasi change events. Defaults to `None` (no validation).

    Returns:
        `None`

    Raises:
        RuntimeError: If the agent's pub/sub component is used and the provided topic is the same as the agent's topic.
    """
    mapper = mapper or (
        lambda event: TriggerAction(
            task=f"{DRASI_TRIGGER_DEFAULT_TASK}: {event.model_dump_json(exclude_unset=True)}"
        )
    )
    operations = set(normalize_to_list(operation))

    filters = {
        "query_id": lambda event: (
            query_id is None or event.payload.source.queryId == query_id
        ),
        "operation": lambda event: operation is None or event.op in operations,
        # Validate both before and after states if they exist
        "result_model": lambda event: (
            result_model is None
            or (
                (
                    event.payload.before is None
                    or validate_message_model(result_model, event.payload.before)
                    is not None
                )
                and (
                    event.payload.after is None
                    or validate_message_model(result_model, event.payload.after)
                    is not None
                )
            )
        ),
    }

    def _activate(ctx: ActivationContext) -> Callable[[], None] | None:
        if (
            ctx.agent.pubsub
            and (pubsub is None or ctx.agent.message_bus_name == pubsub)
            and ctx.agent.topic_name == topic
        ):
            raise RuntimeError(
                "Pub/sub (component, topic) must be different from the agent's (component, topic)."
            )

        if mapper is None:
            logger.warning(
                "No mapper function provided; the agent will be instructed to return the serialized Drasi event as-is."
            )

        if ctx.app is not None:
            logger.info(
                "HTTP routes are not supported by this extension; only pub/sub routes will be wired."
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

    def _passes_filter(
        event: DrasiUnpackedEvent, filter_fn: Callable[[DrasiUnpackedEvent], bool]
    ) -> bool:
        try:
            return filter_fn(event)
        except Exception:
            # Filters can throw so swallow errors here
            pass
        return False

    def _on_event(
        ctx: ActivationContext, event: SubscriptionMessage
    ) -> TopicEventResponse:
        # TODO: make this more robust and separate subscription logic from validate/transform logic by extracting it out
        logger.info(f"Received Drasi event: {event!r}")

        # TODO: add deduplication for exactly-once processing
        try:
            parsed_event, metadata = parse_cloudevent(event, DrasiUnpackedEvent)
        except Exception:
            logger.warning("Cannot parse Drasi event {event!r}; dropping.")
            return TopicEventResponse(TopicEventResponseStatus.drop)

        for filter_name, filter_fn in filters.items():
            if not _passes_filter(parsed_event, filter_fn):
                # TODO: this short-circuits on the first failing filter
                # should have different log levels for expected (different op) and unexpected filter (different query ID) failures
                logger.info(
                    f"Drasi event {event!r} does not match filter '{filter_name}'; dropping."
                )
                return TopicEventResponse(TopicEventResponseStatus.drop)

        try:
            task = mapper(parsed_event)
        except Exception:
            logger.warning("Cannot map Drasi event {event!r} to agent task; dropping.")
            # Can't process this message due to erroneous user logic
            return TopicEventResponse(TopicEventResponseStatus.drop)

        try:
            # Strip optional missing fields
            agent_task = task.model_dump(exclude_unset=True)
            # TODO: move this method to shared utils
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
        # Fall back to the agent's pub/sub component if necessary
        resolved_pubsub = pubsub or ctx.agent.pubsub.pubsub_name
        # Fall back to a derived topic name if necessary
        resolved_topic = topic or f"{DRASI_TRIGGER_DEFAULT_TOPIC_PREFIX}-{query_id}"

        # Ensure pub/sub components are registered
        # TODO: should this be the runner's responsibility?
        _validate_pubsub_components(
            ctx.dapr_client,
            {resolved_pubsub},
            {resolved_pubsub: {resolved_topic}},
            ctx.runner._client_factory,
        )

        logger.info(
            f"Using pub/sub component '{resolved_pubsub}' and topic '{resolved_topic}'"
        )

        return ctx.dapr_client.subscribe_with_handler(
            pubsub_name=resolved_pubsub,
            topic=resolved_topic,
            handler_fn=lambda event: _on_event(ctx, event),
            dead_letter_topic=dead_letter_topic,
        )

    agent.add_activation(_activate)
