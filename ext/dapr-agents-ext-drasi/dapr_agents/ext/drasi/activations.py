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
from typing import Any, Callable

from dapr_agents import DurableAgent
from dapr_agents.agents.schemas import TriggerAction
from dapr_agents.types.activation import ActivationContext
from dapr_agents.ext.drasi.utils.types import DrasiOperation, DrasiUnpackedEvent  # type: ignore[import-not-found]
from dapr_agents.ext.drasi.utils.validation import is_supported_operation, normalize_to_list  # type: ignore[import-not-found]
from dapr_agents.types.workflow import PubSubRouteSpec
from dapr_agents.workflow.utils.core import is_supported_model
from dapr_agents.workflow.utils.registration import register_message_routes
from dapr_agents.workflow.utils.routers import validate_message_model
from dapr_agents.workflow.utils.subscription import MessageContext, ModelFilter, validate_hook

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
    task_mapper: Callable[[DrasiUnpackedEvent, MessageContext], TriggerAction] | None = None,
    operations: DrasiOperation | list[DrasiOperation] | tuple[DrasiOperation] | None = None,
    result_model: type[Any] | None = None,
) -> None:
    """
    Wires pub/sub routes so that the target agent's workflow can be triggered by Drasi change events.
    If the agent's pub/sub component is used and a topic is provided, the topic **MUST** be different from the agent's topic,
    otherwise the activation will fail to avoid indeterministic behavior.

    Args:
        agent: The agent to target.
        query_id: The Drasi query ID whose events should trigger the agent.
        pubsub: Optional name of the Dapr pub/sub component to use. If `None`, the agent's pub/sub component is used.
        topic: Optional topic to subscribe to. If `None`, the topic is derived from the query ID as `"drasi-events-<query_id>"`.
        dead_letter_topic: Optional dead-letter topic to publish failed messages to. Defaults to `None`.
        task_mapper: Optional callable `(DrasiUnpackedEvent, MessageContext) -> TriggerAction` to map Drasi events to agent task messages.
            If `None`, the task message will instruct the agent to return the serialized Drasi event as-is (pass-through).
        operations: Optional Drasi operation(s) to filter events by (`"i"` for insert, `"u"` for update, `"d"` for delete, `"x"` for control).
            Control operations are accepted but are **currently ignored** (no-op). Defaults to `None` (no filtering).
        result_model: Optional model to use to validate the individual changes within Drasi change events. Defaults to `None` (no validation).

    Returns:
        `None`

    Raises:
        RuntimeError: If the agent's pub/sub component is used and the provided topic is the same as the agent's topic.
    """

    def _validate(ctx: ActivationContext) -> None:
        """Semantically validate configuration."""
        try:
            # Ensure pub/sub doesn't overlap with the agent's pub/sub
            if (
                ctx.agent.pubsub
                and (pubsub is None or ctx.agent.message_bus_name == pubsub)
                and ctx.agent.topic_name == topic
            ):
                raise RuntimeError(
                    f"Unable to use pubsub component '{pubsub}' and topic '{topic}' since they are identical to the agent's pubsub component and topic."
                )

            # Ensure task mapper is sync and callable if not omitted
            if task_mapper is not None:
                validate_hook(fn=task_mapper, name="task_mapper")
            else:
                logger.warning(
                    "[drasi-trigger]: No task mapper function provided; the agent will be instructed to return the serialized Drasi event as-is."
                )

            if operations is not None:
                ops = normalize_to_list(operations)
                for op in ops:
                    if not is_supported_operation(op):
                        raise TypeError(f"Unsupported operation type: '{op}'")

            if result_model is not None and not is_supported_model(result_model):
                raise TypeError(f"Unsupported result model type: '{result_model}'")
        except Exception as e:
            logger.exception(f"[drasi-trigger]: Activation failed during validation: {e}")
            raise

    def _make_model_filter() -> ModelFilter:
        """Build a filter mapping conditionally and return the post-schema validation filter for Drasi pub/sub bindings via closure."""
        filters: dict[str, Callable[[DrasiUnpackedEvent], bool]] = {
            "query_id": lambda event: event.payload.source.queryId == query_id
        }

        if operations is not None:
            accepted_operations: set[DrasiOperation] = set(normalize_to_list(operations))
            # Control events are always excluded
            accepted_operations.discard("x")
            filters["operations"] = lambda event: event.op in accepted_operations
        if result_model is not None:
            filters["result_model"] = lambda event: (
                (event.payload.before is None or validate_message_model(result_model, event.payload.before) is not None)
                and (event.payload.after is None or validate_message_model(result_model, event.payload.after) is not None)
            )

        def _model_filter(event: DrasiUnpackedEvent, ctx: MessageContext) -> bool:
            passes_filter = True
            for filter_name, filter_fn in filters.items():
                logger.info(f"[drasi-trigger]: Applying filter '{filter_name}' for handler '{ctx.handler_name}'")
                passes_filter = passes_filter and filter_fn(event)

            return passes_filter

        return _model_filter
    
    def _make_specs() -> PubSubRouteSpec:
        """Resolve user configuration and build pub/sub routes."""
        resolved_pubsub = pubsub or agent.pubsub.pubsub_name
        resolved_topic = topic or f"{DRASI_TRIGGER_DEFAULT_TOPIC_PREFIX}-{query_id}"

        # Create a handler with the agent workflow name registered with the workflow runtime
        # so the workflow client can target the actual agent workflow
        registered_name = agent.agent_workflow_name
        def _stub(*_) -> None:
            pass
        _stub.__name__ = registered_name
        handler_fn = _stub

        # TODO: only unpacked events are currently accepted
        message_model = DrasiUnpackedEvent

        filter_fn = _make_model_filter()
        resolved_task_mapper = task_mapper or (
            lambda event, _: TriggerAction(
                task=f"{DRASI_TRIGGER_DEFAULT_TASK}: {event.model_dump_json()}"
            )
        )

        return [
            PubSubRouteSpec(
                pubsub_name=resolved_pubsub,
                topic=resolved_topic,
                handler_fn=handler_fn,
                message_model=message_model,
                dead_letter_topic=dead_letter_topic,
                model_filter=filter_fn,
                mapper=resolved_task_mapper,
            )
        ]

    def _open_stream(ctx: ActivationContext) -> list[Callable[[], None]]:
        """Wire pub/sub routes and return closers."""
        specs = _make_specs()

        # TODO: only sync delivery is currently supported
        client_factory = getattr(ctx.runner, "_client_factory", None)
        closers = register_message_routes(
            dapr_client=ctx.dapr_client,
            routes=specs,
            wf_client=ctx.wf_client,
            client_factory=client_factory,
        )

        return closers

    def _activate(ctx: ActivationContext) -> Callable[[], None] | None:
        """Activation callback that semantically validates configuration, wires pub/sub routes, and returns an idempotent closer to the runner."""
        if ctx.app is not None:
            logger.info(
                "[drasi-trigger]: HTTP routes are not supported by this extension; only pub/sub routes will be wired."
            )

        _validate(ctx)
        closers = _open_stream(ctx)

        closed = False

        def _close():
            nonlocal closed
            if closed:
                return
            closed = True
            # Propagate errors to the runner
            for closer in closers:
                closer()

        return _close

    agent.add_activation(_activate)
