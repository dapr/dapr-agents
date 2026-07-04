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
from dataclasses import dataclass
from typing import Any, Callable

from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.schemas import TriggerAction
from dapr_agents.types.activation import ActivationContext
from dapr_agents.types.workflow import PubSubRouteSpec
from dapr_agents.workflow.utils.core import is_supported_model
from dapr_agents.workflow.utils.registration import register_message_routes
from dapr_agents.workflow.utils.subscription import (
    MessageContext,
    ModelFilter,
    TTLDedupeBackend,
)

from dapr_agents.ext.drasi.types import DrasiOperation, DrasiChangeEvent
from dapr_agents.ext.drasi.utils.validation import (
    is_supported_operation,
    normalize_to_list,
    validate_model,
)

logger = logging.getLogger(__name__)

_DRASI_TRIGGER_DEFAULT_TASK = (
    "Return the exact JSON payload below, unmodified. "
    "Output ONLY a JSON object; no explanation, no markdown, no extra text.\n\n"
)
_DRASI_TRIGGER_DEFAULT_TOPIC_PREFIX = "drasi-events"


@dataclass(frozen=True)
class _DrasiTriggerConfig:
    """
    Immutable container to hold the resolved `drasi_trigger` configuration
    so callers don't need to thread arguments through multiple call sites.
    A single instance is created per `drasi_trigger` invocation; must not be shared.

    Note: configuration is not guaranteed to be semantically valid; validation should be
    performed after instantiation.
    """

    query_id: str
    pubsub: str | None
    topic: str | None
    dead_letter_topic: str | None
    task_mapper: Callable[[DrasiChangeEvent, MessageContext], TriggerAction] | None
    operations: list[DrasiOperation]
    change_model: type[Any] | None


def _passes_filter(
    filter_fn: Callable[[DrasiChangeEvent], bool],
    event: DrasiChangeEvent,
    ctx: MessageContext,
) -> bool:
    """
    Call the provided filter function with the validated Drasi event model.
    Return `True` if the event passes the filter, `False` otherwise.
    """
    logger.debug(
        f"[drasi-trigger]: Applying filter '{filter_fn.__name__}' for handler '{ctx.handler_name}'"
    )
    return bool(filter_fn(event))


def _is_valid_model_field(
    model: type[Any] | None,
    value: Any,
    *,
    field_name: str,
) -> bool:
    """
    Return `True` if the provided value can be validated/coerced to the target model type,
    `False` otherwise.
    """
    if model is None:
        return False
    try:
        validate_model(model, value)
        return True
    except Exception:
        # Drasi events may omit fields by operation type, making validation failures frequent.
        # Filters are applied independently with no way to validate selectively, so we swallow
        # exceptions and log at DEBUG level to reduce noise.
        logger.debug(
            f"[drasi-trigger]: Unable to validate field '{field_name}' against {model.__name__}, got {value!r}",
            exc_info=True,
        )
        return False


def _make_query_id_filter(
    config: _DrasiTriggerConfig,
) -> Callable[[DrasiChangeEvent], bool]:
    """Return a closure that filters Drasi events by the provided query ID."""

    def query_id_filter(event: DrasiChangeEvent) -> bool:
        return event.payload.source.queryId == config.query_id

    return query_id_filter


def _make_operations_filter(
    config: _DrasiTriggerConfig,
) -> Callable[[DrasiChangeEvent], bool]:
    """Return a closure that filters Drasi events by the provided operation(s)."""

    def operations_filter(event: DrasiChangeEvent) -> bool:
        # Ignore non-change events
        # We pass the enum value into `is_change_operation`
        # since validated events may not share the exact same operation enum type
        return (
            is_supported_operation(event.op.value)
            and event.op.value in config.operations
        )

    return operations_filter


def _make_change_model_filter(
    config: _DrasiTriggerConfig,
) -> Callable[[DrasiChangeEvent], bool]:
    """Return a closure that validates the change data in Drasi events using the provided change model."""

    def change_model_filter(event: DrasiChangeEvent) -> bool:
        change_model_fields: list[tuple[str, dict[str, Any]]] = []

        if event.payload.before is not None:
            change_model_fields.append(("before", event.payload.before.root))
        if event.payload.after is not None:
            change_model_fields.append(("after", event.payload.after.root))

        return all(
            _is_valid_model_field(config.change_model, data, field_name=field_name)
            for field_name, data in change_model_fields
        )

    return change_model_filter


def _make_model_filter(config: _DrasiTriggerConfig) -> ModelFilter:
    """
    Build a filter list conditionally using the **validated** configuration and
    return the post-schema validation filter for Drasi pub/sub bindings via closure.
    """
    filter_fns: list[Callable[[DrasiChangeEvent], bool]] = []

    # Mandatory filters
    filter_fns.append(_make_query_id_filter(config))

    if config.operations:
        filter_fns.append(_make_operations_filter(config))

    if config.change_model is not None:
        filter_fns.append(_make_change_model_filter(config))

    def model_filter(event: DrasiChangeEvent, ctx: MessageContext) -> bool:
        return all(_passes_filter(fn, event, ctx) for fn in filter_fns)

    return model_filter


def _validate_config(ctx: ActivationContext, config: _DrasiTriggerConfig) -> None:
    """
    Semantically validate resolved configuration, logging before re-raising any exceptions
    so the agent runner can rollback the activation.
    Warn if no task mapper is provided, but otherwise accept it.

    Raises:
        RuntimeError: If no pub/sub component can be resolved or
            the user-provided (pubsub, topic) matches the agent's (pubsub, topic).
        TypeError: If an unsupported operation type or change model type is provided.
    """
    try:
        if config.pubsub is None:
            raise RuntimeError(
                "No pub/sub component provided and the agent has no pub/sub configuration; "
                "please set `pubsub=` explicitly or provide a pub/sub configuration for the agent."
            )

        if (
            ctx.agent.pubsub
            and ctx.agent.message_bus_name == config.pubsub
            and ctx.agent.topic_name == config.topic
        ):
            raise RuntimeError(
                f"Configured (pubsub, topic): "
                f"('{config.pubsub}', topic '{config.topic}') "
                f"matches the agent's (pubsub, topic): "
                f"('{ctx.agent.message_bus_name}', topic '{ctx.agent.topic_name}'). "
                f"Please use a different topic to avoid indeterministic behavior."
            )

        if config.task_mapper is None:
            logger.warning(
                "[drasi-trigger]: No task mapper function provided; "
                "the agent will be instructed to return the serialized Drasi event as-is."
            )

        for op in config.operations:
            if not is_supported_operation(op):
                raise TypeError(f"Unsupported operation type: '{op}'")

        if config.change_model is not None and not is_supported_model(
            config.change_model
        ):
            raise TypeError(f"Unsupported change model type: '{config.change_model}'")
    except Exception:
        logger.exception("[drasi-trigger]: Activation failed during validation")
        raise


def _build_pubsub_specs(
    ctx: ActivationContext, config: _DrasiTriggerConfig
) -> list[PubSubRouteSpec]:
    """Build pub/sub routes from resolved configuration."""

    # Create a stub with the agent workflow name registered with the workflow runtime
    # so the workflow client can target the actual agent workflow
    def handler_fn(*_) -> None:
        pass

    handler_fn.__name__ = ctx.agent.agent_workflow_name

    return [
        PubSubRouteSpec(
            pubsub_name=config.pubsub,
            topic=config.topic,
            handler_fn=handler_fn,
            message_model=DrasiChangeEvent,
            dead_letter_topic=config.dead_letter_topic,
            model_filter=_make_model_filter(config),
            mapper=config.task_mapper,
        )
    ]


def _subscribe(
    ctx: ActivationContext, specs: list[PubSubRouteSpec], config: _DrasiTriggerConfig
) -> list[Callable[[], None]]:
    """Wire pub/sub routes and return closers."""
    # This allows tests to pass a mock client factory via the runner
    client_factory = getattr(ctx.runner, "_client_factory", None)

    try:
        deduper = TTLDedupeBackend()
    except ImportError:
        logger.warning(
            f"[drasi-trigger]: cachetools not installed; "
            f"disabling pub/sub message deduplication for pub/sub component {config.pubsub} and topic {config.topic}"
        )
        deduper = None

    # TODO: allow users to customize concurrency and other settings
    closers = register_message_routes(
        dapr_client=ctx.dapr_client,
        routes=specs,
        deduper=deduper,
        wf_client=ctx.wf_client,
        client_factory=client_factory,
    )

    return closers


def drasi_trigger(
    agent: DurableAgent,
    *,
    query_id: str,
    pubsub: str | None = None,
    topic: str | None = None,
    dead_letter_topic: str | None = None,
    task_mapper: Callable[[DrasiChangeEvent, MessageContext], TriggerAction]
    | None = None,
    operations: DrasiOperation | list[DrasiOperation] | None = None,
    change_model: type[Any] | None = None,
) -> None:
    """
    Wires pub/sub routes so that the target agent's workflow can be triggered by Drasi change events.

    If the agent's pub/sub component is used and a topic is provided,
    the topic **must** be different from the agent's topic,
    otherwise the activation will fail to prevent indeterministic behavior.

    Args:
        agent: The agent to target.
        query_id: The Drasi query ID whose change events should trigger the agent.
        pubsub: Optional name of the Dapr pub/sub component to use for publishing and subscriptions.
            If `None`, the agent's pub/sub component is used.
        topic: Optional topic to subscribe to.
            If `None`, the topic is derived from the query ID as `"drasi-events-<query_id>"`.
        dead_letter_topic: Optional dead-letter topic to publish failed messages to.
            Defaults to `None`.
        task_mapper: Optional callable `(DrasiChangeEvent, MessageContext) -> TriggerAction`
            to map Drasi change events to agent task messages.
            If `None`, the task message will instruct the agent to return the serialized Drasi event as-is.
        operations: Optional Drasi operation(s) to filter change events by:

            * `"i"` - Insert
            * `"u"` - Update
            * `"d"` - Delete

            Events that are filtered out will not trigger the agent. Defaults to `None` (all operations are allowed).
        change_model: Optional model to use to validate the change data in Drasi change events.
            Events that fail validation will not trigger the agent. Defaults to `None` (no validation).

    Returns:
        `None`
    """

    def _activate(ctx: ActivationContext) -> Callable[[], None]:
        """
        Activation callback that resolves and semantically validates user-provided configuration,
        wires pub/sub routes, and returns an idempotent closer to the runner.
        """
        if ctx.app is not None:
            logger.info(
                "[drasi-trigger]: HTTP routes are not supported by this extension; only pub/sub routes will be wired."
            )

        agent_name = ctx.agent.name or ctx.agent
        logger.debug(
            f"[drasi-trigger]: Activation callback fired for agent '{agent_name}' with "
            f"query_id='{query_id}', "
            f"pubsub='{pubsub}', "
            f"topic='{topic}', "
            f"dead_letter_topic='{dead_letter_topic}', "
            f"operations={operations}, "
            f"change_model={change_model}"
        )

        # Resolve user-provided configuration with documented fallback values
        resolved_pubsub = pubsub or (
            ctx.agent.pubsub.pubsub_name if ctx.agent.pubsub else None
        )
        resolved_topic = topic or f"{_DRASI_TRIGGER_DEFAULT_TOPIC_PREFIX}-{query_id}"
        resolved_task_mapper = task_mapper or (
            lambda event, _: TriggerAction(
                task=f"{_DRASI_TRIGGER_DEFAULT_TASK}: {event.model_dump_json()}"
            )
        )
        # Deduplicate and normalize operations for easier validation and filtering
        normalized_operations = list(dict.fromkeys(normalize_to_list(operations)))

        config = _DrasiTriggerConfig(
            query_id=query_id,
            pubsub=resolved_pubsub,
            topic=resolved_topic,
            dead_letter_topic=dead_letter_topic,
            task_mapper=resolved_task_mapper,
            operations=normalized_operations,
            change_model=change_model,
        )

        _validate_config(ctx, config)
        specs = _build_pubsub_specs(ctx, config)
        closers = _subscribe(ctx, specs, config)

        closed = False

        def _close():
            nonlocal closed
            if closed:
                return
            closed = True
            # This is fine for now since we only wire one pub/sub route,
            # a closer raising shouldn't result in resource leaks
            for closer in closers:
                closer()

        return _close

    agent.add_activation(_activate)
