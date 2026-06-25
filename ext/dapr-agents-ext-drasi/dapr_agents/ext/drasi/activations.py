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

from dapr_agents import DurableAgent
from dapr_agents.agents.schemas import TriggerAction
from dapr_agents.types.activation import ActivationContext
from dapr_agents.ext.drasi.types import DrasiOperation, DrasiChangeEvent  # type: ignore[import-not-found]
from dapr_agents.ext.drasi.utils.validation import (  # type: ignore[import-not-found]
    is_change_operation,
    is_supported_operation,
    normalize_to_list,
)
from dapr_agents.types.workflow import PubSubRouteSpec
from dapr_agents.workflow.utils.core import is_supported_model
from dapr_agents.workflow.utils.registration import register_message_routes
from dapr_agents.workflow.utils.routers import validate_message_model
from dapr_agents.workflow.utils.subscription import MessageContext, ModelFilter


logger = logging.getLogger(__name__)


DRASI_TRIGGER_DEFAULT_TASK = "Return the following payload exactly as-is"
DRASI_TRIGGER_DEFAULT_TOPIC_PREFIX = "drasi-events"

DrasiTaskMapper = Callable[[DrasiChangeEvent, MessageContext], TriggerAction]


@dataclass(frozen=True)
class _DrasiTriggerConfig:
    """
    Immutable container to hold the user-supplied `drasi_trigger` configuration instead of threading arguments through multiple call sites.
    One instance is created per `drasi_trigger` invocation; must not be shared.
    """

    query_id: str
    pubsub: str | None
    topic: str | None
    dead_letter_topic: str | None
    task_mapper: DrasiTaskMapper | None
    operations: DrasiOperation | list[DrasiOperation] | tuple[DrasiOperation] | None
    result_model: type[Any] | None


def _apply_filter(
    filter_fn: Callable[[DrasiChangeEvent], bool],
    event: DrasiChangeEvent,
    ctx: MessageContext,
) -> bool:
    """
    Call the provided filter function with the validated Drasi model instance.
    Return `True` if the event passes the filter, `False` otherwise.
    """
    logger.debug(
        f"[drasi-trigger]: Applying filter '{filter_fn.__name__}' for handler '{ctx.handler_name}'"
    )
    return filter_fn(event)


def _validate_model_field(target_model: type[Any], model: Any, field_name: str) -> bool:
    """
    Return `True` if the provided model instance can be validated/coerced to the target model type,
    `False` otherwise.

    """
    try:
        return validate_message_model(target_model, model)
    except Exception:
        # Filters are independently applied. Drasi events may omit certain fields depending on the operation type,
        # so validation failures can be frequent;
        # we swallow exceptions here and log them under log level DEBUG to avoid noisy logs.
        logger.debug(
            f"[drasi-trigger]: Failed to validate '{field_name}'", exc_info=True
        )
        return False


def _make_model_filter(config: _DrasiTriggerConfig) -> ModelFilter:
    """
    Build a filter list conditionally using the **validated** configuration and
    return the post-schema validation filter for Drasi pub/sub bindings via closure.
    """
    filter_fns: list[Callable[[DrasiChangeEvent], bool]] = []

    # Deduplicate operations (preserving order) and normalize to a list for easier filtering
    normalized_operations: list[DrasiOperation] = list(
        dict.fromkeys(normalize_to_list(config.operations))
    )

    def query_id_filter(event: DrasiChangeEvent) -> bool:
        return event.payload.source.queryId == config.query_id

    filter_fns.append(query_id_filter)

    if normalized_operations:

        def operations_filter(event: DrasiChangeEvent) -> bool:
            # Ignore non-change events
            # We pass the enum value into `is_change_operation`
            # since validated events may not share the exact same operation enum type
            return (
                is_change_operation(event.op.value)
                and event.op.value in normalized_operations
            )

        filter_fns.append(operations_filter)

    if config.result_model is not None:

        def result_model_filter(event: DrasiChangeEvent) -> bool:
            return (
                event.payload.before is None
                or _validate_model_field(
                    config.result_model,
                    event.payload.before.root,
                    "before",
                )
            ) and (
                event.payload.after is None
                or _validate_model_field(
                    config.result_model,
                    event.payload.after.root,
                    "after",
                )
            )

        filter_fns.append(result_model_filter)

    def _model_filter(event: DrasiChangeEvent, ctx: MessageContext) -> bool:
        return all(_apply_filter(fn, event, ctx) for fn in filter_fns)

    return _model_filter


def _validate_config(ctx: ActivationContext, config: _DrasiTriggerConfig) -> None:
    """
    Semantically validate user-supplied configuration, logging before re-raising any exceptions
    so the agent runner can rollback the activation.
    Warn if no task mapper is provided, but otherwise accept it.

    Raises:
        RuntimeError: If the agent's pub/sub component is used and the provided topic is the same as the agent's topic.
        TypeError: If an unsupported operation type or result model is provided.
    """
    try:
        if (
            ctx.agent.pubsub
            and (config.pubsub is None or ctx.agent.message_bus_name == config.pubsub)
            and ctx.agent.topic_name == config.topic
        ):
            raise RuntimeError(
                f"Pubsub component '{config.pubsub}' and topic '{config.topic}' "
                "is identical to the agent's pubsub component and topic"
            )

        if config.task_mapper is None:
            logger.warning(
                "[drasi-trigger]: No task mapper function provided; "
                "the agent will be instructed to return the serialized Drasi event as-is."
            )

        normalized_operations = list(
            dict.fromkeys(normalize_to_list(config.operations))
        )
        for op in normalized_operations:
            if not is_supported_operation(op):
                raise TypeError(f"Unsupported operation type: '{op}'")

        if config.result_model is not None and not is_supported_model(
            config.result_model
        ):
            raise TypeError(f"Unsupported result model type: '{config.result_model}'")
    except Exception:
        logger.exception("[drasi-trigger]: Activation failed during validation")
        raise


def _build_pubsub_specs(
    ctx: ActivationContext, config: _DrasiTriggerConfig
) -> list[PubSubRouteSpec]:
    """Resolve user configuration with fallback values and return pub/sub routes."""
    resolved_pubsub = config.pubsub or ctx.agent.pubsub.pubsub_name
    resolved_topic = (
        config.topic or f"{DRASI_TRIGGER_DEFAULT_TOPIC_PREFIX}-{config.query_id}"
    )
    filter_fn = _make_model_filter(config)
    resolved_task_mapper: DrasiTaskMapper = config.task_mapper or (
        lambda event, _: TriggerAction(
            task=f"{DRASI_TRIGGER_DEFAULT_TASK}: {event.model_dump_json()}"
        )
    )

    # Create a handler with the agent workflow name registered with the workflow runtime
    # so the workflow client can target the actual agent workflow
    def _stub(*_) -> None:
        pass

    _stub.__name__ = ctx.agent.agent_workflow_name

    handler_fn = _stub

    return [
        PubSubRouteSpec(
            pubsub_name=resolved_pubsub,
            topic=resolved_topic,
            handler_fn=handler_fn,
            message_model=DrasiChangeEvent,
            dead_letter_topic=config.dead_letter_topic,
            model_filter=filter_fn,
            mapper=resolved_task_mapper,
        )
    ]


def _subscribe(
    ctx: ActivationContext, specs: list[PubSubRouteSpec]
) -> list[Callable[[], None]]:
    """Wire pub/sub routes and return closers."""
    # TODO: does this need to be publically accessible or is this even necessary
    client_factory = getattr(ctx.runner, "_client_factory", None)

    # TODO: allow users to customize concurrency and other params
    closers = register_message_routes(
        dapr_client=ctx.dapr_client,
        routes=specs,
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
    task_mapper: DrasiTaskMapper | None = None,
    operations: DrasiOperation
    | list[DrasiOperation]
    | tuple[DrasiOperation]
    | None = None,
    result_model: type[Any] | None = None,
) -> None:
    """
    Wires pub/sub routes so that the target agent's workflow can be triggered by Drasi change events.

    If the agent's pub/sub component is used and a topic is provided,
    the topic **MUST** be different from the agent's topic,
    otherwise the activation will fail to avoid indeterministic behavior.

    Args:
        agent: The agent to target.
        query_id: The Drasi query ID whose events should trigger the agent.
        pubsub: Optional name of the Dapr pub/sub component to use.
            If `None`, the agent's pub/sub component is used.
        topic: Optional topic to subscribe to.
            If `None`, the topic is derived from the query ID as `"drasi-events-<query_id>"`.
        dead_letter_topic: Optional dead-letter topic to publish failed messages to.
            Defaults to `None`.
        task_mapper: Optional callable `(DrasiChangeEvent, MessageContext) -> TriggerAction`
            to map Drasi events to agent task messages.
            If `None`, the task message will instruct the agent to return the serialized Drasi event as-is.
        operations: Optional Drasi operation(s) to filter events by
            (`"i"` for insert, `"u"` for update, `"d"` for delete).
            Defaults to `None` (no filtering).
        result_model: Optional model to use to validate the individual changes within Drasi change events.
            Defaults to `None` (no validation).

    Returns:
        `None`

    Raises:
        RuntimeError: If the agent's pub/sub component is used and the provided topic is the same as the agent's topic.
    """

    def _activate(ctx: ActivationContext) -> Callable[[], None] | None:
        """
        Activation callback that semantically validates user-supplied configuration,
        wires pub/sub routes, and returns an idempotent closer to the runner.
        """
        if ctx.app is not None:
            logger.info(
                "[drasi-trigger]: HTTP routes are not supported by this extension; only pub/sub routes will be wired."
            )

        logger.debug(
            f"[drasi-trigger]: Activation callback fired for agent '{ctx.agent.name}' with "
            f"query_id='{query_id}', "
            f"pubsub='{pubsub}', "
            f"topic='{topic}', "
            f"dead_letter_topic='{dead_letter_topic}', "
            f"operations={operations}, "
            f"result_model={result_model}"
        )

        config = _DrasiTriggerConfig(
            query_id=query_id,
            pubsub=pubsub,
            topic=topic,
            dead_letter_topic=dead_letter_topic,
            task_mapper=task_mapper,
            operations=operations,
            result_model=result_model,
        )

        _validate_config(ctx, config)
        specs = _build_pubsub_specs(ctx, config)
        closers = _subscribe(ctx, specs)

        closed = False

        def _close():
            nonlocal closed
            if closed:
                return
            closed = True
            # TODO: This is fine for now since we only wire one pub/sub route,
            # a closer raising shouldn't result in resource leaks
            for closer in closers:
                closer()

        return _close

    agent.add_activation(_activate)
