import asyncio
import functools
import json
import logging
from typing import Any, Dict, Optional

from ..constants import (
    AGENT,
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    Status,
    StatusCode,
    WORKFLOW_RUN_SUPPRESSION_KEY,
    context_api,
    safe_json_dumps,
)
from ..context_propagation import extract_otel_context
from ..context_storage import store_workflow_context
from ..utils import bind_arguments
from openinference.instrumentation import get_attributes_from_context

logger = logging.getLogger(__name__)


def _extract_agent_metadata(workflow: Any) -> Dict[str, Any]:
    """Best-effort extraction of agent metadata from the bound workflow callable."""

    agent = getattr(workflow, "__self__", None)
    if agent is None:
        return {}

    details: Dict[str, Any] = {}
    details["agent.name"] = getattr(agent, "name", agent.__class__.__name__)

    profile = getattr(agent, "profile", None)
    role = getattr(profile, "role", getattr(agent, "role", None))
    goal = getattr(profile, "goal", getattr(agent, "goal", None))
    if role is not None:
        details["agent.role"] = role
    if goal is not None:
        details["agent.goal"] = goal

    execution = getattr(agent, "execution", None)
    if execution is not None:
        max_iterations = getattr(execution, "max_iterations", None)
        if max_iterations is not None:
            details["agent.execution.max_iterations"] = max_iterations

    return details


def _resolve_payload(arguments: Dict[str, Any], args: Any) -> Optional[Any]:
    payload = arguments.get("payload")
    if payload is not None:
        return payload
    # Method signature: run_workflow(self, workflow, payload=None, ...)
    if len(args) >= 2:
        return args[1]
    return None


def _normalize_output(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:  # noqa: BLE001
            return value
    return value


class WorkflowRunWrapper:
    """
    Wraps ``WorkflowRunner.run_workflow`` to emit an AGENT span per scheduled instance.

    This covers orchestrators (detach=True) as well as durable agents (detach=False),
    captures payload metadata, and persists the W3C context for downstream activities.
    """

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        if context_api:
            if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
                bound = wrapped.__get__(instance, type(instance))
                return bound(*args, **kwargs)
            if context_api.get_value(WORKFLOW_RUN_SUPPRESSION_KEY):
                bound = wrapped.__get__(instance, type(instance))
                return bound(*args, **kwargs)

        arguments = bind_arguments(wrapped, instance, *args, **kwargs)
        workflow = arguments.get("workflow")
        workflow_name = self._infer_workflow_name(workflow)
        agent_details = _extract_agent_metadata(workflow)
        agent_name = agent_details.get(
            "agent.name", getattr(instance, "name", instance.__class__.__name__)
        )
        span_name = f"{agent_name}.{workflow_name}"
        attributes = self._build_attributes(
            instance,
            workflow_name,
            payload=_resolve_payload(arguments, args),
            agent_details=agent_details,
        )

        with self._tracer.start_as_current_span(
            span_name, attributes=attributes
        ) as span:
            bound = wrapped.__get__(instance, type(instance))
            try:
                instance_id = bound(*args, **kwargs)
                if instance_id:
                    span.set_attribute("workflow.instance_id", instance_id)
                    self._store_context(instance_id)
                span.set_status(Status(StatusCode.OK))
                return instance_id
            except Exception as exc:  # noqa: BLE001
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                raise

    @staticmethod
    def _infer_workflow_name(workflow: Any) -> str:
        if isinstance(workflow, str):
            return workflow
        for attr in ("_workflow_name", "__name__", "name"):
            if hasattr(workflow, attr):
                value = getattr(workflow, attr)
                if value:
                    return value
        return "AgenticWorkflow"

    def _build_attributes(
        self,
        runner: Any,
        workflow_name: str,
        payload: Optional[Any],
        agent_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        attributes = {
            OPENINFERENCE_SPAN_KIND: AGENT,
            INPUT_MIME_TYPE: "application/json",
            OUTPUT_MIME_TYPE: "application/json",
            "workflow.name": workflow_name,
            "workflow.operation": "run",
            "agent.name": agent_details.get(
                "agent.name", getattr(runner, "name", runner.__class__.__name__)
            ),
        }
        attributes[INPUT_VALUE] = safe_json_dumps(payload or {})
        for key in ("agent.role", "agent.goal", "agent.execution.max_iterations"):
            if key in agent_details and agent_details[key] is not None:
                attributes[key] = agent_details[key]
        attributes.update(get_attributes_from_context())
        return attributes

    @staticmethod
    def _store_context(instance_id: str) -> None:
        context = extract_otel_context()
        if not context or not instance_id:
            return
        key = f"__workflow_context_{instance_id}__"
        store_workflow_context(key, context)


class WorkflowMonitorWrapper:
    """
    Wraps ``WorkflowRunner.run_workflow_async`` to emit AGENT spans for monitored runs.

    When ``detach`` is False (DurableAgent semantics), this represents the full lifecycle
    of a workflow run (schedule + wait). When detach=True the wrapper is suppressed so
    orchestrators do not double-log spans.
    """

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        arguments = bind_arguments(wrapped, instance, *args, **kwargs)
        detach = bool(arguments.get("detach"))
        if detach:
            # Orchestrator path – just invoke underlying method.
            bound = wrapped.__get__(instance, type(instance))
            return bound(*args, **kwargs)

        workflow = arguments.get("workflow")
        workflow_name = WorkflowRunWrapper._infer_workflow_name(workflow)
        agent_details = _extract_agent_metadata(workflow)
        agent_name = agent_details.get(
            "agent.name", getattr(instance, "name", instance.__class__.__name__)
        )
        span_name = f"{agent_name}.{workflow_name}.monitor"
        payload = _resolve_payload(arguments, args)

        attributes = {
            OPENINFERENCE_SPAN_KIND: AGENT,
            "workflow.name": workflow_name,
            "workflow.operation": "run_and_wait",
            "agent.name": agent_name,
            INPUT_MIME_TYPE: "application/json",
            INPUT_VALUE: safe_json_dumps(payload or {}),
        }
        for key in ("agent.role", "agent.goal", "agent.execution.max_iterations"):
            if key in agent_details and agent_details[key] is not None:
                attributes[key] = agent_details[key]
        attributes.update(get_attributes_from_context())

        suppress_token = None
        if context_api:
            suppress_token = context_api.attach(
                context_api.set_value(WORKFLOW_RUN_SUPPRESSION_KEY, True)
            )

        async def _async_call():
            bound = wrapped.__get__(instance, type(instance))
            with self._tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                try:
                    result = await bound(*args, **kwargs)
                    span.set_attribute(
                        OUTPUT_VALUE, safe_json_dumps(_normalize_output(result))
                    )
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:  # noqa: BLE001
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise

        try:
            if asyncio.iscoroutinefunction(wrapped):
                return _async_call()
            bound = wrapped.__get__(instance, type(instance))
            with self._tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                try:
                    result = bound(*args, **kwargs)
                    span.set_attribute(
                        OUTPUT_VALUE, safe_json_dumps(_normalize_output(result))
                    )
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:  # noqa: BLE001
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise
        finally:
            if suppress_token is not None and context_api:
                context_api.detach(suppress_token)


__all__ = [
    "WorkflowRunWrapper",
    "WorkflowMonitorWrapper",
]
