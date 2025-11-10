import asyncio
import functools
import logging
from typing import Any, Callable, Optional

from ..constants import (
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    Status,
    StatusCode,
    TASK,
    TOOL,
    LLM,
)
from ..context_propagation import create_child_span_with_context
from ..context_storage import get_workflow_context
from ..utils import safe_json_dumps

logger = logging.getLogger(__name__)


class WorkflowActivityRegistrationWrapper:
    """
    Wraps `WorkflowRuntime.register_activity` so every activity executes inside a span.

    The Dapr Workflow runtime invokes the registered activity callable directly,
    so we wrap the callable before it is registered to capture span metadata,
    restore the stored parent trace context, and emit TOOL/LLM/TASK spans.
    """

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        if not args:
            bound = wrapped.__get__(instance, type(instance))
            return bound(*args, **kwargs)

        activity_fn = args[0]
        if not callable(activity_fn):
            bound = wrapped.__get__(instance, type(instance))
            return bound(*args, **kwargs)

        if getattr(activity_fn, "_dapr_agents_instrumented", False):
            instrumented_fn = activity_fn
        else:
            instrumented_fn = self._wrap_activity(activity_fn)
            setattr(instrumented_fn, "_dapr_agents_instrumented", True)

        new_args = (instrumented_fn,) + tuple(args[1:])
        bound_method = wrapped.__get__(instance, type(instance))
        return bound_method(*new_args, **kwargs)

    # ------------------------------------------------------------------ #
    # Activity instrumentation
    # ------------------------------------------------------------------ #

    def _wrap_activity(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        agent_instance = getattr(fn, "__self__", None)
        agent_name = getattr(
            agent_instance,
            "name",
            agent_instance.__class__.__name__ if agent_instance else "WorkflowActivity",
        )
        activity_name = getattr(fn, "__name__", "activity")
        span_kind = self._determine_span_kind(activity_name)
        span_name = f"{agent_name}.{activity_name}"

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*activity_args, **activity_kwargs):
                ctx, payload = self._extract_ctx_and_payload(
                    activity_args, activity_kwargs
                )
                instance_id = self._extract_instance_id(ctx)
                otel_context = self._resolve_context(instance_id)
                attributes = self._build_attributes(
                    agent_name, activity_name, instance_id, span_kind, payload
                )

                with create_child_span_with_context(
                    self._tracer, span_name, otel_context, attributes
                ) as span:
                    try:
                        result = await fn(*activity_args, **activity_kwargs)
                        self._record_output(span, result)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as exc:  # noqa: BLE001
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        span.record_exception(exc)
                        raise

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*activity_args, **activity_kwargs):
            ctx, payload = self._extract_ctx_and_payload(activity_args, activity_kwargs)
            instance_id = self._extract_instance_id(ctx)
            otel_context = self._resolve_context(instance_id)
            attributes = self._build_attributes(
                agent_name, activity_name, instance_id, span_kind, payload
            )

            with create_child_span_with_context(
                self._tracer, span_name, otel_context, attributes
            ) as span:
                try:
                    result = fn(*activity_args, **activity_kwargs)
                    self._record_output(span, result)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:  # noqa: BLE001
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise

        return sync_wrapper

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_ctx_and_payload(args: Any, kwargs: Any) -> tuple[Optional[Any], Any]:
        ctx = args[0] if args else kwargs.get("ctx")
        payload = None
        if len(args) > 1:
            payload = args[1]
        elif "payload" in kwargs:
            payload = kwargs["payload"]
        else:
            payload = kwargs.get("inp")
        return ctx, payload

    @staticmethod
    def _extract_instance_id(ctx: Any) -> Optional[str]:
        if ctx is None:
            return None
        for attr in ("instance_id", "workflow_id"):
            if hasattr(ctx, attr):
                value = getattr(ctx, attr)
                if value:
                    return value
        if hasattr(ctx, "get_inner_context"):
            try:
                inner = ctx.get_inner_context()
                return getattr(inner, "workflow_id", None)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Failed to extract workflow_id from inner context", exc_info=True
                )
        return None

    @staticmethod
    def _determine_span_kind(activity_name: str) -> str:
        lowered = activity_name.lower()
        if "llm" in lowered or "call_llm" in lowered or "generate" in lowered:
            return LLM
        if "tool" in lowered:
            return TOOL
        return TASK

    def _build_attributes(
        self,
        agent_name: str,
        activity_name: str,
        instance_id: Optional[str],
        span_kind: str,
        payload: Any,
    ) -> dict:
        attributes = {
            OPENINFERENCE_SPAN_KIND: span_kind,
            INPUT_MIME_TYPE: "application/json",
            OUTPUT_MIME_TYPE: "application/json",
            "agent.name": agent_name,
            "workflow.activity": activity_name,
        }
        if instance_id:
            attributes["workflow.instance_id"] = instance_id
        if payload is not None:
            attributes[INPUT_VALUE] = safe_json_dumps(payload)
        return attributes

    @staticmethod
    def _record_output(span: Any, result: Any) -> None:
        if span is None:
            return
        if asyncio.iscoroutine(result):
            # Should not happen because we await before calling
            return
        span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result))

    @staticmethod
    def _resolve_context(instance_id: Optional[str]) -> Optional[dict]:
        context = None
        if instance_id:
            key = f"__workflow_context_{instance_id}__"
            context = get_workflow_context(key)

        if context is None:
            context = get_workflow_context("__current_workflow_context__")
            if context:
                logger.debug(
                    "Using fallback workflow context for instance %s", instance_id
                )

        return context


__all__ = ["WorkflowActivityRegistrationWrapper"]
