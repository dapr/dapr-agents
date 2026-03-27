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

import asyncio
from typing import Any

from ..constants import (
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    Status,
    StatusCode,
    TOOL,
    GEN_AI_OPERATION_NAME,
    GEN_AI_TOOL_NAME,
    GenAiOperationNameValues,
    context_api,
)
from ..utils import bind_arguments, get_input_value

from openinference.instrumentation import get_attributes_from_context

# ============================================================================
# Individual Tool Execution Wrapper
# ============================================================================


class RunToolWrapper:
    """
    Wrapper for `AgentToolExecutor.run_tool()` to emit TOOL spans per tool call.

    This wrapper instruments the executor layer, creating a span for each
    tool function invocation regardless of which agent issued it.

    Key features:
    - Captures individual tool execution with tool name extraction and identification
    - Handles both async and sync tool execution patterns
    - Proper input/output value extraction and JSON serialization
    - Error handling with span status management and exception recording
    - Tool name-based span naming for clear hierarchy and identification
    - Comprehensive attribute extraction including tool metadata
    - GenAI semconv dual-emit (gen_ai.operation.name=execute_tool, gen_ai.tool.name)

    The wrapper creates spans with TOOL span kind and uses the actual tool
    name as the span name for clear identification in trace hierarchies.
    """

    def __init__(self, tracer: Any) -> None:
        """
        Initialize the run tool wrapper with OpenTelemetry tracer.

        Args:
            tracer (Any): OpenTelemetry tracer instance for creating individual TOOL spans
        """
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        """
        Wrap the executor method with TOOL span tracing.

        Args:
            wrapped (callable): Original `AgentToolExecutor.run_tool`
            instance (AgentToolExecutor): Executor instance
            args (tuple): Positional arguments - typically (tool_name, *tool_args)
            kwargs (dict): Keyword arguments passed to the original method

        Returns:
            Any: Result from wrapped method execution with span attributes capturing
                 individual tool execution context, inputs, outputs, and tool metadata
        """
        # Check for instrumentation suppression
        if context_api and context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return wrapped(*args, **kwargs)

        # Extract agent and tool information for span naming
        agent_name = getattr(instance, "name", instance.__class__.__name__)
        arguments = bind_arguments(wrapped, *args, **kwargs)
        tool_name = arguments.get("tool_name", args[0] if args else "unknown_tool")
        span_name = f"execute_tool {tool_name}"

        # Build span attributes
        attributes = {
            OPENINFERENCE_SPAN_KIND: TOOL,
            INPUT_VALUE: get_input_value(wrapped, *args, **kwargs),
            INPUT_MIME_TYPE: "application/json",
            "agent.name": agent_name,
            "tool.name": tool_name,
            # GenAI semconv
            GEN_AI_OPERATION_NAME: GenAiOperationNameValues.EXECUTE_TOOL,
            GEN_AI_TOOL_NAME: tool_name,
        }

        # Add context attributes
        attributes.update(get_attributes_from_context())

        # Handle async vs sync execution
        if asyncio.iscoroutinefunction(wrapped):
            return self._handle_async_execution(
                wrapped, args, kwargs, span_name, attributes
            )
        else:
            return self._handle_sync_execution(
                wrapped, args, kwargs, span_name, attributes
            )

    def _handle_async_execution(
        self, wrapped: Any, args: Any, kwargs: Any, span_name: str, attributes: dict
    ) -> Any:
        """
        Handle asynchronous individual tool execution with comprehensive span tracing.

        Args:
            wrapped (callable): Original async run_tool method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (e.g., "execute_tool search")
            attributes (dict): Span attributes including agent context and tool metadata

        Returns:
            Any: Coroutine that executes the wrapped method with proper span instrumentation,
                 capturing individual tool execution and result processing
        """

        async def async_wrapper():
            with self._tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                try:
                    result = await wrapped(*args, **kwargs)
                    span.set_attribute(OUTPUT_VALUE, str(result))
                    span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("error.type", type(e).__qualname__)
                    span.record_exception(e)
                    raise

        return async_wrapper()

    def _handle_sync_execution(
        self, wrapped: Any, args: Any, kwargs: Any, span_name: str, attributes: dict
    ) -> Any:
        """
        Handle synchronous individual tool execution with comprehensive span tracing.

        Args:
            wrapped (callable): Original sync run_tool method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (e.g., "execute_tool search")
            attributes (dict): Span attributes including agent context and tool metadata

        Returns:
            Any: Result from wrapped method execution with proper span instrumentation,
                 capturing individual tool execution and result processing
        """
        with self._tracer.start_as_current_span(
            span_name, attributes=attributes
        ) as span:
            try:
                result = wrapped(*args, **kwargs)
                span.set_attribute(OUTPUT_VALUE, str(result))
                span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error.type", type(e).__qualname__)
                span.record_exception(e)
                raise


# ============================================================================
# Exported Classes
# ============================================================================

__all__ = [
    "RunToolWrapper",
]
