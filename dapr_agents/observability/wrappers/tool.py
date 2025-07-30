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
    context_api,
)
from ..utils import bind_arguments, get_input_value

try:
    from openinference.instrumentation import get_attributes_from_context
except ImportError:
    raise ImportError(
        "OpenInference not installed - please install with `pip install dapr-agents[observability]`"
    )

# ============================================================================
# Batch Tool Execution Wrapper
# ============================================================================


class ExecuteToolsWrapper:
    """
    Wrapper for Agent.execute_tools() method to create TOOL spans for batch tool execution.

    This wrapper instruments the Agent.execute_tools() method, creating
    TOOL spans that capture batch tool execution operations within an agent's
    workflow with comprehensive tracing and error handling.

    Key features:
    - Captures batch tool execution flow and coordination
    - Handles both async and sync tool execution patterns
    - Proper input/output value extraction and JSON serialization
    - Error handling with span status management and exception recording
    - Agent context preservation and attribute extraction
    - Maintains compatibility with OpenInference TOOL span standards

    The wrapper creates spans with TOOL span kind to represent the batch
    coordination of multiple tool executions within agent workflows.
    """

    def __init__(self, tracer: Any) -> None:
        """
        Initialize the execute tools wrapper with OpenTelemetry tracer.

        Args:
            tracer (Any): OpenTelemetry tracer instance for creating TOOL spans
        """
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        """
        Wrap Agent.execute_tools() method with comprehensive TOOL span tracing.

        Creates TOOL spans that capture batch tool execution operations,
        including input processing, tool coordination, and result aggregation
        with proper OpenTelemetry context propagation.

        Args:
            wrapped (callable): Original Agent.execute_tools method to be instrumented
            instance (Agent): Agent instance containing tools and configuration
            args (tuple): Positional arguments passed to execute_tools method
            kwargs (dict): Keyword arguments passed to the original method

        Returns:
            Any: Result from wrapped method execution with span attributes capturing
                 batch tool execution context, inputs, and aggregated results
        """
        # Check for instrumentation suppression
        if context_api and context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return wrapped(*args, **kwargs)

        # Extract agent information for span naming
        agent_name = getattr(instance, "name", instance.__class__.__name__)
        span_name = f"{agent_name}.execute_tools"

        # Build span attributes
        attributes = {
            OPENINFERENCE_SPAN_KIND: TOOL,
            INPUT_VALUE: get_input_value(wrapped, *args, **kwargs),
            INPUT_MIME_TYPE: "application/json",
            "agent.name": agent_name,
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
        Handle asynchronous batch tool execution with comprehensive span tracing.

        Manages async execute_tools execution by creating spans with proper
        attribute handling, result aggregation, and error management for
        coordinated tool execution workflows.

        Args:
            wrapped (callable): Original async execute_tools method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (e.g., "MyAgent.execute_tools")
            attributes (dict): Span attributes including agent context and tool information

        Returns:
            Any: Coroutine that executes the wrapped method with proper span instrumentation,
                 capturing batch tool execution and result aggregation
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
                    span.record_exception(e)
                    raise

        return async_wrapper()

    def _handle_sync_execution(
        self, wrapped: Any, args: Any, kwargs: Any, span_name: str, attributes: dict
    ) -> Any:
        """
        Handle synchronous batch tool execution with comprehensive span tracing.

        Manages sync execute_tools execution by creating spans with proper
        attribute handling, result aggregation, and error management for
        coordinated tool execution workflows.

        Args:
            wrapped (callable): Original sync execute_tools method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (e.g., "MyAgent.execute_tools")
            attributes (dict): Span attributes including agent context and tool information

        Returns:
            Any: Result from wrapped method execution with proper span instrumentation,
                 capturing batch tool execution and result aggregation
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
                span.record_exception(e)
                raise


# ============================================================================
# Individual Tool Execution Wrapper
# ============================================================================


class RunToolWrapper:
    """
    Wrapper for Agent.run_tool() method to create TOOL spans for individual tool execution.

    This wrapper instruments the Agent.run_tool() method, creating individual
    TOOL spans for each tool execution within an agent's workflow with
    detailed tool identification and comprehensive tracing.

    Key features:
    - Captures individual tool execution with tool name extraction and identification
    - Handles both async and sync tool execution patterns
    - Proper input/output value extraction and JSON serialization
    - Error handling with span status management and exception recording
    - Tool name-based span naming for clear hierarchy and identification
    - Comprehensive attribute extraction including tool metadata

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
        Wrap Agent.run_tool() method with comprehensive TOOL span tracing.

        Creates individual TOOL spans for each tool execution with tool name
        identification, input/output processing, and detailed attribute capture
        for comprehensive tool execution visibility.

        Args:
            wrapped (callable): Original Agent.run_tool method to be instrumented
            instance (Agent): Agent instance containing tool configurations
            args (tuple): Positional arguments - typically (tool_name, tool_args)
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
        span_name = f"{tool_name}"

        # Build span attributes
        attributes = {
            OPENINFERENCE_SPAN_KIND: TOOL,
            INPUT_VALUE: get_input_value(wrapped, *args, **kwargs),
            INPUT_MIME_TYPE: "application/json",
            "agent.name": agent_name,
            "tool.name": tool_name,
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

        Manages async run_tool execution by creating spans with proper
        attribute handling, tool result processing, and error management for
        individual tool execution workflows.

        Args:
            wrapped (callable): Original async run_tool method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (typically the tool name)
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
                    span.record_exception(e)
                    raise

        return async_wrapper()

    def _handle_sync_execution(
        self, wrapped: Any, args: Any, kwargs: Any, span_name: str, attributes: dict
    ) -> Any:
        """
        Handle synchronous individual tool execution with comprehensive span tracing.

        Manages sync run_tool execution by creating spans with proper
        attribute handling, tool result processing, and error management for
        individual tool execution workflows.

        Args:
            wrapped (callable): Original sync run_tool method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (typically the tool name)
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
                span.record_exception(e)
                raise


# ============================================================================
# Exported Classes
# ============================================================================

__all__ = [
    "ExecuteToolsWrapper",
    "RunToolWrapper",
]
