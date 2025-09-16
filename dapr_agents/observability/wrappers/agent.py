import asyncio
import logging
from typing import Any

from ..constants import (
    AGENT,
    CHAIN,
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    OPENINFERENCE_SPAN_KIND,
    Status,
    StatusCode,
    context_api,
)
from ..utils import bind_arguments, extract_content_from_result, get_input_value

logger = logging.getLogger(__name__)

try:
    from openinference.instrumentation import get_attributes_from_context
except ImportError:
    raise ImportError(
        "OpenInference not installed - please install with `pip install dapr-agents[observability]`"
    )

# ============================================================================
# Agent Execution Wrapper
# ============================================================================


class AgentRunWrapper:
    """
    Wrapper for Agent.run() method to create root AGENT spans for top-level execution.

    This wrapper instruments the primary Agent.run() method, creating the
    root AGENT span that captures the complete agent execution lifecycle
    including metadata extraction, input processing, and comprehensive tracing.

    Key features:
    - Extracts comprehensive agent metadata (name, role, goal, tools, max_iterations)
    - Handles input_data processing with automatic user query extraction
    - Supports both async and sync agent execution patterns
    - Proper error handling with span status management and exception recording
    - Phoenix UI compatibility with OpenInference attribute standards

    The wrapper creates spans with AGENT span kind and captures the complete
    agent execution flow from input processing to final result generation.
    """

    def __init__(self, tracer: Any) -> None:
        """
        Initialize the agent run wrapper.

        Args:
            tracer: OpenTelemetry tracer instance
        """
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        """
        Wrap Agent.run() method with comprehensive AGENT span tracing.

        Creates a root AGENT span that captures the complete agent execution
        lifecycle, including metadata extraction, input processing, and result
        handling with proper OpenTelemetry context propagation.

        Args:
            wrapped (callable): Original Agent.run method to be instrumented
            instance (Agent): Agent instance containing metadata (name, role, goal, tools)
            args (tuple): Positional arguments passed to Agent.run, typically (input_data,)
            kwargs (dict): Keyword arguments passed to the original method

        Returns:
            Any: Result from wrapped method execution with span attributes capturing
                 agent metadata, input queries, and execution outcomes
        """
        # Check for instrumentation suppression
        if context_api and context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return wrapped(*args, **kwargs)

        # Skip DurableAgent instances - they use WorkflowMonitorWrapper instead
        if instance.__class__.__name__ == "DurableAgent":
            return wrapped(*args, **kwargs)

        # Extract agent information for span naming and attributes
        agent_name = getattr(instance, "name", instance.__class__.__name__)
        span_name = f"{agent_name}.run"

        # Process input arguments to extract user query
        arguments = bind_arguments(wrapped, *args, **kwargs)
        input_data = arguments.get("input_data")

        # Build comprehensive span attributes
        attributes = self._build_agent_attributes(instance, agent_name, input_data)

        # Handle async vs sync execution
        if asyncio.iscoroutinefunction(wrapped):
            return self._handle_async_execution(
                wrapped, args, kwargs, span_name, attributes
            )
        else:
            return self._handle_sync_execution(
                wrapped, args, kwargs, span_name, attributes
            )

    def _build_agent_attributes(
        self, instance: Any, agent_name: str, input_data: Any
    ) -> dict:
        """
        Build comprehensive span attributes for agent execution tracing.

        Extracts detailed agent metadata and processes input data to create
        comprehensive OpenTelemetry span attributes that provide full visibility
        into agent capabilities, configuration, and execution context.

        Args:
            instance (Agent): Agent instance with metadata attributes (role, goal, tools, etc.)
            agent_name (str): Agent name extracted for span identification
            input_data (Any): Raw input data from Agent.run arguments, processed for user query extraction

        Returns:
            dict: Comprehensive span attributes including agent metadata, tool information,
                  processed input value, and OpenInference-compatible attribute structure
        """
        attributes = {
            OPENINFERENCE_SPAN_KIND: AGENT,
            INPUT_MIME_TYPE: "application/json",
            OUTPUT_MIME_TYPE: "application/json",
            "agent.name": agent_name,
            "agent.role": getattr(instance, "role", None),
            "agent.goal": getattr(instance, "goal", None),
            "agent.tools": [tool.name for tool in getattr(instance, "tools", [])],
            "agent.tools.count": len(getattr(instance, "tools", [])),
            "agent.max_iterations": getattr(instance, "max_iterations", None),
        }

        # Extract actual input value - the user's query/request
        if input_data:
            if isinstance(input_data, str):
                attributes[INPUT_VALUE] = input_data
            elif isinstance(input_data, dict):
                # Extract the actual user input from input_data
                user_input = input_data.get("input_data") or str(input_data)
                attributes[INPUT_VALUE] = user_input
            else:
                attributes[INPUT_VALUE] = str(input_data)
        else:
            attributes[INPUT_VALUE] = ""

        # Add context attributes
        attributes.update(get_attributes_from_context())

        return attributes

    def _handle_async_execution(
        self, wrapped: Any, args: Any, kwargs: Any, span_name: str, attributes: dict
    ) -> Any:
        """
        Handle asynchronous agent execution with comprehensive span tracing.

        Manages async Agent.run execution by creating spans with proper
        attribute handling, result extraction, and error management within
        the OpenTelemetry tracing context.

        Args:
            wrapped (callable): Original async Agent.run method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (e.g., "MyAgent.run")
            attributes (dict): Comprehensive span attributes including agent metadata

        Returns:
            Any: Coroutine that executes the wrapped method with proper span instrumentation,
                 result extraction, and status management
        """

        async def async_wrapper():
            with self._tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                # Store span ID for cross-process parent-child relationship
                try:
                    from ..context_propagation import extract_otel_context
                    from ..context_storage import store_workflow_context

                    # Extract current context from the AGENT span
                    current_context = extract_otel_context()
                    if current_context.get("traceparent"):
                        # Store the span ID for workflow tasks to use as parent
                        span_id = span.get_span_context().span_id
                        trace_id = span.get_span_context().trace_id

                        # Store span context with trace and span IDs
                        span_context = {
                            "trace_id": format(
                                trace_id, "032x"
                            ),  # Convert to 32-char hex string
                            "span_id": format(
                                span_id, "016x"
                            ),  # Convert to 16-char hex string
                            "traceparent": current_context.get("traceparent"),
                            "tracestate": current_context.get("tracestate", ""),
                        }

                        # Store under multiple keys for different lookup patterns
                        store_workflow_context(
                            "__current_workflow_context__", span_context
                        )
                        logger.debug(
                            f"Stored Agent span context: trace_id={format(trace_id, '032x')}, span_id={format(span_id, '016x')}"
                        )
                    else:
                        logger.warning("No traceparent found in AGENT span context")
                except Exception as e:
                    logger.warning(f"Failed to store span context: {e}")

                try:
                    result = await wrapped(*args, **kwargs)

                    # Extract and set output value
                    output_content = extract_content_from_result(result)
                    span.set_attribute(OUTPUT_VALUE, output_content)

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
        Handle synchronous agent execution with comprehensive span tracing.

        Manages sync Agent.run execution by creating spans with proper
        attribute handling, result extraction, and error management within
        the OpenTelemetry tracing context.

        Args:
            wrapped (callable): Original sync Agent.run method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (e.g., "MyAgent.run")
            attributes (dict): Comprehensive span attributes including agent metadata

        Returns:
            Any: Result from wrapped method execution with proper span instrumentation,
                 result extraction, and status management
        """
        with self._tracer.start_as_current_span(
            span_name, attributes=attributes
        ) as span:
            # Store span ID for cross-process parent-child relationship
            try:
                from ..context_propagation import extract_otel_context
                from ..context_storage import store_workflow_context

                # Extract current context from the AGENT span
                current_context = extract_otel_context()
                if current_context.get("traceparent"):
                    # Store the span ID for workflow tasks to use as parent
                    span_id = span.get_span_context().span_id
                    trace_id = span.get_span_context().trace_id

                    # Context will be stored per-instance by WorkflowMonitorWrapper
                    # to avoid cross-instance contamination
                    logger.debug(
                        f"Stored Agent span context: trace_id={format(trace_id, '032x')}, span_id={format(span_id, '016x')}"
                    )
                else:
                    logger.warning("No traceparent found in AGENT span context")
            except Exception as e:
                logger.warning(f"Failed to store span context: {e}")

            try:
                result = wrapped(*args, **kwargs)

                # Extract and set output value
                output_content = extract_content_from_result(result)
                span.set_attribute(OUTPUT_VALUE, output_content)

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


# ============================================================================
# Process Iterations Wrapper
# ============================================================================


class ProcessIterationsWrapper:
    """
    Wrapper for Agent.conversation() method to create CHAIN spans for processing logic.

    This wrapper instruments the Agent.conversation() method, creating
    CHAIN spans that capture the iterative processing and workflow execution
    within an agent's reasoning cycle.

    Key features:
    - Captures iterative processing flow and reasoning steps
    - Handles both async and sync processing patterns
    - Proper input/output value extraction and serialization
    - Error handling with span status management and exception recording
    - Maintains consistent tracing hierarchy as child of AGENT spans

    The wrapper creates spans with CHAIN span kind to represent the sequential
    processing logic that occurs during agent reasoning and decision-making.
    """

    def __init__(self, tracer: Any) -> None:
        """
        Initialize the process iterations wrapper.

        Args:
            tracer: OpenTelemetry tracer instance
        """
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        """
        Wrap Agent.conversation() method with CHAIN span tracing.

        Creates CHAIN spans that capture the iterative processing logic
        and reasoning flow within agent execution, providing visibility
        into the step-by-step processing workflow.

        Args:
            wrapped (callable): Original Agent.conversation method to be instrumented
            instance (Agent): Agent instance containing metadata and configuration
            args (tuple): Positional arguments passed to conversation method
            kwargs (dict): Keyword arguments passed to the original method

        Returns:
            Any: Result from wrapped method execution with span attributes capturing
                 processing flow, input/output values, and execution context
        """
        # Check for instrumentation suppression
        if context_api and context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return wrapped(*args, **kwargs)

        # Extract agent information for span naming
        agent_name = getattr(instance, "name", instance.__class__.__name__)
        span_name = f"{agent_name}.conversation"

        # Build span attributes
        attributes = {
            OPENINFERENCE_SPAN_KIND: CHAIN,
            INPUT_VALUE: get_input_value(wrapped, *args, **kwargs),
            INPUT_MIME_TYPE: "application/json",
            OUTPUT_MIME_TYPE: "application/json",
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
        Handle asynchronous process iterations execution with comprehensive span tracing.

        Manages async conversation execution by creating CHAIN spans with
        proper attribute handling, result extraction, and error management for
        iterative agent processing workflows.

        Args:
            wrapped (callable): Original async conversation method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (e.g., "MyAgent.conversation")
            attributes (dict): Span attributes including agent name and processing context

        Returns:
            Any: Coroutine that executes the wrapped method with proper span instrumentation,
                 capturing processing iterations and reasoning flow
        """

        async def async_wrapper():
            with self._tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                try:
                    result = await wrapped(*args, **kwargs)

                    # Extract and set output value
                    output_content = extract_content_from_result(result)
                    span.set_attribute(OUTPUT_VALUE, output_content)

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
        Handle synchronous conversation execution with comprehensive span tracing.

        Manages sync conversation execution by creating CHAIN spans with
        proper attribute handling, result extraction, and error management for
        iterative agent processing workflows.

        Args:
            wrapped (callable): Original sync conversation method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (e.g., "MyAgent.conversation")
            attributes (dict): Span attributes including agent name and processing context

        Returns:
            Any: Result from wrapped method execution with proper span instrumentation,
                 capturing processing iterations and reasoning flow
        """
        with self._tracer.start_as_current_span(
            span_name, attributes=attributes
        ) as span:
            try:
                result = wrapped(*args, **kwargs)

                # Extract and set output value
                output_content = extract_content_from_result(result)
                span.set_attribute(OUTPUT_VALUE, output_content)

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
    "AgentRunWrapper",
    "ProcessIterationsWrapper",
]
