import logging
import asyncio
from typing import Any, Dict

from ..constants import (
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    Status,
    StatusCode,
    context_api,
    safe_json_dumps,
)
from ..utils import bind_arguments

try:
    from openinference.instrumentation import get_attributes_from_context
except ImportError:
    raise ImportError(
        "OpenInference not installed - please install with `pip install dapr-agents[observability]`"
    )

logger = logging.getLogger(__name__)

# ============================================================================
# Workflow Run Wrapper
# ============================================================================


class WorkflowRunWrapper:
    """
    Wrapper for WorkflowApp.run_workflow method.

    This wrapper instruments the fire-and-forget workflow execution method used by
    Orchestrators to start new Dapr Workflow instances. Unlike DurableAgent which
    uses run_and_monitor_workflow_async, Orchestrators use run_workflow directly
    for event-driven workflow initiation without waiting for completion.

    Features:
    - Workflow instance ID tracking (primary output for Orchestrators)
    - Input payload capture (trigger action/message for orchestrator)
    - Workflow name and metadata extraction
    - Agent-level span for orchestrator workflow starts
    - Integration with Dapr Workflow runtime

    Note: This creates an AGENT span because for Orchestrators, the workflow
    start represents the agent's response to an external trigger (HTTP/PubSub).
    """

    def __init__(self, tracer: Any) -> None:
        """
        Initialize the workflow run wrapper.

        Args:
            tracer: OpenTelemetry tracer instance
        """
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        """
        Wrap WorkflowApp.run_workflow with AGENT span tracing.

        Args:
            wrapped: Original WorkflowApp.run_workflow method
            instance: WorkflowApp instance (Orchestrator classes)
            args: Positional arguments (workflow, input)
            kwargs: Keyword arguments

        Returns:
            str: Workflow instance ID from wrapped method execution
        """
        # Check for instrumentation suppression
        if context_api and context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return wrapped(*args, **kwargs)

        # Extract arguments
        arguments = bind_arguments(wrapped, *args, **kwargs)
        workflow = arguments.get("workflow")

        # Extract workflow name
        workflow_name = (
            workflow
            if isinstance(workflow, str)
            else getattr(workflow, "__name__", "unknown_workflow")
        )

        # Build span attributes
        attributes = self._build_workflow_attributes(instance, workflow_name, arguments)

        # Debug logging to confirm wrapper is being called
        logger.debug(
            f"🔍 WorkflowRunWrapper creating AGENT span for workflow: {workflow_name}"
        )

        # Create AGENT span (this IS the agent execution for workflow-based agents)
        span_name = f"Agent.{workflow_name}"

        with self._tracer.start_as_current_span(
            span_name, attributes=attributes
        ) as span:
            # Debug logging to confirm span creation
            logger.debug(f"✅ Created AGENT span: {span_name}")
            logger.debug(f"📋 Span context: {span.get_span_context()}")

            try:
                # Execute the workflow start
                instance_id = wrapped(*args, **kwargs)

                # Set output attributes
                span.set_attribute(
                    OUTPUT_VALUE, safe_json_dumps({"instance_id": instance_id})
                )
                span.set_attribute("workflow.instance_id", instance_id)

                span.set_status(Status(StatusCode.OK))
                logger.debug(f"🎯 AGENT span completed successfully: {span_name}")
                return instance_id

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"❌ AGENT span failed: {span_name} - {e}")
                raise

    def _build_workflow_attributes(
        self, instance: Any, workflow_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build span attributes for orchestrator workflow execution.

        Args:
            instance: WorkflowApp instance (Orchestrator)
            workflow_name: Name of the workflow being started
            arguments: Bound method arguments from the wrapped call

        Returns:
            Dict[str, Any]: Span attributes for the AGENT span
        """
        agent_name = getattr(instance, "name", instance.__class__.__name__)

        attributes = {
            "openinference.span.kind": "AGENT",  # Orchestrator workflow start is an agent action
            INPUT_MIME_TYPE: "application/json",
            OUTPUT_MIME_TYPE: "application/json",
            "workflow.name": workflow_name,
            "workflow.operation": "run",
            "agent.name": agent_name,
            "agent.type": instance.__class__.__name__,
        }

        # Add workflow runtime info if available
        if hasattr(instance, "wf_runtime_is_running"):
            attributes["workflow.runtime_running"] = instance.wf_runtime_is_running

        # Serialize input arguments
        attributes[INPUT_VALUE] = safe_json_dumps(arguments)

        # Add context attributes
        attributes.update(get_attributes_from_context())

        return attributes


# ============================================================================
# Workflow Monitor Wrapper
# ============================================================================


class WorkflowMonitorWrapper:
    """
    Wrapper for WorkflowApp.run_and_monitor_workflow_async method.

    This wrapper instruments the async workflow execution and monitoring method
    that manages the complete workflow lifecycle from start to completion.
    For DurableAgent, this represents the top-level AGENT execution that includes
    both starting the workflow and waiting for its completion.

    Features:
    - Async workflow execution tracking
    - Workflow state monitoring
    - Complete lifecycle management (start to finish)
    - Error handling for workflow failures
    - Result capture and serialization
    - Agent-level span for complete workflow execution

    Note: This creates an AGENT span because it represents the complete
    agent execution cycle for workflow-based agents.
    """

    def __init__(self, tracer: Any) -> None:
        """
        Initialize the workflow monitor wrapper.

        Args:
            tracer: OpenTelemetry tracer instance
        """
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        """
        Wrap WorkflowApp.run_and_monitor_workflow_async with AGENT span tracing.

        Creates the top-level AGENT span for DurableAgent workflow execution and
        stores the global workflow context immediately for task correlation.

        Args:
            wrapped (callable): Original WorkflowApp.run_and_monitor_workflow_async method
            instance (Any): WorkflowApp instance (DurableAgent)
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method

        Returns:
            Any: Result from wrapped method execution (workflow output)
        """
        # Check for instrumentation suppression
        if context_api and context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return wrapped(*args, **kwargs)

        workflow_name = self._extract_workflow_name(args, kwargs)
        # Extract agent name from the instance
        agent_name = getattr(instance, "name", "DurableAgent")
        attributes = self._build_workflow_attributes(
            workflow_name, agent_name, args, kwargs
        )

        # Store global context immediately when this method is called
        # This ensures workflow tasks can access it before async execution begins
        try:
            from ..context_propagation import extract_otel_context
            from ..context_storage import store_workflow_context

            captured_context = extract_otel_context()
            if captured_context.get("traceparent"):
                logger.debug(
                    f"Captured traceparent: {captured_context.get('traceparent')}"
                )

                store_workflow_context("__global_workflow_context__", captured_context)
            else:
                logger.warning(
                    "No traceparent found in captured context during __call__"
                )
        except Exception as e:
            logger.warning(f"Failed to capture/store workflow context in __call__: {e}")

        # Handle async vs sync execution
        if asyncio.iscoroutinefunction(wrapped):
            return self._handle_async_execution(
                wrapped, args, kwargs, attributes, workflow_name, agent_name
            )
        else:
            return self._handle_sync_execution(
                wrapped, args, kwargs, attributes, workflow_name, agent_name
            )

    def _extract_workflow_name(self, args: Any, kwargs: Any) -> str:
        """
        Extract workflow name from method arguments.

        Args:
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            str: Workflow name
        """
        if args and len(args) > 0:
            workflow = args[0]
        else:
            workflow = kwargs.get("workflow")

        # Extract workflow name
        workflow_name = (
            workflow
            if isinstance(workflow, str)
            else getattr(workflow, "__name__", "unknown_workflow")
        )

        return workflow_name

    def _build_workflow_attributes(
        self, workflow_name: str, agent_name: str, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """
        Build span attributes for workflow execution.

        Args:
            workflow_name: Name of the workflow
            agent_name: Name of the agent
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Dict[str, Any]: Span attributes for the AGENT span
        """
        # Build basic attributes
        attributes = {
            "openinference.span.kind": "AGENT",  # DurableAgent workflow execution is the agent action
            "workflow.name": workflow_name,
            "agent.execution_mode": "workflow_based",
            "agent.name": agent_name,
            OUTPUT_MIME_TYPE: "application/json",
        }

        # Add input payload if available
        if args and len(args) > 1:
            # Second argument is typically the input
            input_data = args[1]
            if input_data is not None:
                attributes[INPUT_VALUE] = safe_json_dumps(input_data)
                attributes[INPUT_MIME_TYPE] = "application/json"
        elif "input" in kwargs and kwargs["input"] is not None:
            attributes[INPUT_VALUE] = safe_json_dumps(kwargs["input"])
            attributes[INPUT_MIME_TYPE] = "application/json"

        # Add context attributes
        attributes.update(get_attributes_from_context())

        return attributes

    def _handle_async_execution(
        self,
        wrapped: Any,
        args: Any,
        kwargs: Any,
        attributes: Dict[str, Any],
        workflow_name: str,
        agent_name: str,
    ) -> Any:
        """
        Handle async workflow monitoring execution with context propagation.

        Args:
            wrapped: Original async method to execute
            args: Positional arguments for the wrapped method
            kwargs: Keyword arguments for the wrapped method
            attributes: Pre-built span attributes
            workflow_name: Name of the workflow for span naming

        Returns:
            Coroutine: Async wrapper function for execution
        """

        async def async_wrapper():
            span_name = f"Agent.{workflow_name}"

            # Debug logging to confirm span creation
            logger.debug(f"Creating AGENT span: {span_name}")

            with self._tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                try:
                    # Execute workflow and get result
                    result = await wrapped(*args, **kwargs)

                    # Set output attributes - handle both string and object results consistently
                    if isinstance(result, str):
                        # If result is already a JSON string, use it directly
                        span.set_attribute(OUTPUT_VALUE, result)
                    else:
                        # If result is an object, serialize it to match input format
                        span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result))

                    span.set_status(Status(StatusCode.OK))
                    logger.debug(f"AGENT span completed successfully: {span_name}")
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    logger.error(f"AGENT span failed: {span_name} - {e}")
                    raise

        return async_wrapper()

    def _handle_sync_execution(
        self,
        wrapped: Any,
        args: Any,
        kwargs: Any,
        attributes: Dict[str, Any],
        workflow_name: str,
        agent_name: str,
    ) -> Any:
        """
        Handle synchronous workflow monitoring execution with context propagation.

        Args:
            wrapped: Original synchronous method to execute
            args: Positional arguments for the wrapped method
            kwargs: Keyword arguments for the wrapped method
            attributes: Pre-built span attributes
            workflow_name: Name of the workflow for span naming

        Returns:
            Any: Result from wrapped method execution
        """
        span_name = f"Agent.{workflow_name}"

        # Debug logging to confirm span creation
        logger.debug(f"Creating AGENT span: {span_name}")

        with self._tracer.start_as_current_span(
            span_name, attributes=attributes
        ) as span:
            try:
                # Execute workflow and get result
                result = wrapped(*args, **kwargs)

                # Set output attributes - handle both string and object results consistently
                if isinstance(result, str):
                    # If result is already a JSON string, use it directly
                    span.set_attribute(OUTPUT_VALUE, result)
                else:
                    # If result is an object, serialize it to match input format
                    span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result))

                span.set_status(Status(StatusCode.OK))
                logger.debug(f"AGENT span completed successfully: {span_name}")
                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"AGENT span failed: {span_name} - {e}")
                raise


# ============================================================================
# Exported Classes
# ============================================================================

__all__ = [
    "WorkflowRunWrapper",
    "WorkflowMonitorWrapper",
]
