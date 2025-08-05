import asyncio
import logging
from typing import Any, Dict, Optional

from ..constants import (
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    LLM,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    Status,
    StatusCode,
    TASK,
    TOOL,
    context_api,
    safe_json_dumps,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Workflow Task Wrapper
# ============================================================================


class WorkflowTaskWrapper:
    """
    Wrapper for WorkflowTask.__call__ method.

    This wrapper instruments the actual execution layer of workflow tasks where
    individual activities like generate_response, run_tool, etc. are executed
    by the Dapr Workflow runtime.

    Key features:
    - Task type detection for appropriate span kinds
    - Instance ID tracking from WorkflowActivityContext
    - Proper span hierarchy within workflow traces
    - Support for LLM, Tool, and Agent-based tasks

    Span kinds by task type:
    - LLM-based tasks (generate_response) → LLM span
    - Tool execution tasks (run_tool) → TOOL span
    - Other workflow tasks → TASK span
    """

    def __init__(self, tracer: Any) -> None:
        """
        Initialize the workflow task wrapper.

        Args:
            tracer: OpenTelemetry tracer instance
        """
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        """
        Wrap WorkflowTask.__call__ with TASK span tracing for workflow activities.

        Creates spans with appropriate OpenInference span kinds (LLM, TOOL, or TASK)
        based on the task type and restores OpenTelemetry context using workflow
        instance ID for proper trace hierarchy.

        Args:
            wrapped (callable): Original WorkflowTask.__call__ method to be instrumented
            instance (WorkflowTask): WorkflowTask instance containing task function and metadata
            args (tuple): Positional arguments - typically (ctx: WorkflowActivityContext, payload: Any)
            kwargs (dict): Keyword arguments passed to the original method

        Returns:
            Any: Result from wrapped method execution, with span attributes capturing
                 input/output and task categorization
        """
        # Check for instrumentation suppression
        if context_api and context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return wrapped(*args, **kwargs)

        # Extract WorkflowActivityContext and payload
        ctx = args[0] if args else None
        payload = args[1] if len(args) > 1 else kwargs.get("payload")

        # Determine task details
        task_name = (
            getattr(instance.func, "__name__", "unknown_task")
            if instance.func
            else "workflow_task"
        )
        span_kind = self._determine_span_kind(instance, task_name)

        # Build span attributes
        attributes = self._build_task_attributes(
            instance, ctx, payload, span_kind, task_name
        )

        # Create span name
        span_name = f"WorkflowTask.{task_name}"

        # Handle async vs sync execution like other wrappers
        if asyncio.iscoroutinefunction(wrapped):
            return self._handle_async_execution(
                wrapped, args, kwargs, span_name, attributes
            )
        else:
            return self._handle_sync_execution(
                wrapped, args, kwargs, span_name, attributes
            )

    def _determine_span_kind(self, instance: Any, task_name: str) -> str:
        """
        Determine appropriate OpenInference span kind based on task characteristics.

        Analyzes the WorkflowTask instance and function name to select the most
        appropriate OpenInference span kind for observability categorization.

        Args:
            instance (WorkflowTask): WorkflowTask instance with task function and attributes
            task_name (str): Name of the task function being executed

        Returns:
            str: OpenInference span kind - 'LLM' for language model tasks,
                 'TOOL' for tool execution tasks, or 'TASK' for other workflow activities
        """
        # Check if this is an LLM-based task
        if (
            hasattr(instance, "llm") and instance.llm is not None
        ) or "generate_response" in task_name:
            return LLM

        # Check if this is a tool execution task
        if (
            hasattr(instance, "agent") and instance.agent is not None
        ) or "run_tool" in task_name:
            return TOOL

        # For workflow orchestration tasks, use custom TASK span kind
        # This provides semantic clarity for workflow-specific operations
        return TASK

    def _build_task_attributes(
        self,
        instance: Any,
        ctx: Any,
        payload: Any,
        span_kind: str,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build comprehensive span attributes for workflow task execution tracing.

        Constructs detailed attributes including workflow context, task metadata,
        executor information, and correlation identifiers for Phoenix UI grouping
        and trace hierarchy establishment.

        Args:
            instance (WorkflowTask): WorkflowTask instance containing task function and metadata
            ctx (WorkflowActivityContext): Dapr workflow activity context with instance information
            payload (Any): Task input payload to be serialized as span input
            span_kind (str): OpenInference span kind (LLM, TOOL, or TASK)
            task_name (str, optional): Name of the task function, extracted if not provided

        Returns:
            Dict[str, Any]: Comprehensive span attributes including workflow.instance_id,
                           task metadata, executor information, and serialized input payload
        """
        attributes = {
            OPENINFERENCE_SPAN_KIND: span_kind,
            INPUT_MIME_TYPE: "application/json",
            OUTPUT_MIME_TYPE: "application/json",
        }

        # Add workflow context information
        if ctx:
            # Extract instance ID using the same method as instrumentor.py
            instance_id = "unknown"
            try:
                inner_ctx = ctx.get_inner_context()
                instance_id = getattr(inner_ctx, "workflow_id", "unknown")
            except Exception as e:
                logger.debug(f"Failed to extract instance_id from context: {e}")

            attributes["workflow.instance_id"] = instance_id

            # Add explicit grouping attributes for better trace correlation
            attributes["session.id"] = instance_id  # Phoenix UI session grouping
            attributes["trace.group.id"] = instance_id  # Custom trace grouping
            attributes[
                "workflow.correlation_id"
            ] = instance_id  # Additional correlation

            # Add resource-level attributes for Phoenix UI grouping
            attributes["resource.workflow.instance_id"] = instance_id
            attributes[
                "resource.workflow.name"
            ] = "ToolCallingWorkflow"  # Could be dynamic

            # Log the trace context for debugging (expected to be disconnected for Dapr Workflows)
            from opentelemetry import trace

            current_span = trace.get_current_span()
            if current_span:
                current_trace_id = current_span.get_span_context().trace_id
                logger.debug(
                    f"⚡️ Task in instance {instance_id} starting with trace_id=0x{current_trace_id:x} (will be restored via W3C context)"
                )

            logger.debug(f"🔗 Grouping span by instance_id: {instance_id}")

        # Add task type information
        if hasattr(instance, "func") and instance.func:
            task_name = task_name or getattr(instance.func, "__name__", "unknown")
            attributes["task.name"] = task_name
            attributes["task.module"] = getattr(instance.func, "__module__", "unknown")

        # Add custom workflow task identification
        if span_kind == TASK and task_name:
            attributes["workflow.task.type"] = "orchestration"
            attributes["workflow.task.category"] = self._categorize_workflow_task(
                task_name
            )

        # Add executor type
        if hasattr(instance, "llm") and instance.llm:
            attributes["task.executor"] = "llm"
            attributes["llm.client_type"] = type(instance.llm).__name__
        elif hasattr(instance, "agent") and instance.agent:
            attributes["task.executor"] = "agent"
            attributes["agent.name"] = getattr(instance.agent, "name", "unknown")
        elif hasattr(instance, "func") and instance.func:
            attributes["task.executor"] = "python"

        # Serialize input payload
        if payload is not None:
            attributes[INPUT_VALUE] = safe_json_dumps(payload)

        return attributes

    def _handle_async_execution(
        self, wrapped: Any, args: Any, kwargs: Any, span_name: str, attributes: dict
    ) -> Any:
        """
        Handle asynchronous workflow task execution with OpenTelemetry context restoration.

        Manages async task execution by restoring stored OpenTelemetry context using
        the workflow instance ID, creating child spans with proper trace hierarchy,
        and capturing task results with comprehensive error handling.

        Args:
            wrapped (callable): Original async WorkflowTask method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (e.g., "WorkflowTask.generate_response")
            attributes (dict): Span attributes including workflow context and task metadata

        Returns:
            Any: Coroutine that executes the wrapped method with proper span instrumentation
                 and context restoration from stored W3C trace context
        """

        async def async_wrapper():
            # Get OpenTelemetry context from storage using instance_id
            otel_context = None
            instance_id = attributes.get("workflow.instance_id", "unknown")

            if instance_id != "unknown":
                from ..context_storage import get_workflow_context

                otel_context = get_workflow_context(instance_id)

            # Create span with restored context if available
            from ..context_propagation import create_child_span_with_context

            with create_child_span_with_context(
                self._tracer, span_name, otel_context, attributes
            ) as span:
                # Debug logging to show context restoration
                from opentelemetry import trace

                current_span = trace.get_current_span()
                task_category = attributes.get("workflow.task.category", "UNKNOWN")

                if otel_context:
                    logger.debug(
                        f"🔗 Creating {task_category} span with RESTORED context: {span_name}"
                    )
                    logger.debug(
                        f"👨‍👦 Restored from traceparent: {otel_context.get('traceparent', 'None')}"
                    )
                else:
                    logger.debug(
                        f"⚠️ Creating {task_category} span WITHOUT context restoration: {span_name}"
                    )

                logger.debug(
                    f"🎯 Current span context: {current_span.get_span_context() if current_span else 'None'}"
                )

                try:
                    result = await wrapped(*args, **kwargs)
                    span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result))
                    span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    logger.error(
                        f"Error in async workflow task execution: {e}", exc_info=True
                    )
                    raise

        return async_wrapper()

    def _handle_sync_execution(
        self, wrapped: Any, args: Any, kwargs: Any, span_name: str, attributes: dict
    ) -> Any:
        """
        Handle synchronous workflow task execution with OpenTelemetry context restoration.

        Manages sync task execution by restoring stored OpenTelemetry context using
        the workflow instance ID, creating child spans with proper trace hierarchy,
        and capturing task results with comprehensive error handling.

        Args:
            wrapped (callable): Original sync WorkflowTask method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            span_name (str): Name for the created span (e.g., "WorkflowTask.run_tool")
            attributes (dict): Span attributes including workflow context and task metadata

        Returns:
            Any: Result from wrapped method execution with span instrumentation
                 and context restoration from stored W3C trace context
        """
        # Get OpenTelemetry context from storage using instance_id
        otel_context = None
        instance_id = attributes.get("workflow.instance_id", "unknown")

        if instance_id != "unknown":
            from ..context_storage import get_workflow_context

            otel_context = get_workflow_context(instance_id)

        # Create span with restored context if available
        from ..context_propagation import create_child_span_with_context

        with create_child_span_with_context(
            self._tracer, span_name, otel_context, attributes
        ) as span:
            # Debug logging to show context restoration
            from opentelemetry import trace

            current_span = trace.get_current_span()
            task_category = attributes.get("workflow.task.category", "UNKNOWN")

            if otel_context:
                logger.debug(
                    f"🔗 Creating {task_category} span with RESTORED context: {span_name}"
                )
                logger.debug(
                    f"👨‍👦 Restored from traceparent: {otel_context.get('traceparent', 'None')}"
                )
            else:
                logger.debug(
                    f"⚠️ Creating {task_category} span WITHOUT context restoration: {span_name}"
                )

            logger.debug(
                f"🎯 Current span context: {current_span.get_span_context() if current_span else 'None'}"
            )

            try:
                result = wrapped(*args, **kwargs)
                span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result))
                span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(
                    f"Error in sync workflow task execution: {e}", exc_info=True
                )
                raise

    def _categorize_workflow_task(self, task_name: str) -> str:
        """
        Categorize workflow tasks for enhanced observability and trace organization.

        Analyzes task function names to assign semantic categories that help
        organize traces in observability tools and provide better insights
        into workflow execution patterns.

        Task categories:
        - initialization: Task setup and workflow entry recording
        - state_management: Message appending and state updates
        - finalization: Workflow completion and cleanup
        - communication: Agent messaging and response handling
        - llm_generation: Language model response generation
        - tool_execution: Tool calling and external system integration
        - orchestration: General workflow coordination activities

        Args:
            task_name (str): Name of the workflow task function

        Returns:
            str: Semantic category for the task type
        """
        if task_name in ["record_initial_entry", "get_workflow_entry_info"]:
            return "initialization"
        elif task_name in ["append_assistant_message", "append_tool_message"]:
            return "state_management"
        elif task_name in ["finalize_workflow", "finish_workflow"]:
            return "finalization"
        elif task_name in ["broadcast_message_to_agents", "send_response_back"]:
            return "communication"
        elif task_name == "generate_response":
            return "llm_generation"
        elif task_name == "run_tool":
            return "tool_execution"
        else:
            return "orchestration"
