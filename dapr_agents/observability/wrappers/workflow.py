import functools
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
# Note: We pass instance id as part of the inputs to the workflow so that the workflow tasks can find the proper parent AGENT span context


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
            logger.debug("Instrumentation suppressed, skipping span creation")
            bound_method = wrapped.__get__(instance, type(instance))
            return bound_method(*args, **kwargs)

        # Extract arguments
        arguments = bind_arguments(wrapped, *args, **kwargs)
        workflow = arguments.get("workflow")

        # Extract workflow name
        if isinstance(workflow, str):
            workflow_name = workflow
        else:
            # Try to get the name from the workflow function/class
            workflow_name = None
            # First try the workflow decorator name
            if hasattr(workflow, "_workflow_name"):
                workflow_name = workflow._workflow_name
            # Then try __name__ for function workflows
            if not workflow_name and hasattr(workflow, "__name__"):
                workflow_name = workflow.__name__
            # Finally try the name attribute
            if not workflow_name and hasattr(workflow, "name"):
                workflow_name = workflow.name
            # Fallback
            if not workflow_name:
                workflow_name = "AgenticWorkflow"
        logger.debug(f"Extracted workflow_name: {workflow_name}")

        # Build span attributes
        attributes = self._build_workflow_attributes(instance, workflow_name, arguments)

        # Create AGENT span (this IS the agent execution for workflow-based agents)
        agent_name = getattr(instance, "name", instance.__class__.__name__)
        span_name = f"{agent_name}.{workflow_name}"
        logger.debug(f"Creating AGENT span: {span_name}")

        with self._tracer.start_as_current_span(
            span_name, attributes=attributes
        ) as span:
            logger.debug(f"Span context: {span.get_span_context()}")

            try:
                # Execute the workflow start
                bound_method = wrapped.__get__(instance, type(instance))
                instance_id = bound_method(*args, **kwargs)

                if instance_id:
                    span.set_attribute("workflow.instance_id", instance_id)
                    logger.debug(f"Added workflow.instance_id attribute: {instance_id}")

                # Store span context for workflow tasks to find
                try:
                    from ..context_propagation import extract_otel_context
                    from ..context_storage import store_workflow_context

                    # Extract current context from the AGENT span
                    current_context = extract_otel_context()
                    if current_context.get("traceparent"):
                        # Store the span ID for workflow tasks to use as parent
                        span_id = span.get_span_context().span_id
                        trace_id = span.get_span_context().trace_id
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

                        # Store ONLY instance-specific context to prevent cross-instance contamination upon app restarts
                        # that creates new workflow instances
                        if instance_id:
                            context_key = f"__workflow_context_{instance_id}__"
                            store_workflow_context(context_key, span_context)
                            logger.info(
                                f"Stored Agent span context for instance {instance_id} with key '{context_key}': trace_id={format(trace_id, '032x')}, span_id={format(span_id, '016x')}"
                            )
                        else:
                            logger.warning(
                                "No instance_id available, cannot store instance-specific context"
                            )
                    else:
                        logger.warning("No traceparent found in AGENT span context")
                except Exception as e:
                    logger.warning(f"Failed to store span context: {e}")

                # Set output attributes
                span.set_attribute(
                    OUTPUT_VALUE, safe_json_dumps({"instance_id": instance_id})
                )

                span.set_status(Status(StatusCode.OK))
                return instance_id

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"AGENT span failed: {span_name} - {e}")
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
        logger.debug(
            f"WorkflowMonitorWrapper.__call__ triggered with wrapped={wrapped.__name__}, instance={instance.__class__.__name__}"
        )
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
            bound_method = wrapped.__get__(instance, type(instance))
            return bound_method(*args, **kwargs)

        # Always create AGENT span for workflow execution
        from opentelemetry import trace

        logger.debug(
            f"WorkflowMonitorWrapper called - creating AGENT span for args: {args}, kwargs: {kwargs}"
        )

        workflow_name = self._extract_workflow_name(args, kwargs)
        # Extract agent name from the instance
        agent_name = getattr(instance, "name", "DurableAgent")
        logger.debug(f"Agent name: {agent_name}, Workflow name: {workflow_name}")

        attributes = self._build_workflow_attributes(
            workflow_name, agent_name, args, kwargs
        )

        # Create Agent span immediately - this will establish the trace context
        # The Agent span will be the parent of all workflow tasks
        logger.debug("Creating AGENT span for workflow execution")

        # Handle async vs sync execution
        if asyncio.iscoroutinefunction(wrapped):
            return self._handle_async_execution(
                wrapped, instance, args, kwargs, attributes, workflow_name, agent_name
            )
        else:
            return self._handle_sync_execution(
                wrapped, instance, args, kwargs, attributes, workflow_name, agent_name
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
        # Debug logging
        logger.debug(f"_extract_workflow_name: args={args}, kwargs={kwargs}")

        if args and len(args) > 0:
            workflow = args[0]
        else:
            workflow = kwargs.get("workflow")

        # Extract workflow name with better fallback chain
        if isinstance(workflow, str):
            workflow_name = workflow
        else:
            # Try to get the name from the workflow function/class
            workflow_name = None
            # First try the workflow decorator name
            if hasattr(workflow, "_workflow_name"):
                workflow_name = workflow._workflow_name
            # Then try __name__ for function workflows
            if not workflow_name and hasattr(workflow, "__name__"):
                workflow_name = workflow.__name__
            # Finally try the name attribute
            if not workflow_name and hasattr(workflow, "name"):
                workflow_name = workflow.name
            # Fallback
            if not workflow_name:
                workflow_name = "AgenticWorkflow"
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

    def _store_context_in_workflow_state(
        self, instance_id: str, context: Dict[str, Any]
    ) -> None:
        """
        Store OpenTelemetry context in workflow state for persistence across app restarts.

        This method stores the context in the DurableAgent's workflow state so it persists
        across app restarts and can be retrieved by workflow tasks when they resume.

        Args:
            instance_id: Workflow instance ID
            context: OpenTelemetry context data
        """
        try:
            # Import here to avoid circular imports
            from dapr_agents.agents.durableagent.state import DurableAgentWorkflowEntry

            # This is a simplified approach - in practice, you'd need to access
            # the actual workflow state through the Dapr Workflow runtime
            # For now, we'll store it in the in-memory storage as a fallback
            # but with a key that indicates it should be persisted
            from ..context_storage import store_workflow_context

            # Store with a special prefix to indicate it should be persisted
            persistent_key = f"__persistent_context_{instance_id}__"
            store_workflow_context(persistent_key, context)

            logger.debug(f"Stored context for persistence: {persistent_key}")

        except Exception as e:
            logger.warning(f"Failed to store context in workflow state: {e}")

    def _handle_async_execution(
        self,
        wrapped: Any,
        instance: Any,
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

        async def async_wrapper(*wrapper_args, **wrapper_kwargs):
            span_name = f"{agent_name}.{workflow_name}"

            # Note: This wrapper creates AGENT spans for NEW workflows only.
            # Resumed workflows get their trace context restored during startup, not here.

            # Debug logging to confirm span creation
            logger.debug(f"Creating AGENT span: {span_name}")

            with self._tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                # Store context immediately when AGENT span is created
                # This ensures workflow tasks can find it during execution
                try:
                    from ..context_propagation import extract_otel_context
                    from ..context_storage import store_workflow_context

                    # Extract current context from the AGENT span
                    current_context = extract_otel_context()
                    if current_context.get("traceparent"):
                        # Store the span ID for workflow tasks to use as parent
                        span_id = span.get_span_context().span_id
                        trace_id = span.get_span_context().trace_id
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

                        # Pass span context through workflow input instead of temporary storage
                        # This eliminates the need for temporary keys and prevents conflicts
                        logger.debug(
                            f"AGENT span created with trace_id={format(trace_id, '032x')}, span_id={format(span_id, '016x')}"
                        )
                        logger.debug(
                            "Will pass span context through workflow input to avoid temporary key conflicts"
                        )
                    else:
                        logger.warning("No traceparent found in AGENT span context")
                        span_context = None
                except Exception as e:
                    logger.warning(f"Failed to extract span context: {e}")
                    span_context = None

                try:
                    # Inject span context into workflow input
                    modified_kwargs = dict(kwargs)
                    if (
                        span_context
                        and "input" in modified_kwargs
                        and isinstance(modified_kwargs["input"], dict)
                    ):
                        # Add span context to workflow input
                        modified_kwargs["input"]["_otel_span_context"] = span_context
                        logger.debug("Injected span context into workflow input")

                    # Execute workflow and get result
                    bound_method = wrapped.__get__(instance, type(instance))
                    result = await bound_method(*args, **modified_kwargs)

                    # Note: Context storage is handled by the workflow itself when it starts
                    # See DurableAgent.tool_calling_workflow line ~216 where it stores the context
                    # This ensures context is available DURING workflow execution, not after

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

        return async_wrapper(*args, **kwargs)

    def _handle_sync_execution(
        self,
        wrapped: Any,
        instance: Any,
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
        span_name = f"{agent_name}.{workflow_name}"
        logger.debug(f"Creating AGENT span: {span_name}")

        with self._tracer.start_as_current_span(
            span_name, attributes=attributes
        ) as span:
            # Store context immediately when AGENT span is created
            # This ensures workflow tasks can find it during execution
            try:
                from ..context_propagation import extract_otel_context
                from ..context_storage import store_workflow_context

                # Extract current context from the AGENT span
                current_context = extract_otel_context()
                if current_context.get("traceparent"):
                    # Store the span ID for workflow tasks to use as parent
                    span_id = span.get_span_context().span_id
                    trace_id = span.get_span_context().trace_id
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

                    # Store immediately with temporary key based on workflow input hash
                    # This allows WorkflowTask spans to find it during execution
                    import hashlib

                    input_hash = hashlib.md5(
                        str(args + tuple(kwargs.items())).encode()
                    ).hexdigest()[:8]
                    temp_key = f"__workflow_context_temp_{input_hash}_{format(span_id, '016x')}__"
                    store_workflow_context(temp_key, span_context)
                    logger.debug(
                        f"AGENT span created with trace_id={format(trace_id, '032x')}, span_id={format(span_id, '016x')}"
                    )
                else:
                    logger.warning("No traceparent found in AGENT span context")
                    span_context = None
            except Exception as e:
                logger.warning(f"Failed to extract span context: {e}")
                span_context = None

            try:
                # We need to intercept the workflow execution to capture the instance ID
                # The sync method calls run_and_monitor_workflow_async which calls run_workflow
                bound_method = wrapped.__get__(instance, type(instance))

                # Store context before execution so workflow tasks can find it
                if span_context:
                    # For sync execution, we need to hook into the run_workflow call
                    # to get the instance ID. Use object.__setattr__ to bypass Pydantic validation.
                    original_run_workflow = instance.run_workflow

                    def patched_run_workflow(*run_args, **run_kwargs):
                        instance_id = original_run_workflow(*run_args, **run_kwargs)
                        # Store the context with the instance ID
                        store_workflow_context(
                            f"__workflow_context_{instance_id}__", span_context
                        )
                        logger.debug(
                            f"Stored workflow context for instance {instance_id}"
                        )
                        return instance_id

                    # Temporarily replace the method using object.__setattr__ to bypass Pydantic
                    object.__setattr__(instance, "run_workflow", patched_run_workflow)
                    try:
                        result = bound_method(*args, **kwargs)
                    finally:
                        # Restore the original method
                        object.__setattr__(
                            instance, "run_workflow", original_run_workflow
                        )
                else:
                    result = bound_method(*args, **kwargs)

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
# Workflow Registration Wrapper
# ============================================================================


class WorkflowRegistrationWrapper:
    """
    Wrapper for WorkflowApp._register_workflows method.

    This wrapper instruments the workflow registration process to add AGENT span
    tracing for orchestrator workflows when they are registered with the Dapr runtime.
    This ensures that when workflows are executed directly by the Dapr runtime
    (bypassing our monitored methods), they still get proper AGENT span tracing.
    """

    def __init__(self, tracer: Any) -> None:
        """
        Initialize the workflow registration wrapper.

        Args:
            tracer: OpenTelemetry tracer instance
        """
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        """
        Wrap WorkflowApp._register_workflows to add AGENT span tracing to workflows.

        Args:
            wrapped: Original _register_workflows method
            instance: WorkflowApp instance
            args: Positional arguments (wfs: Dict[str, Callable])
            kwargs: Keyword arguments

        Returns:
            None (original method returns None)
        """
        logger.info(
            f"WorkflowRegistrationWrapper called for {instance.__class__.__name__}"
        )

        # Check for instrumentation suppression
        if context_api and context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ):
            bound_method = wrapped.__get__(instance, type(instance))
            return bound_method(*args, **kwargs)

        # Get the workflows dictionary
        wfs = args[0] if args else {}
        logger.info(f"Found {len(wfs)} workflows to register: {list(wfs.keys())}")

        # Wrap each workflow method to add AGENT span tracing
        for wf_name, method in wfs.items():
            # Only create AGENT spans for orchestrator workflows, not individual agent workflows
            # Check if this is an orchestrator workflow by looking at the instance type
            is_orchestrator = (
                hasattr(instance, "orchestrator_topic_name")
                or "Orchestrator" in instance.__class__.__name__
            )

            if is_orchestrator:
                logger.info(
                    f"Wrapping orchestrator workflow '{wf_name}' with AGENT span tracing"
                )

                # Create a wrapper that adds AGENT span tracing for orchestrator workflows
                def create_agent_span_wrapper(original_method, workflow_name):
                    if not hasattr(create_agent_span_wrapper, "_spans_created"):
                        create_agent_span_wrapper._spans_created = set()

                    @functools.wraps(original_method)
                    def wrapped_workflow(*workflow_args, **workflow_kwargs):
                        instance_id = "unknown"
                        if workflow_args and len(workflow_args) > 0:
                            ctx = workflow_args[0]
                            if hasattr(ctx, "instance_id"):
                                instance_id = ctx.instance_id
                            elif hasattr(ctx, "get_inner_context"):
                                try:
                                    inner_ctx = ctx.get_inner_context()
                                    instance_id = getattr(
                                        inner_ctx, "workflow_id", "unknown"
                                    )
                                except Exception as e:
                                    logger.debug(
                                        f"Failed to extract instance_id from context: {e}"
                                    )

                        # Create a unique key for this instance
                        span_key = f"{instance_id}_{workflow_name}"

                        # Only create AGENT span for the first execution of this workflow instance
                        if span_key not in create_agent_span_wrapper._spans_created:
                            create_agent_span_wrapper._spans_created.add(span_key)

                            # Create AGENT span for orchestrator workflow
                            agent_name = getattr(
                                instance, "name", instance.__class__.__name__
                            )
                            span_name = f"{agent_name}.{workflow_name}"

                            logger.info(
                                f"Creating AGENT span: {span_name} for instance {instance_id}"
                            )

                            # Build basic attributes
                            attributes = {
                                "openinference.span.kind": "AGENT",
                                "workflow.name": workflow_name,
                                "agent.execution_mode": "workflow_based",
                                "agent.name": agent_name,
                                "workflow.orchestrator": True,
                                "workflow.instance_id": instance_id,
                            }

                            # Add input data if available
                            if workflow_args and len(workflow_args) > 1:
                                input_data = workflow_args[
                                    1
                                ]  # Second argument is usually the input
                                if input_data:
                                    attributes["input.value"] = str(input_data)
                                    attributes["input.mime_type"] = "application/json"

                            with self._tracer.start_as_current_span(
                                span_name, attributes=attributes
                            ) as span:
                                try:
                                    # Store span context for workflow tasks to find
                                    try:
                                        from ..context_propagation import (
                                            extract_otel_context,
                                        )
                                        from ..context_storage import (
                                            store_workflow_context,
                                        )

                                        # Extract current context from the AGENT span
                                        current_context = extract_otel_context()
                                        if current_context.get("traceparent"):
                                            # Store the span ID for workflow tasks to use as parent
                                            span_id = span.get_span_context().span_id
                                            trace_id = span.get_span_context().trace_id
                                            span_context = {
                                                "trace_id": format(
                                                    trace_id, "032x"
                                                ),  # Convert to 32-char hex string
                                                "span_id": format(
                                                    span_id, "016x"
                                                ),  # Convert to 16-char hex string
                                                "traceparent": current_context.get(
                                                    "traceparent"
                                                ),
                                                "tracestate": current_context.get(
                                                    "tracestate", ""
                                                ),
                                            }

                                            # Store with instance-specific key
                                            context_key = (
                                                f"__workflow_context_{instance_id}__"
                                            )
                                            store_workflow_context(
                                                context_key, span_context
                                            )
                                            logger.info(
                                                f"Stored Agent span context with key '{context_key}': trace_id={format(trace_id, '032x')}, span_id={format(span_id, '016x')}"
                                            )
                                        else:
                                            logger.warning(
                                                "No traceparent found in AGENT span context"
                                            )
                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to store span context: {e}"
                                        )

                                    result = original_method(
                                        *workflow_args, **workflow_kwargs
                                    )
                                    span.set_attribute(
                                        "output.value", str(result) if result else ""
                                    )
                                    span.set_attribute(
                                        "output.mime_type", "application/json"
                                    )
                                    return result
                                except Exception as e:
                                    span.record_exception(e)
                                    raise
                        else:
                            # For subsequent executions, just run the original method without creating a new span
                            logger.debug(
                                f"Skipping AGENT span creation for subsequent execution of {workflow_name} (instance {instance_id})"
                            )
                            return original_method(*workflow_args, **workflow_kwargs)

                    return wrapped_workflow

                # Replace the method in the workflows dictionary
                wfs[wf_name] = create_agent_span_wrapper(method, wf_name)
            else:
                # For non-orchestrator workflows, just pass through without AGENT span tracing
                # They will still get traced by WorkflowTaskWrapper if they have tasks
                pass

        # Call the original method with the wrapped workflows
        bound_method = wrapped.__get__(instance, type(instance))
        return bound_method(wfs, **kwargs)


# ============================================================================
# Exported Classes
# ============================================================================

__all__ = [
    "WorkflowRunWrapper",
    "WorkflowMonitorWrapper",
    "WorkflowRegistrationWrapper",
]
