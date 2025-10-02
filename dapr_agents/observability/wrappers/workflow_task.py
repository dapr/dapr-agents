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
            bound_method = wrapped.__get__(instance, type(instance))
            return bound_method(*args, **kwargs)

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
                wrapped, instance, args, kwargs, span_name, attributes
            )
        else:
            return self._handle_sync_execution(
                wrapped, instance, args, kwargs, span_name, attributes
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

        # Check if this is a tool execution task (including agent tasks)
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
            attributes["resource.workflow.name"] = "AgenticWorkflow"  # Could be dynamic

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

    def _get_context_from_workflow_state(
        self, instance_id: str, instance: Any = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve OpenTelemetry context from workflow state for persistence across app restarts.

        This method retrieves the context from the DurableAgent's workflow state so it can
        be used to create child spans with proper parent-child relationships even after
        app restarts when the workflow resumes.

        Args:
            instance_id: Workflow instance ID
            instance: DurableAgent instance to access state from

        Returns:
            OpenTelemetry context data if found, None otherwise
        """
        try:
            if (
                instance
                and hasattr(instance, "load_state")
                and hasattr(instance, "state")
            ):
                # Load state to get the latest data from database
                instance.load_state()

                # Try to get the workflow instance data to check for stored trace context
                if "instances" in instance.state:
                    instance_data = instance.state["instances"].get(instance_id, {})
                    stored_trace_context = instance_data.get("trace_context")
                    if stored_trace_context:
                        logger.debug(
                            f"Retrieved trace context from database for instance {instance_id}"
                        )
                        return stored_trace_context
                    else:
                        logger.debug(
                            f"No trace context found in database for {instance_id}"
                        )
                        return None
                else:
                    logger.debug(f"No instances found in state for {instance_id}")
                    return None
            else:
                logger.debug(f"Cannot access instance state for {instance_id}")
                return None

        except Exception as e:
            logger.warning(f"Failed to get context from workflow state: {e}")
            return None

    # TODO: in future this needs to be updated to properly capture all of the AGENT span context for resumed workflows.
    def _create_context_for_resumed_workflow(
        self, instance_id: str, instance: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Create AGENT span context for resumed workflows to restore proper trace hierarchy.

        This creates a safe, non-blocking AGENT span that establishes the trace context
        for child workflow tasks to nest under, fixing the missing parent span issue.
        """
        try:
            from ..context_storage import get_workflow_context, store_workflow_context
            from opentelemetry import trace

            # Check if this is a resumed workflow by looking at the workflow state
            if not (hasattr(instance, "state") and "instances" in instance.state):
                logger.debug(
                    f"No state found for instance to check if {instance_id} is resumed"
                )
                return None
            if hasattr(instance, "load_state"):
                instance.load_state()

            instance_data = instance.state["instances"].get(instance_id, {})
            stored_trace_context = instance_data.get("trace_context", {})

            if not stored_trace_context:
                logger.debug(
                    f"No stored trace context found for {instance_id} - not a resumed workflow"
                )
                return None

            # Check if this is an orchestrator workflow or individual agent workflow
            agent_name = getattr(instance, "name", "DurableAgent")
            is_orchestrator = (
                hasattr(instance, "orchestrator_topic_name")
                or "Orchestrator" in instance.__class__.__name__
            )

            # For individual agent workflows, we need to check if they're being triggered by an orchestrator
            # If they have a triggering_workflow_instance_id, they should be nested under the orchestrator
            # If they don't, they should create their own AGENT span (standalone usage)
            if not is_orchestrator:
                # Check if this agent workflow was triggered by an orchestrator
                triggering_workflow_id = instance_data.get(
                    "triggering_workflow_instance_id"
                )
                if triggering_workflow_id:
                    logger.debug(
                        f"Skipping AGENT span creation for individual agent workflow {agent_name}.{instance_id} - triggered by orchestrator {triggering_workflow_id}, should be nested"
                    )
                    return None
                else:
                    logger.debug(
                        f"Creating AGENT span for standalone individual agent workflow {agent_name}.{instance_id}"
                    )

            if is_orchestrator:
                logger.debug(
                    f"Creating AGENT span for resumed orchestrator workflow {instance_id}"
                )
                workflow_name = instance_data.get(
                    "workflow_name", "OrchestratorWorkflow"
                )
                is_orchestrator_flag = True
            else:
                logger.debug(
                    f"Creating AGENT span for resumed individual agent workflow {instance_id}"
                )
                workflow_name = instance_data.get("workflow_name", "AgenticWorkflow")
                is_orchestrator_flag = False

            workflow_name = instance_data.get("workflow_name", "AgenticWorkflow")
            span_name = f"{agent_name}.{workflow_name}"
            attributes = {
                "openinference.span.kind": "AGENT",
                "workflow.name": workflow_name,
                "agent.execution_mode": "workflow_based",
                "agent.name": agent_name,
                "workflow.instance_id": instance_id,
                "workflow.resumed": True,
                "workflow.orchestrator": is_orchestrator_flag,
                "input.value": instance_data.get("input", ""),
                "input.mime_type": "application/json",
                "output.value": instance_data.get("output", ""),
                "output.mime_type": "application/json",
            }

            # Check if we've already created an agent span for this workflow to avoid duplicates
            if instance_data.get("agent_span_created"):
                logger.debug(
                    f"Agent span already created for {instance_id}, reusing context"
                )
                existing_context = get_workflow_context(
                    f"__workflow_context_{instance_id}__"
                )
                if existing_context:
                    return existing_context

            # Create and start the AGENT span - this will be the parent for all workflow tasks
            agent_span = self._tracer.start_span(span_name, attributes=attributes)

            try:
                span_id = agent_span.get_span_context().span_id
                trace_id = agent_span.get_span_context().trace_id
                span_context = {
                    "trace_id": format(trace_id, "032x"),
                    "span_id": format(span_id, "016x"),
                    "traceparent": f"00-{format(trace_id, '032x')}-{format(span_id, '016x')}-01",
                    "tracestate": stored_trace_context.get("tracestate", ""),
                    "instance_id": instance_id,
                    "resumed": True,
                    "agent_span": agent_span,  # Keep reference to close later
                }

                # Store context for subsequent workflow tasks to find
                store_workflow_context(
                    f"__workflow_context_{instance_id}__", span_context
                )

                logger.debug(
                    f"Created AGENT span for resumed workflow {instance_id}: {span_name}"
                )

                instance_data["agent_span_created"] = True
                if hasattr(instance, "save_state"):
                    instance.save_state()

                return span_context

            except Exception as e:
                # If something goes wrong, end the span
                agent_span.end()
                raise e

        except Exception as e:
            logger.error(
                f"Failed to create AGENT span for resumed workflow {instance_id}: {e}"
            )
            return None

    def _handle_async_execution(
        self,
        wrapped: Any,
        instance: Any,
        args: Any,
        kwargs: Any,
        span_name: str,
        attributes: dict,
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

        async def async_wrapper(instance, *wrapper_args, **wrapper_kwargs):
            logger.debug(f"WorkflowTaskWrapper called for {span_name}")

            # Get OpenTelemetry context from storage using instance_id
            instance_id = attributes.get("workflow.instance_id", "unknown")

            # Also try to extract instance_id directly from workflow context if available
            if len(wrapper_args) > 0:
                ctx = wrapper_args[
                    0
                ]  # First argument is usually the WorkflowActivityContext
                try:
                    if hasattr(ctx, "get_inner_context"):
                        inner_ctx = ctx.get_inner_context()
                        direct_instance_id = getattr(inner_ctx, "workflow_id", None)
                        if direct_instance_id and direct_instance_id != "unknown":
                            instance_id = direct_instance_id
                            logger.debug(
                                f"Extracted instance_id from context: {instance_id}"
                            )
                except Exception as e:
                    logger.warning(f"Failed to extract instance_id from context: {e}")

            # Try to get stored Agent span context for parent-child relationship
            from ..context_storage import get_workflow_context
            from opentelemetry import trace
            from opentelemetry.trace import SpanContext, TraceFlags

            # Get stored Agent span context - ONLY use instance-specific context
            logger.debug(
                f"WorkflowTask looking for Agent span context for instance: {instance_id}"
            )
            agent_context = None

            # Get instance-specific context directly (created by WorkflowMonitorWrapper or startup process)
            if instance_id and instance_id != "unknown":
                context_key = f"__workflow_context_{instance_id}__"
                agent_context = get_workflow_context(context_key)
                if agent_context:
                    logger.info(
                        f"Found instance-specific context for {instance_id} with key '{context_key}'"
                    )
                else:
                    logger.warning(
                        f"No instance-specific context found for {instance_id} with key '{context_key}'"
                    )
                    # Try to get all contexts for debugging
                    from ..context_storage import get_all_workflow_contexts

                    all_contexts = get_all_workflow_contexts()
                    logger.warning(
                        f"Available context keys: {list(all_contexts.keys())}"
                    )
                    agent_context = self._create_context_for_resumed_workflow(
                        instance_id, instance
                    )
            else:
                logger.warning(
                    f"No valid instance_id ({instance_id}), cannot lookup context"
                )
                agent_context = None

            if agent_context:
                logger.debug(f"Found Agent span context: {agent_context}")
                # Check if this is a restored trace context
                if hasattr(instance, "state") and "instances" in instance.state:
                    instance_data = instance.state["instances"].get(instance_id, {})
                    if instance_data.get("trace_context") == agent_context:
                        logger.debug(
                            f"Using restored trace context for resumed workflow {instance_id}"
                        )
            else:
                logger.warning("No Agent span context found")

            if (
                agent_context
                and agent_context.get("trace_id")
                and agent_context.get("span_id")
            ):
                # Create parent span context from stored data
                try:
                    # Convert hex strings to integers (OpenTelemetry expects int, not bytes)
                    trace_id = int(agent_context["trace_id"], 16)
                    parent_span_id = int(agent_context["span_id"], 16)

                    # Create SpanContext for the parent
                    parent_span_context = SpanContext(
                        trace_id=trace_id,
                        span_id=parent_span_id,
                        trace_flags=TraceFlags(
                            0x01
                        ),  # Use TraceFlags constructor with sampled flag
                        is_remote=True,
                    )

                    # Create child span with explicit parent - use simpler approach
                    logger.debug(
                        f"Creating {span_name} as child of Agent span: {agent_context['span_id']}"
                    )
                    parent_context = trace.set_span_in_context(
                        trace.NonRecordingSpan(parent_span_context)
                    )
                    with self._tracer.start_as_current_span(
                        span_name, attributes=attributes, context=parent_context
                    ) as span:
                        logger.debug(f"Started child span {span_name}")
                        try:
                            bound_method = wrapped.__get__(instance, type(instance))
                            result = await bound_method(*wrapper_args, **wrapper_kwargs)
                            span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result))
                            span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
                            span.set_status(Status(StatusCode.OK))
                            logger.debug(f"Completed child span {span_name}")
                            return result
                        except Exception as e:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(e)
                            logger.error(
                                f"Error in async workflow task execution: {e}",
                                exc_info=True,
                            )
                            logger.error(f"Failed child span {span_name}: {e}")
                            raise
                except Exception as e:
                    logger.warning(
                        f"Failed to create child span with parent context: {e}"
                    )
                    logger.warning(
                        f"Parent span creation failed: {e}, executing without span"
                    )
                    # Fall through to no-span execution

            # No parent context available - execute without creating orphaned spans
            logger.warning(
                f"No parent context available for {span_name}, executing without span"
            )
            try:
                bound_method = wrapped.__get__(instance, type(instance))
                result = await bound_method(*wrapper_args, **wrapper_kwargs)
                return result
            except Exception as e:
                logger.error(
                    f"Error in async workflow task execution (no span): {e}",
                    exc_info=True,
                )
                raise

        return async_wrapper(instance, *args, **kwargs)

    def _handle_sync_execution(
        self,
        wrapped: Any,
        instance: Any,
        args: Any,
        kwargs: Any,
        span_name: str,
        attributes: dict,
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
        # Always try to get context from current workflow first (since there's only one workflow)
        from ..context_storage import get_workflow_context

        # Extract instance ID first for instance-specific context lookup
        instance_id = attributes.get("workflow.instance_id", "unknown")

        # Try instance-specific context first (for individual agent workflows)
        otel_context = None
        if instance_id != "unknown":
            context_key = f"__workflow_context_{instance_id}__"
            otel_context = get_workflow_context(context_key)
            if otel_context:
                logger.info(
                    f"🔗 Found instance-specific context for {instance_id} with key '{context_key}'"
                )
            else:
                logger.warning(
                    f"⚠️ No instance-specific context found for {instance_id} with key '{context_key}'"
                )
                # Try to get all contexts for debugging
                from ..context_storage import get_all_workflow_contexts

                all_contexts = get_all_workflow_contexts()
                logger.warning(f"Available context keys: {list(all_contexts.keys())}")

        # If still not found and we have an instance ID, try stored trace context from database
        if not otel_context and instance_id != "unknown":
            # Try to get context from workflow state first (persists across restarts)
            otel_context = self._get_context_from_workflow_state(instance_id, instance)

            # If not found in workflow state, try in-memory storage as fallback
            if not otel_context:
                otel_context = get_workflow_context(instance_id)

        # Create span with restored context if available
        from ..context_propagation import create_child_span_with_context

        with create_child_span_with_context(
            self._tracer, span_name, otel_context, attributes
        ) as span:
            return self._execute_task_with_span(
                wrapped, instance, args, kwargs, span, otel_context, span_name
            )

    def _execute_task_with_span(
        self, wrapped, instance, args, kwargs, span, otel_context, span_name
    ):
        """Execute the task within the provided span context."""
        # Debug logging to show context restoration
        from opentelemetry import trace

        current_span = trace.get_current_span()
        task_category = span.attributes.get("workflow.task.category", "UNKNOWN")

        if otel_context:
            logger.debug(
                f"Creating {task_category} span with RESTORED context: {span_name}"
            )
            logger.debug(
                f"Restored from traceparent: {otel_context.get('traceparent', 'None')}"
            )
        else:
            logger.debug(
                f"Creating {task_category} span WITHOUT context restoration: {span_name}"
            )

        logger.debug(
            f"Current span context: {current_span.get_span_context() if current_span else 'None'}"
        )

        try:
            bound_method = wrapped.__get__(instance, type(instance))
            result = bound_method(*args, **kwargs)
            span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result))
            span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            logger.error(f"Error in sync workflow task execution: {e}", exc_info=True)
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
        if task_name in ["record_initial_entry"]:
            return "initialization"
        elif task_name in ["finalize_workflow", "finish_workflow"]:
            return "finalization"
        elif task_name in ["broadcast_message_to_agents", "send_response_back"]:
            return "communication"
        elif task_name == "call_llm":
            return "llm_generation"
        elif task_name == "run_tool":
            return "tool_execution"
        else:
            return "orchestration"
