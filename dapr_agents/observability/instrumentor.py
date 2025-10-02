"""
OpenTelemetry instrumentor for Dapr Agents with comprehensive tracing and Phoenix UI compatibility.

This instrumentor provides complete observability for Dapr Agents by creating proper
hierarchical span structures that follow OpenInference semantic conventions. It supports
all agent types with W3C Trace Context propagation across Dapr Workflow boundaries.

Span Hierarchy by Agent Type:

**Regular Agent (Direct Execution):**
- Agent.run (AGENT span) - Root span for agent execution
  └── Agent.conversation (CHAIN span) - Processing and reasoning logic
      ├── Agent.execute_tools (TOOL span) - Batch tool coordination
      │   └── AgentToolExecutor.run_tool (TOOL span) - Individual tool execution
      └── ChatClient.generate (LLM span) - Language model interactions

**DurableAgent (Workflow-based with Monitoring):**
- WorkflowApp.run_and_monitor_workflow_async (AGENT span) - DurableAgent with lifecycle monitoring
  └── DurableAgent.tool_calling_workflow (CHAIN span) - Workflow orchestration logic
      ├── WorkflowTask.__call__ (TOOL span) - Tool execution via Dapr Workflow activities
      └── WorkflowTask.__call__ (LLM span) - LLM calls via Dapr Workflow activities

**Orchestrators (Workflow-based, Fire-and-Forget):**
- WorkflowApp.run_workflow (AGENT span) - Orchestrator execution without monitoring
  └── Orchestrator.main_workflow (CHAIN span) - Multi-agent coordination workflow
      ├── WorkflowTask.__call__ (TASK span) - Workflow orchestration activities
      └── WorkflowTask.__call__ (LLM span) - LLM calls via Dapr Workflow activities

Key Features:
- OpenInference semantic conventions for Phoenix UI visualization
- W3C Trace Context propagation across Dapr Workflow runtime boundaries
- Tool call capture and display with proper Phoenix UI formatting
- Message format conversion to OpenInference Message standards
- Token usage tracking and comprehensive metadata extraction
- Thread-safe context storage for workflow task execution
- Automatic LLM provider discovery and instrumentation
- Comprehensive error handling with graceful degradation

W3C Context Propagation:
The instrumentor implements a sophisticated context propagation mechanism that solves
the challenge of maintaining distributed traces across Dapr Workflow boundaries using
W3C Trace Context format (traceparent/tracestate headers) and thread-safe storage.
"""

import logging
from typing import Any, Collection

from .constants import (
    OPENTELEMETRY_AVAILABLE,
    WRAPT_AVAILABLE,
    BaseInstrumentor,
    trace_api,
    context_api,
    wrap_function_wrapper,
)
from .wrappers import (
    AgentRunWrapper,
    ExecuteToolsWrapper,
    LLMWrapper,
    ProcessIterationsWrapper,
    RunToolWrapper,
    WorkflowMonitorWrapper,
    WorkflowRegistrationWrapper,
    WorkflowRunWrapper,
    WorkflowTaskWrapper,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Main Instrumentor Class
# ============================================================================


class DaprAgentsInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentor for comprehensive Dapr Agents observability.

    This instrumentor provides complete tracing coverage for all Dapr Agent types
    by implementing OpenInference semantic conventions and W3C Trace Context
    propagation across Dapr Workflow runtime boundaries.

    Architecture:
    - Regular Agents: Direct method instrumentation with hierarchical spans
    - DurableAgents: Workflow-based execution with context propagation via W3C format
    - Orchestrators: Multi-agent coordination with proper span categorization

    Context Propagation Strategy:
    - Extract W3C Trace Context (traceparent/tracestate) during workflow task creation
    - Store context using workflow instance IDs in thread-safe storage
    - Restore context during workflow task execution for proper span hierarchy
    - Maintain distributed trace continuity across Dapr runtime serialization

    Phoenix UI Integration:
    - OpenInference Message format for proper conversation display
    - Tool call extraction and formatting for UI visualization
    - Token usage tracking and metadata capture for cost analysis
    - Proper span kinds (AGENT, CHAIN, TOOL, LLM, TASK) for categorization

    Usage:
        >>> instrumentor = DaprAgentsInstrumentor()
        >>> instrumentor.instrument()
        >>> # Your Dapr Agents code runs with full observability
        >>> instrumentor.uninstrument()
    """

    # Class-level tracer for global access
    _global_tracer = None

    def __init__(self) -> None:
        """
        Initialize the Dapr Agents instrumentor.

        Sets up the instrumentor with no active tracer until instrument() is called.
        """
        super().__init__()
        self._tracer = None

    def instrumentation_dependencies(self) -> Collection[str]:
        """
        Get the list of dependencies required for instrumentation.

        Returns:
            Collection[str]: Package names that must be available for instrumentation
        """
        return ("dapr-agents",)

    def instrument(self, **kwargs: Any) -> None:
        """
        Public method to instrument Dapr Agents with OpenTelemetry tracing.

        This method is called by users to enable instrumentation. It delegates
        to the private _instrument method which contains the actual implementation.

        Args:
            **kwargs: Instrumentation configuration including:
                     - tracer_provider: Optional OpenTelemetry tracer provider
                     - Additional provider-specific configuration
        """
        self._instrument(**kwargs)

    def _instrument(self, **kwargs: Any) -> None:
        """
        Instrument Dapr Agents with comprehensive OpenTelemetry tracing.

        Applies instrumentation in the correct order:
        1. Dependency validation (OpenTelemetry, wrapt)
        2. Tracer initialization with provider configuration
        3. W3C context propagation setup for Dapr Workflows
        4. Method wrapper application for all agent types

        Args:
            **kwargs: Instrumentation configuration including:
                     - tracer_provider: Optional OpenTelemetry tracer provider
                     - Additional provider-specific configuration
        """
        # Validate required dependencies for instrumentation
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available - instrumentation disabled")
            return

        if not WRAPT_AVAILABLE:
            logger.warning("wrapt not available - instrumentation disabled")
            return

        # Initialize OpenTelemetry tracer with provider configuration
        self._initialize_tracer(kwargs)

        # Apply W3C context propagation fix for Dapr Workflows (critical for proper tracing)
        self._apply_context_propagation_fix()

        # Apply agent wrappers (Regular Agent class execution paths)
        self._apply_agent_wrappers()

        # Apply workflow wrappers directly since start_runtime() is now called in model_post_init
        # This eliminates the need for callback mechanism
        self._apply_workflow_wrappers()

        # LLM provider integrations
        self._apply_llm_wrappers()

        logger.info("✅ Dapr Agents OpenTelemetry instrumentation enabled")

    def _initialize_tracer(self, kwargs: dict) -> None:
        """
        Initialize OpenTelemetry tracer with provider configuration.

        Sets up the tracer that will be used by all wrapper instances
        to create spans with proper trace relationships.

        Args:
            kwargs (dict): Configuration including optional tracer_provider
        """
        tracer_provider = kwargs.get("tracer_provider")
        if not tracer_provider:
            tracer_provider = trace_api.get_tracer_provider()

        self._tracer = trace_api.get_tracer(__name__, tracer_provider=tracer_provider)
        # Store globally for access by resumed workflows
        DaprAgentsInstrumentor._global_tracer = self._tracer

    def _apply_context_propagation_fix(self) -> None:
        """
        Apply W3C Trace Context propagation fix for Dapr Workflow task execution.

        Solves the critical issue where OpenTelemetry context is lost across Dapr
        Workflow runtime boundaries due to serialization/deserialization and new
        event loop creation. Implements monkey-patching of WorkflowApp._make_task_wrapper
        to inject W3C context storage and retrieval mechanism.

        Context Propagation Flow:
        1. Extract W3C context during workflow task wrapper creation
        2. Store context using workflow instance ID in thread-safe storage
        3. Retrieve and restore context during workflow task execution
        4. Maintain proper parent-child span relationships across boundaries
        """
        logger.debug("Applying W3C context propagation fix for Dapr Workflows...")

        try:
            # Import required Dapr Workflow components
            from dapr_agents.workflow.base import WorkflowApp
            import asyncio
            import functools
            from dapr.ext.workflow import WorkflowActivityContext

            logger.debug(
                "Successfully imported Dapr Workflow modules for context propagation"
            )

            def run_sync_with_context(coro):
                """
                Execute coroutine synchronously while preserving OpenTelemetry context.

                Handles event loop management and context attachment for workflow tasks
                that may execute in new event loops created by the Dapr runtime.
                """
                # Capture current context BEFORE any event loop operations
                current_context = None
                current_span = None

                if context_api:
                    current_context = context_api.get_current()
                    current_span = trace_api.get_current_span()
                    logger.debug(
                        f"Captured context for async execution: {current_span.get_span_context() if current_span else 'No span'}"
                    )

                async def context_wrapped_coro():
                    """Run coroutine with preserved context."""
                    if current_context:
                        # Attach the captured context
                        token = context_api.attach(current_context)
                        try:
                            return await coro
                        finally:
                            context_api.detach(token)
                    else:
                        return await coro

                try:
                    # Try to use existing event loop
                    loop = asyncio.get_running_loop()
                    return loop.run_until_complete(context_wrapped_coro())
                except RuntimeError:
                    # No running loop - create new one
                    # TODO: eventually clean this up by using the tracing setup from dapr upstream
                    # when we have trace propagation in the SDKs for workflows.
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        return loop.run_until_complete(context_wrapped_coro())
                    except Exception as e:
                        logger.warning(
                            f"Failed to run coroutine with new event loop: {e}"
                        )
                        # Fallback: run in thread pool to avoid blocking
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                lambda: asyncio.run(context_wrapped_coro())
                            )
                            return future.result()
                    finally:
                        try:
                            if "loop" in locals() and not loop.is_closed():
                                loop.close()
                        except Exception as e:
                            logger.debug(f"Error closing event loop: {e}")

            def make_context_aware_task_wrapper(
                self, task_name: str, method, task_instance
            ):
                """
                Create enhanced task wrapper with W3C context propagation via storage.

                This factory method creates workflow task wrappers that implement W3C Trace
                Context propagation across Dapr Workflow runtime boundaries. The wrapper:

                1. Extracts workflow instance ID from Dapr context structure
                2. Retrieves stored W3C context using the instance ID
                3. Associates global context with specific workflow instances
                4. Executes tasks with proper OpenTelemetry context restoration

                Args:
                    task_name (str): Name of the workflow task being wrapped
                    method: Task method reference for wrapping
                    task_instance: Original task function or method instance

                Returns:
                    Callable: Enhanced wrapper function with W3C context propagation

                Technical Implementation:
                    - Uses WorkflowActivityContext for Dapr integration
                    - Implements context_storage for thread-safe context management
                    - Provides comprehensive error handling and debug logging
                    - Maintains non-invasive approach (no parameter modification)
                """
                logger.debug(f"Creating context-aware wrapper for task: {task_name}")

                @functools.wraps(method)
                def wrapper(ctx, *args, **kwargs):
                    logger.debug(f"Executing context-aware workflow task: {task_name}")

                    # Create workflow activity context wrapper
                    wf_ctx = WorkflowActivityContext(ctx)

                    # Extract workflow instance ID from Dapr context structure
                    instance_id = None
                    try:
                        inner_ctx = wf_ctx.get_inner_context()
                        instance_id = getattr(inner_ctx, "workflow_id", None)
                        logger.debug(f"Extracted workflow instance_id: {instance_id}")
                    except Exception as e:
                        logger.warning(f"Failed to extract workflow instance_id: {e}")

                    # Implement W3C context storage and association
                    from .context_storage import (
                        store_workflow_context,
                        get_workflow_context,
                    )

                    if instance_id:
                        # Try to get instance-specific context first
                        instance_context = get_workflow_context(
                            f"__workflow_context_{instance_id}__"
                        )
                        if instance_context and instance_context.get("traceparent"):
                            logger.debug(
                                f"Found instance-specific context for {instance_id}"
                            )
                        else:
                            logger.debug(
                                f"No instance-specific context found for {instance_id}"
                            )
                    else:
                        logger.warning(
                            "No workflow instance_id - cannot access W3C context"
                        )

                    try:
                        # Execute original task function without parameter modification
                        call = task_instance(wf_ctx, *args, **kwargs)
                        if asyncio.iscoroutine(call):
                            result = run_sync_with_context(call)
                        else:
                            result = call

                        logger.debug(
                            f"Completed context-aware workflow task: {task_name}"
                        )
                        return result
                    except Exception as e:
                        logger.exception(
                            f"Context-aware workflow task '{task_name}' failed: {e}"
                        )
                        raise

                return wrapper

            # Apply enhanced context-aware task wrapper to Dapr Workflow runtime
            logger.debug(
                "Monkey patching WorkflowApp._make_task_wrapper for W3C context propagation"
            )
            original_method = getattr(WorkflowApp, "_make_task_wrapper", None)
            logger.debug(f"Original _make_task_wrapper method: {original_method}")

            WorkflowApp._make_task_wrapper = make_context_aware_task_wrapper
            logger.debug(
                "Applied W3C context propagation enhancement for Dapr Workflow tasks"
            )

        except ImportError as e:
            logger.warning(
                f"Could not apply W3C context propagation fix (ImportError): {e}"
            )
        except Exception as e:
            logger.error(
                f"Error applying W3C context propagation fix: {e}", exc_info=True
            )

    def _apply_agent_wrappers(self) -> None:
        """
        Apply observability wrappers for agent execution methods.

        Instruments core agent methods to create AGENT spans (top-level) and CHAIN spans
        (processing steps) for comprehensive agent execution tracing in Phoenix UI.
        """
        try:
            from dapr_agents.agents.agent.agent import Agent
            from dapr_agents.tool.executor import AgentToolExecutor

            # Main agent run wrapper (AGENT span - top level)
            # Note: This only instruments regular Agent.run, not DurableAgent.run
            # DurableAgent uses WorkflowMonitorWrapper instead
            wrap_function_wrapper(
                module="dapr_agents.agents.agent.agent",
                name="Agent.run",
                wrapper=AgentRunWrapper(self._tracer),
            )

            # Process iterations wrapper (CHAIN span - processing steps)
            wrap_function_wrapper(
                module="dapr_agents.agents.agent.agent",
                name="Agent.conversation",
                wrapper=ProcessIterationsWrapper(self._tracer),
            )

            # Tool execution batch wrapper (TOOL span - batch execution)
            wrap_function_wrapper(
                module="dapr_agents.agents.agent.agent",
                name="Agent.execute_tools",
                wrapper=ExecuteToolsWrapper(self._tracer),
            )

            # Individual tool execution wrapper (TOOL span - actual tool execution)
            wrap_function_wrapper(
                module="dapr_agents.tool.executor",
                name="AgentToolExecutor.run_tool",
                wrapper=RunToolWrapper(self._tracer),
            )

            # Note: DurableAgent.run is not instrumented here because it's instrumented in the _apply_workflow_wrappers method
            # and we don't want to double instrument it. And also, instrumenting DurableAgent.run here will cause issues with the async nature of the method.
            # So we instrument it in the _apply_workflow_wrappers method.

        except Exception as e:
            logger.error(f"Error applying agent wrappers: {e}", exc_info=True)

    def _apply_workflow_wrappers(self) -> None:
        """
        Apply observability wrappers for workflow orchestration methods.

        Instruments DurableAgent workflow execution to create comprehensive AGENT spans
        that capture the complete workflow lifecycle from start to completion, enabling
        detailed monitoring of workflow orchestration in Phoenix UI.
        """
        try:
            from dapr_agents.workflow.base import WorkflowApp
            from dapr_agents.workflow.task import WorkflowTask

            wrap_function_wrapper(
                module="dapr_agents.workflow.base",
                name="WorkflowApp.run_and_monitor_workflow_async",
                wrapper=WorkflowMonitorWrapper(self._tracer),
            )

            wrap_function_wrapper(
                module="dapr_agents.workflow.base",
                name="WorkflowApp.run_and_monitor_workflow_sync",
                wrapper=WorkflowMonitorWrapper(self._tracer),
            )

            # This is necessary to create the parent workflow span for the 09 quickstart...
            wrap_function_wrapper(
                module="dapr_agents.workflow.base",
                name="WorkflowApp.run_workflow",
                wrapper=WorkflowRunWrapper(self._tracer),
            )

            # Instrument workflow registration to add AGENT spans for orchestrator workflows
            wrap_function_wrapper(
                module="dapr_agents.workflow.base",
                name="WorkflowApp._register_workflows",
                wrapper=WorkflowRegistrationWrapper(self._tracer),
            )

            # Instrument workflow registration to add AGENT spans for orchestrator workflows
            wrap_function_wrapper(
                module="dapr_agents.workflow.base",
                name="WorkflowApp._register_workflows",
                wrapper=WorkflowRegistrationWrapper(self._tracer),
            )

            wrap_function_wrapper(
                module="dapr_agents.workflow.task",
                name="WorkflowTask.__call__",
                wrapper=WorkflowTaskWrapper(self._tracer),
            )
        except Exception as e:
            logger.error(f"Error applying workflow wrappers: {e}", exc_info=True)

    def _apply_llm_wrappers(self) -> None:
        """
        Apply observability wrappers for LLM chat completion methods.

        Automatically discovers and instruments all LLM chat client implementations
        that extend ChatClientBase, creating LLM spans with OpenInference formatting
        for comprehensive language model interaction tracking in Phoenix UI.
        """
        try:
            # Import base chat client class
            from dapr_agents.llm.chat import ChatClientBase

            # Discover all chat client implementations
            chat_client_classes = self._discover_chat_clients(ChatClientBase)

            # Instrument each concrete chat client implementation
            for chat_client_class, module_name in chat_client_classes:
                wrap_function_wrapper(
                    module=module_name,
                    name=f"{chat_client_class.__name__}.generate",
                    wrapper=LLMWrapper(self._tracer),
                )
                logger.debug(
                    f"Instrumented {chat_client_class.__name__} in {module_name}"
                )

        except ImportError as e:
            logger.warning(f"Could not instrument LLM chat clients: {e}")

    def _discover_chat_clients(self, base_class: type) -> list:
        """
        Discover all ChatClient subclasses across LLM provider modules.

        Args:
            base_class: Base ChatClientBase class

        Returns:
            list: List of (class, module_name) tuples
        """
        chat_client_classes = []

        # Check each LLM provider module for ChatClient classes
        for provider in ["openai", "nvidia", "huggingface", "dapr"]:
            try:
                module = __import__(f"dapr_agents.llm.{provider}.chat", fromlist=[""])

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, base_class)
                        and attr is not base_class
                    ):
                        chat_client_classes.append(
                            (attr, f"dapr_agents.llm.{provider}.chat")
                        )

            except ImportError:
                logger.debug(f"Could not import dapr_agents.llm.{provider}.chat")
                continue

        return chat_client_classes

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Uninstrument Dapr Agents by clearing the tracer.

        Args:
            **kwargs: Uninstrumentation configuration (unused)
        """
        self._tracer = None
        logger.info("Dapr Agents OpenTelemetry instrumentation disabled")


# ============================================================================
# Exported Classes
# ============================================================================

__all__ = [
    "DaprAgentsInstrumentor",
]
