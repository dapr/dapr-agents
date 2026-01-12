"""
OpenTelemetry instrumentor for Dapr Agents with comprehensive tracing and Phoenix UI compatibility.

This instrumentor provides complete observability for Dapr Agents by creating proper
hierarchical span structures that follow OpenInference semantic conventions. It supports
all agent types with W3C Trace Context propagation across Dapr Workflow boundaries.

Span Hierarchy by Agent Type:

**Regular Agent (Direct Execution):**
- Agent.run (AGENT span) - Root span for agent execution
  └── Agent._conversation_loop (CHAIN span) - Reasoning turns
      ├── Agent._execute_tool_calls (TOOL span) - Batch tool coordination
      │   └── AgentToolExecutor.run_tool (TOOL span) - Individual tool execution
      └── ChatClient.generate (LLM span) - Language model interactions

**DurableAgent (Workflow-based with Monitoring):**
- WorkflowRunner.run_workflow_async (AGENT span when wait=True) - DurableAgent lifecycle
  └── registered workflow entry (CHAIN span) - Orchestration logic
      ├── registered activities (TOOL span) - Tool execution via workflow activities
      └── registered activities (LLM span) - LLM calls via workflow activities

**Orchestrators (Workflow-based, Fire-and-Forget):**
- WorkflowRunner.run_workflow (AGENT span) - Orchestrator execution without monitoring
  └── registered workflow entry (CHAIN span) - Multi-agent coordination workflow
      ├── registered activities (TASK span) - Workflow orchestration activities
      └── registered activities (LLM span) - LLM calls via workflow activities

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
from typing import Any, Collection, Optional

from .constants import (
    BaseInstrumentor,
    trace_api,
    logs_api,
    wrap_function_wrapper,
)
from .wrappers import (
    AgentRunWrapper,
    ExecuteToolsWrapper,
    LLMWrapper,
    ProcessIterationsWrapper,
    RunToolWrapper,
    WorkflowMonitorWrapper,
    WorkflowRunWrapper,
    WorkflowActivityRegistrationWrapper,
)

from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
from opentelemetry.trace import TracerProvider

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
    _global_logger = None

    def __init__(self) -> None:
        """
        Initialize the Dapr Agents instrumentor.

        Sets up the instrumentor with no active tracer until instrument() is called.
        """
        super().__init__()
        self._tracer = None
        self._logger = None
        self._grpc_instrumentor: Optional[GrpcInstrumentorClient] = None

    def instrumentation_dependencies(self) -> Collection[str]:
        """
        Get the list of dependencies required for instrumentation.

        Returns:
            Collection[str]: Package names that must be available for instrumentation
        """
        return ()

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

        # Initialize logger for the instrumentor
        self._initialize_logger(kwargs)

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

        # Instrument gRPC client for context propagation to Dapr sidecar
        self._grpc_instrumentor = GrpcInstrumentorClient()
        self._grpc_instrumentor.instrument()

        logger.info("✅ Dapr Agents OpenTelemetry instrumentation enabled")

    def _initialize_logger(self, kwargs: dict[str, Any]) -> None:
        """
        Initialize the logger for the instrumentor.

        Sets up the logger to use the module-level logger defined at the top.
        """
        logger_provider = kwargs.get("logger_provider")
        if not logger_provider:
            logger_provider = logs_api.get_logger_provider()

        self._logger = logs_api.get_logger(__name__, logger_provider=logger_provider)
        DaprAgentsInstrumentor._global_logger = self._logger

    def _initialize_tracer(self, kwargs: dict[str, Any]) -> None:
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
        Ensure workflow activities run with restored W3C context by wrapping
        ``WorkflowRuntime.register_activity``.
        """
        try:
            wrap_function_wrapper(
                module="dapr.ext.workflow.workflow_runtime",
                name="WorkflowRuntime.register_activity",
                wrapper=WorkflowActivityRegistrationWrapper(self._tracer),
            )
            logger.debug(
                "Instrumented WorkflowRuntime.register_activity for context propagation."
            )
        except ImportError as exc:
            logger.warning(
                "Unable to import WorkflowRuntime for instrumentation: %s", exc
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Error applying workflow activity instrumentation: %s",
                exc,
                exc_info=True,
            )

    def _apply_agent_wrappers(self) -> None:
        """
        Apply observability wrappers for agent execution methods.

        Instruments core agent methods to create AGENT spans (top-level) and CHAIN spans
        (processing steps) for comprehensive agent execution tracing in Phoenix UI.
        """
        try:
            # Main agent run wrapper (AGENT span - top level)
            # Note: This only instruments regular Agent.run, not DurableAgent.run
            # DurableAgent uses WorkflowMonitorWrapper instead
            wrap_function_wrapper(
                module="dapr_agents.agents.standalone",
                name="Agent.run",
                wrapper=AgentRunWrapper(self._tracer),
            )

            # Process iterations wrapper (CHAIN span - processing steps)
            wrap_function_wrapper(
                module="dapr_agents.agents.standalone",
                name="Agent._conversation_loop",
                wrapper=ProcessIterationsWrapper(self._tracer),
            )

            # Tool execution batch wrapper (TOOL span - batch execution)
            wrap_function_wrapper(
                module="dapr_agents.agents.standalone",
                name="Agent._execute_tool_calls",
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
        Apply observability wrappers for workflow scheduling/monitoring APIs.
        """
        try:
            wrap_function_wrapper(
                module="dapr_agents.workflow.runners.base",
                name="WorkflowRunner.run_workflow_async",
                wrapper=WorkflowMonitorWrapper(self._tracer),
            )

            wrap_function_wrapper(
                module="dapr_agents.workflow.runners.base",
                name="WorkflowRunner.run_workflow",
                wrapper=WorkflowRunWrapper(self._tracer),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Error applying workflow wrappers: %s", e, exc_info=True)

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
        logger.debug("Uninstrumenting Dapr Agents OpenTelemetry instrumentation")

        try:
            tracer_provider: TracerProvider = trace_api.get_tracer_provider()
            if hasattr(tracer_provider, "force_flush"):
                tracer_provider.force_flush(timeout_millis=5000)  # type: ignore
                logger.debug("Flushed tracer provider spans")
        except Exception:  # noqa: BLE001
            logger.exception("Error while shutting down tracer provider", exc_info=True)

        self._grpc_instrumentor = None
        self._logger = None
        self._tracer = None
        logger.info("Dapr Agents OpenTelemetry instrumentation disabled")


# ============================================================================
# Exported Classes
# ============================================================================

__all__ = [
    "DaprAgentsInstrumentor",
]
