"""
Thread-safe context storage for Dapr Workflow OpenTelemetry context propagation.

This module provides a global storage mechanism for OpenTelemetry contexts
that can cross Dapr Workflow boundaries without modifying function signatures.

The key challenge with Dapr Workflows is that OpenTelemetry context doesn't
naturally propagate across workflow task boundaries due to the Dapr runtime's
serialization/deserialization process. This storage provides a non-invasive
solution by storing W3C Trace Context data that can be retrieved by workflow
tasks using their instance ID.

Architecture:
- Store W3C context during workflow task creation (in instrumentor.py)
- Retrieve context during workflow task execution (in workflow_task.py)
- Use thread-safe storage to handle concurrent workflow executions
- Automatic cleanup to prevent memory leaks from completed workflows
"""

import logging
import threading
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class WorkflowContextStorage:
    """
    Thread-safe storage for workflow OpenTelemetry contexts with W3C Trace Context support.

    This class provides a centralized storage mechanism for OpenTelemetry contexts
    that need to be propagated across Dapr Workflow execution boundaries. It uses
    workflow instance IDs as keys to store and retrieve W3C Trace Context data.

    Key features:
    - Thread-safe operations using RLock for concurrent workflow execution
    - Instance ID-based storage for precise context retrieval
    - W3C Trace Context format support (traceparent/tracestate)
    - Memory management with cleanup capabilities
    - Debug statistics for monitoring storage usage

    Usage pattern:
    1. Store context during workflow task creation (monkey-patched in instrumentor)
    2. Retrieve context during workflow task execution (in workflow_task wrapper)
    3. Clean up context when workflow completes to prevent memory leaks
    """

    def __init__(self):
        """
        Initialize the workflow context storage.

        Creates thread-safe storage using RLock to handle concurrent access
        from multiple workflow instances executing simultaneously.
        """
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def store_context(self, instance_id: str, otel_context: Dict[str, Any]) -> None:
        """
        Store OpenTelemetry context for a workflow instance using W3C Trace Context format.

        Stores the W3C Trace Context data (traceparent/tracestate) that was extracted
        during workflow task creation, allowing it to be retrieved later during
        workflow task execution to maintain distributed trace continuity.

        Args:
            instance_id (str): Unique workflow instance ID from WorkflowActivityContext
            otel_context (Dict[str, Any]): W3C context data from extract_otel_context()
                                          containing traceparent, tracestate, and debug info
        """
        with self._lock:
            self._storage[instance_id] = otel_context
            logger.debug(f"🔗 Stored context for instance {instance_id}")

    def get_context(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve OpenTelemetry context for a workflow instance using W3C Trace Context format.

        Retrieves the stored W3C Trace Context data that can be used to restore
        OpenTelemetry context and create child spans with proper parent-child
        relationships in distributed traces.

        Args:
            instance_id (str): Unique workflow instance ID from WorkflowActivityContext

        Returns:
            Optional[Dict[str, Any]]: W3C context data with traceparent/tracestate headers,
                                     or None if no context found for the instance
        """
        with self._lock:
            context = self._storage.get(instance_id)
            if context:
                logger.debug(f"🔗 Retrieved context for instance {instance_id}")
            else:
                logger.warning(f"⚠️ No context found for instance {instance_id}")
            return context

    def create_resumed_workflow_context(
        self,
        instance_id: str,
        agent_name: Optional[str] = None,
        stored_trace_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new trace context for a resumed workflow after app restart.

        When an app restarts, the in-memory context storage is lost. This method
        creates a new trace context for resumed workflows so they can still be
        traced, even though they won't be connected to the original trace.

        Args:
            instance_id (str): Unique workflow instance ID

        Returns:
            Dict[str, Any]: New W3C context data for the resumed workflow
        """
        try:
            from opentelemetry import trace
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            # Create a new trace for the resumed workflow with proper AGENT span
            tracer = trace.get_tracer(__name__)

            # Create AGENT span with proper agent name for resumed workflow
            agent_display_name = agent_name or "DurableAgent"
            span_name = f"{agent_display_name}.AgenticWorkflow"
            with tracer.start_as_current_span(span_name) as span:
                # Set AGENT span attributes
                from .constants import OPENINFERENCE_SPAN_KIND

                span.set_attribute(OPENINFERENCE_SPAN_KIND, "AGENT")
                span.set_attribute("workflow.instance_id", instance_id)
                span.set_attribute("workflow.resumed", True)
                span.set_attribute("agent.name", agent_name)
                # Extract the new context
                propagator = TraceContextTextMapPropagator()
                carrier = {}
                propagator.inject(carrier)

                context_data = {
                    "traceparent": carrier.get("traceparent"),
                    "tracestate": carrier.get("tracestate"),
                    "instance_id": instance_id,
                    "resumed": True,
                    "debug_info": f"New trace created for resumed workflow {instance_id}",
                }

                # Store the new context
                self.store_context(instance_id, context_data)
                logger.info(
                    f"Created new trace context for resumed workflow {instance_id}"
                )

                return context_data

        except Exception as e:
            logger.error(
                f"Failed to create resumed workflow context for {instance_id}: {e}"
            )
            return {
                "traceparent": None,
                "tracestate": None,
                "instance_id": instance_id,
                "resumed": True,
                "error": str(e),
            }

    def cleanup_context(self, instance_id: str) -> None:
        """
        Clean up stored context for a completed workflow instance to prevent memory leaks.

        Removes the stored W3C context data when a workflow completes to prevent
        the storage from growing indefinitely. Should be called when workflow
        execution finishes or fails.

        Args:
            instance_id (str): Unique workflow instance ID to clean up
        """
        with self._lock:
            if instance_id in self._storage:
                del self._storage[instance_id]
                logger.debug(f"🧹 Cleaned up context for instance {instance_id}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored contexts for debugging and monitoring.

        Provides visibility into the storage state for troubleshooting context
        propagation issues and monitoring memory usage from stored contexts.

        Returns:
            Dict[str, Any]: Storage statistics including:
                          - stored_instances: Number of currently stored contexts
                          - instance_ids: List of workflow instance IDs with stored contexts
        """
        with self._lock:
            return {
                "stored_instances": len(self._storage),
                "instance_ids": list(self._storage.keys()),
            }


# Global instance for workflow context storage across the application
_context_storage = WorkflowContextStorage()


def store_workflow_context(instance_id: str, otel_context: Dict[str, Any]) -> None:
    """
    Store OpenTelemetry context for a workflow instance using the global storage.

    Convenience function that provides a simple interface to store W3C Trace Context
    data for workflow instances. Used during workflow task creation to preserve
    context across Dapr Workflow runtime boundaries.

    Args:
        instance_id (str): Unique workflow instance ID from WorkflowActivityContext.workflow_id
        otel_context (Dict[str, Any]): W3C context data from extract_otel_context()
                                      containing traceparent, tracestate, and debug components
    """
    _context_storage.store_context(instance_id, otel_context)


def get_workflow_context(instance_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve OpenTelemetry context for a workflow instance using the global storage.

    Convenience function that provides a simple interface to retrieve stored
    W3C Trace Context data for workflow instances. Used during workflow task
    execution to restore context and maintain distributed trace continuity.

    Args:
        instance_id (str): Unique workflow instance ID from WorkflowActivityContext.workflow_id

    Returns:
        Optional[Dict[str, Any]]: W3C context data with traceparent/tracestate headers
                                 for creating child spans, or None if not found
    """
    return _context_storage.get_context(instance_id)


def get_all_workflow_contexts() -> Dict[str, Dict[str, Any]]:
    """
    Retrieve all stored OpenTelemetry contexts from the global storage.

    Used for debugging and fallback context lookup when instance-specific
    context retrieval fails due to timing issues.

    Returns:
        Dict[str, Dict[str, Any]]: All stored contexts keyed by instance_id/key
    """
    with _context_storage._lock:
        return dict(_context_storage._storage)


def cleanup_workflow_context(instance_id: str) -> None:
    """
    Clean up stored context for a completed workflow instance using the global storage.

    Convenience function that provides a simple interface to clean up stored
    context data when workflows complete. Important for preventing memory
    leaks in long-running applications with many workflow executions.

    Args:
        instance_id (str): Unique workflow instance ID to clean up from storage
    """
    _context_storage.cleanup_context(instance_id)


def get_context_storage_stats() -> Dict[str, Any]:
    """
    Get statistics about stored contexts for debugging and monitoring using the global storage.

    Convenience function that provides visibility into the storage state for
    troubleshooting context propagation issues and monitoring memory usage.
    Useful for debugging workflow context issues and ensuring proper cleanup.

    Returns:
        Dict[str, Any]: Storage statistics including count of stored instances
                       and list of active workflow instance IDs
    """
    return _context_storage.get_storage_stats()
