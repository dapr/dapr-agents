"""
OpenTelemetry context propagation utilities for Dapr Workflow boundaries.

This module provides utilities for properly propagating OpenTelemetry context
across Dapr Workflow execution boundaries using W3C Trace Context format.

The key insight is that Dapr Workflow runtime can serialize/deserialize context
if we use the proper W3C format (traceparent/tracestate headers).

W3C Trace Context Format:
- traceparent: "00-{trace_id}-{span_id}-{flags}" (32-hex trace, 16-hex span, 2-hex flags)
- tracestate: Vendor-specific trace state information (optional)
- Enables cross-system distributed tracing with standard format
"""

import logging
from typing import Any, Dict, Optional

try:
    from opentelemetry import trace
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # OpenTelemetry is not available - context propagation will be disabled
    # To enable observability features, install: pip install dapr-agents[observability]
    trace = None
    TraceContextTextMapPropagator = None
    OPENTELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global W3C Trace Context propagator instance for consistent context handling
if OPENTELEMETRY_AVAILABLE:
    _propagator = TraceContextTextMapPropagator()
else:
    _propagator = None


def extract_otel_context() -> Dict[str, Any]:
    """
    Extract current OpenTelemetry context using W3C Trace Context format.

    Converts the current OpenTelemetry span context into W3C Trace Context format
    (traceparent/tracestate) that can be properly serialized and transmitted across
    Dapr Workflow boundaries. This ensures distributed trace continuity.

    W3C traceparent format: "version-trace_id-parent_id-trace_flags"
    - version: Always "00" (current W3C spec version)
    - trace_id: 32-character hex string (128-bit trace identifier)
    - parent_id: 16-character hex string (64-bit span identifier)
    - trace_flags: 2-character hex string ("01" if sampled, "00" if not)

    Returns:
        Dict[str, Any]: Serializable context data containing:
                       - traceparent: W3C trace context header
                       - tracestate: Vendor-specific trace state (optional)
                       - trace_id, span_id, trace_flags: Individual components for debugging

        If OpenTelemetry is not available, returns empty dict.
        Install with: pip install dapr-agents[observability]
    """
    if not OPENTELEMETRY_AVAILABLE:
        logger.debug("OpenTelemetry not available - cannot extract context")
        return {}

    if not _propagator or not trace:
        logger.debug(
            "OpenTelemetry components not initialized - cannot extract context"
        )
        return {}

    carrier: Dict[str, Any] = {}
    # Use W3C TraceContext propagator to inject standard headers
    _propagator.inject(carrier)

    span = trace.get_current_span()
    ctx = span.get_span_context()

    # Extract trace components in W3C format (always extract for consistency)
    trace_id = format(ctx.trace_id, "032x")  # 32-char hex (128-bit)
    span_id = format(ctx.span_id, "016x")  # 16-char hex (64-bit)
    flags = "01" if ctx.trace_flags.sampled else "00"  # W3C sampling flag

    # Ensure traceparent exists - create manually if propagator didn't inject
    # This handles edge cases where OpenTelemetry context might be incomplete
    if "traceparent" not in carrier and span and span.is_recording():
        carrier["traceparent"] = f"00-{trace_id}-{span_id}-{flags}"

    # Ensure tracestate exists in carrier (W3C spec allows empty tracestate)
    if "tracestate" not in carrier:
        carrier["tracestate"] = ""

    # Add individual components for debugging and verification
    carrier["trace_id"] = trace_id
    carrier["span_id"] = span_id
    carrier["trace_flags"] = flags

    logger.debug(
        f"🔗 Extracted W3C context: traceparent={carrier.get('traceparent', '')}"
    )
    return carrier


def restore_otel_context(otel_context: Optional[Dict[str, Any]]) -> Optional[object]:
    """
    Restore OpenTelemetry context from W3C Trace Context format.

    Converts W3C Trace Context headers (traceparent/tracestate) back into
    OpenTelemetry context objects that can be used as parent context for
    creating child spans, maintaining distributed trace continuity.

    This function is the inverse of extract_otel_context() and enables
    proper trace propagation across Dapr Workflow runtime boundaries.

    Args:
        otel_context (Optional[Dict[str, Any]]): Context data from extract_otel_context()
                                                containing W3C traceparent and tracestate

    Returns:
        Optional[object]: OpenTelemetry context object for creating child spans,
                         or None if restoration fails, context is invalid, or
                         OpenTelemetry is not available

    Note:
        If OpenTelemetry dependencies are not installed, this returns None.
        Install with: pip install dapr-agents[observability]
    """
    if not OPENTELEMETRY_AVAILABLE:
        logger.debug("OpenTelemetry not available - cannot restore context")
        return None

    if not _propagator or not trace:
        logger.debug(
            "OpenTelemetry components not initialized - cannot restore context"
        )
        return None

    if not otel_context:
        logger.debug("No context provided for restoration")
        return None

    try:
        # Reconstruct W3C carrier format from Dapr-serialized context
        # This converts our stored format back to the standard W3C headers
        carrier = {
            "traceparent": otel_context.get("traceparent", ""),
            "tracestate": otel_context.get("tracestate", ""),
        }

        # Use W3C TraceContext propagator to extract OpenTelemetry context
        ctx = _propagator.extract(carrier=carrier)
        logger.debug(
            f"🔗 Restored W3C context: traceparent={carrier.get('traceparent', '')}"
        )
        return ctx

    except Exception as e:
        logger.warning(
            f"⚠️ Failed to restore OpenTelemetry context from W3C format: {e}"
        )
        return None


def create_child_span_with_context(
    tracer: Any,
    span_name: str,
    otel_context: Optional[Dict[str, Any]],
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Create a child span with restored W3C Trace Context as parent.

    This function creates a new OpenTelemetry span that maintains the distributed
    trace hierarchy by using restored W3C Trace Context as the parent context.
    This ensures proper trace continuity across Dapr Workflow boundaries.

    The function handles both successful context restoration (creating a proper
    child span) and fallback scenarios (creating a root span) gracefully.

    Args:
        tracer (Any): OpenTelemetry tracer instance for creating spans
        span_name (str): Name for the new span (e.g., "WorkflowTask.generate_response")
        otel_context (Optional[Dict[str, Any]]): W3C context data from extract_otel_context()
                                                containing traceparent/tracestate headers
        attributes (Optional[Dict[str, Any]]): Optional span attributes to set on creation

    Returns:
        Span context manager that can be used in 'with' statements for proper
        span lifecycle management with restored parent-child relationships
    """
    # Try to restore context from W3C format first
    parent_ctx = restore_otel_context(otel_context)

    if parent_ctx:
        return tracer.start_as_current_span(
            span_name, context=parent_ctx, attributes=attributes
        )
    else:
        # Fallback: try to use current active span as parent
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            # Use current span as parent by creating a child span
            return tracer.start_as_current_span(span_name, attributes=attributes)
        else:
            # Last resort: create root span
            return tracer.start_as_current_span(span_name, attributes=attributes)
