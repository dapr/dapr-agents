from typing import Any

# ============================================================================
# Library Availability Detection and Imports
# ============================================================================

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry import _logs as logs_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.types import AttributeValue

from wrapt import wrap_function_wrapper

from openinference.instrumentation import safe_json_dumps as oi_safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
# ============================================================================
# OpenInference Semantic Conventions with Fallback Constants
# ============================================================================

# Primary constants from OpenInference
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS

# Message attributes for proper input/output structure
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS

# LLM-specific attributes for chat completions
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL

# Span kinds as values
AGENT = OpenInferenceSpanKindValues.AGENT.value
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
LLM = OpenInferenceSpanKindValues.LLM.value

# Custom span kind for workflow tasks
TASK = "TASK"  # Custom span kind for workflow orchestration tasks

# Tool call attributes
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
WORKFLOW_RUN_SUPPRESSION_KEY = "dapr_agents.workflow.skip_run_span"

# ============================================================================
# Utility Functions with OpenInference Integration
# ============================================================================


def safe_json_dumps(obj: Any) -> str:
    """
    Safely serialize object to JSON string with error handling.

    Provides a unified JSON serialization interface that uses OpenInference's
    implementation when available, or falls back to a basic JSON serialization
    with string conversion for non-serializable objects.

    Args:
        obj (Any): Object to serialize - can be dict, list, string, or any serializable type

    Returns:
        str: JSON string representation of the object with proper error handling
    """
    return oi_safe_json_dumps(obj)


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    # OpenTelemetry core
    "trace_api",
    "logs_api",
    "context_api",
    "BaseInstrumentor",
    "Status",
    "StatusCode",
    "AttributeValue",
    "wrap_function_wrapper",
    # Span attributes
    "OPENINFERENCE_SPAN_KIND",
    "INPUT_VALUE",
    "OUTPUT_VALUE",
    "INPUT_MIME_TYPE",
    "OUTPUT_MIME_TYPE",
    "TOOL_NAME",
    "TOOL_DESCRIPTION",
    "TOOL_PARAMETERS",
    # Message attributes
    "LLM_INPUT_MESSAGES",
    "LLM_OUTPUT_MESSAGES",
    "MESSAGE_ROLE",
    "MESSAGE_CONTENT",
    "MESSAGE_TOOL_CALLS",
    # LLM attributes
    "LLM_MODEL_NAME",
    "LLM_TOKEN_COUNT_PROMPT",
    "LLM_TOKEN_COUNT_COMPLETION",
    "LLM_TOKEN_COUNT_TOTAL",
    # Span kinds
    "AGENT",
    "CHAIN",
    "TOOL",
    "LLM",
    "TASK",
    # Tool call attributes
    "TOOL_CALL_FUNCTION_NAME",
    "TOOL_CALL_FUNCTION_ARGUMENTS_JSON",
    "WORKFLOW_RUN_SUPPRESSION_KEY",
    # Helper functions
    "safe_json_dumps",
]
