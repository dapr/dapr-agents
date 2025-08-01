import json
from typing import Any

# ============================================================================
# Library Availability Detection and Imports
# ============================================================================

try:
    from opentelemetry import context as context_api
    from opentelemetry import trace as trace_api
    from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.util.types import AttributeValue

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # OpenTelemetry is not available - instrumentation will be disabled
    # To enable observability features, install: pip install dapr-agents[observability]
    trace_api = None
    BaseInstrumentor = object
    context_api = None
    Status = None
    StatusCode = None
    AttributeValue = str  # Fallback type for mypy
    OPENTELEMETRY_AVAILABLE = False

try:
    from wrapt import wrap_function_wrapper

    WRAPT_AVAILABLE = True
except ImportError:
    # wrapt is not available - function wrapping will be disabled
    # To enable observability features, install: pip install dapr-agents[observability]
    wrap_function_wrapper = None
    WRAPT_AVAILABLE = False

try:
    from openinference.instrumentation import safe_json_dumps as oi_safe_json_dumps
    from openinference.semconv.trace import (
        MessageAttributes,
        OpenInferenceSpanKindValues,
        SpanAttributes,
        ToolCallAttributes,
    )

    OPENINFERENCE_AVAILABLE = True
except ImportError:
    # OpenInference is not available - enhanced Phoenix UI features will be disabled
    # To enable enhanced observability features, install: pip install dapr-agents[observability]
    oi_safe_json_dumps = None
    MessageAttributes = None
    OpenInferenceSpanKindValues = None
    SpanAttributes = None
    ToolCallAttributes = None
    OPENINFERENCE_AVAILABLE = False

# ============================================================================
# OpenInference Semantic Conventions with Fallback Constants
# ============================================================================

if OPENINFERENCE_AVAILABLE:
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
    TOOL_CALL_FUNCTION_ARGUMENTS_JSON = (
        ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
    )

else:
    # Fallback constants - direct string values following SmolAgents pattern
    OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
    INPUT_VALUE = "input.value"
    OUTPUT_VALUE = "output.value"
    INPUT_MIME_TYPE = "input.mime_type"
    OUTPUT_MIME_TYPE = "output.mime_type"
    TOOL_NAME = "tool.name"
    TOOL_DESCRIPTION = "tool.description"
    TOOL_PARAMETERS = "tool.parameters"

    # Message attributes for proper input/output structure
    LLM_INPUT_MESSAGES = "llm.input_messages"
    LLM_OUTPUT_MESSAGES = "llm.output_messages"
    MESSAGE_ROLE = "message.role"
    MESSAGE_CONTENT = "message.content"
    MESSAGE_TOOL_CALLS = "message.tool_calls"

    # LLM-specific attributes for chat completions
    LLM_MODEL_NAME = "llm.model_name"
    LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"

    # Span kinds
    AGENT = "AGENT"
    CHAIN = "CHAIN"
    TOOL = "TOOL"
    LLM = "LLM"

    # Custom span kind for workflow tasks
    TASK = "TASK"  # Custom span kind for workflow orchestration tasks

    # Tool call attributes
    TOOL_CALL_FUNCTION_NAME = "tool_call.function.name"
    TOOL_CALL_FUNCTION_ARGUMENTS_JSON = "tool_call.function.arguments_json"

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
    if OPENINFERENCE_AVAILABLE and oi_safe_json_dumps:
        return oi_safe_json_dumps(obj)
    else:
        try:
            return json.dumps(obj, default=str)
        except Exception:
            return str(obj)


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    # Availability flags
    "OPENTELEMETRY_AVAILABLE",
    "WRAPT_AVAILABLE",
    "OPENINFERENCE_AVAILABLE",
    # OpenTelemetry core
    "trace_api",
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
    # Helper functions
    "safe_json_dumps",
]
