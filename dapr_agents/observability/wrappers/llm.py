import asyncio
import logging
from typing import Any, Dict

from ..constants import (
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    LLM,
    LLM_MODEL_NAME,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    Status,
    StatusCode,
    context_api,
    safe_json_dumps,
)
from ..message_processors import (
    convert_messages_to_openinference,
    extract_token_usage,
    extract_tool_schemas,
    get_input_message_attributes,
    get_output_message_attributes,
    process_llm_response,
)
from ..utils import bind_arguments, serialize_tools_for_tracing, strip_method_args

try:
    from openinference.instrumentation import get_attributes_from_context
except ImportError:
    raise ImportError(
        "OpenInference not installed - please install with `pip install dapr-agents[observability]`"
    )

logger = logging.getLogger(__name__)

# ============================================================================
# LLM Interaction Wrapper
# ============================================================================


class LLMWrapper:
    """
    Wrapper for LLM chat completion calls with comprehensive message and tool tracing.

    This wrapper instruments LLM interactions to create LLM spans with proper
    OpenInference formatting for Phoenix UI compatibility. It captures complete
    conversation context, tool schemas, token usage, and response handling.

    Key features:
    - Input message processing with OpenInference Message format conversion
    - Tool schema extraction from AgentTool instances using to_function_call()
    - Output message handling with assistant tool_calls for Phoenix UI display
    - Token usage tracking from LLM response metadata
    - Comprehensive error handling with fallback serialization
    - Template information capture when available from prompt templates
    - Async/sync execution support with proper OpenTelemetry context propagation

    The implementation follows OpenInference standards to ensure proper display
    of tool calls, messages, and schemas in Phoenix UI observability dashboards.
    """

    def __init__(self, tracer: Any) -> None:
        """
        Initialize the LLM wrapper with OpenTelemetry tracer.

        Args:
            tracer (Any): OpenTelemetry tracer instance for creating LLM spans
        """
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        """
        Wrap LLM generate calls with comprehensive LLM span tracing.

        Creates LLM spans with OpenInference-formatted attributes that capture
        input messages, tool schemas, and output messages for proper Phoenix UI
        display including tool calls and conversation context.

        Args:
            wrapped (callable): Original LLM generate method to be instrumented
            instance (Any): LLM client instance containing model and configuration
            args (tuple): Positional arguments - typically (messages,) for chat completions
            kwargs (dict): Keyword arguments including model, tools, temperature, etc.

        Returns:
            Any: Result from wrapped method execution with comprehensive span attributes
                 capturing LLM interaction details, token usage, and tool call information
        """
        # Check for instrumentation suppression
        if context_api and context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return wrapped(*args, **kwargs)

        # Extract method parameters and build attributes
        attributes = self._build_llm_attributes(wrapped, instance, args, kwargs)

        # Handle async vs sync execution
        if asyncio.iscoroutinefunction(wrapped):
            return self._handle_async_execution(wrapped, args, kwargs, attributes)
        else:
            return self._handle_sync_execution(wrapped, args, kwargs, attributes)

    def _build_llm_attributes(
        self, wrapped: Any, instance: Any, args: Any, kwargs: Any
    ) -> Dict[str, Any]:
        """
        Build comprehensive LLM span attributes with OpenInference formatting.

        Constructs detailed span attributes including input messages, tool schemas,
        model information, and serialized parameters following OpenInference
        standards for proper Phoenix UI visualization and trace analysis.

        Args:
            wrapped (callable): Original LLM method for parameter binding
            instance (Any): LLM client instance with model and configuration attributes
            args (tuple): Positional arguments containing messages and parameters
            kwargs (dict): Keyword arguments with model, tools, and LLM parameters

        Returns:
            Dict[str, Any]: Comprehensive span attributes including OpenInference-formatted
                           input messages, tool schemas, model metadata, and serialized parameters
        """
        # Extract method parameters
        messages = args[0] if args else kwargs.get("messages")
        model = kwargs.get("model") or getattr(instance, "model", None)
        tools = kwargs.get("tools") or []
        agent_name = getattr(instance, "agent_name", None) or "ChatClient"

        # Build base span attributes
        attributes = {
            OPENINFERENCE_SPAN_KIND: LLM,
            INPUT_MIME_TYPE: "application/json",
            OUTPUT_MIME_TYPE: "application/json",
            "agent.name": agent_name,
        }

        if model:
            attributes[LLM_MODEL_NAME] = model

        # Serialize input arguments with proper tool formatting
        input_args = bind_arguments(wrapped, *args, **kwargs)
        input_args = strip_method_args(input_args)

        # Convert tools to proper JSON schemas for tracing
        if "tools" in input_args and input_args["tools"]:
            input_args["tools"] = serialize_tools_for_tracing(input_args["tools"])

        attributes[INPUT_VALUE] = safe_json_dumps(input_args)

        # Add OpenInference message attributes for Phoenix UI
        if messages:
            input_message_attrs = self._extract_input_messages(messages)
            attributes.update(input_message_attrs)

        # Add tool schema attributes
        if tools:
            tools_attrs = extract_tool_schemas(tools)
            attributes.update(tools_attrs)

        # Add context and template information
        attributes.update(get_attributes_from_context())

        # Add prompt template information if available
        prompt_template = getattr(instance, "prompt_template", None)
        if prompt_template is not None:
            template_info = self._extract_template_info(prompt_template, instance)
            attributes["llm.prompt_template"] = safe_json_dumps(template_info)

        return attributes

    def _extract_input_messages(self, messages: Any) -> Dict[str, Any]:
        """
        Extract and format input messages using OpenInference Message standards.

        Converts various message formats (string, dict, object) to the standardized
        OpenInference Message format with proper tool_calls structure, ensuring
        Phoenix UI can correctly display conversation history and tool interactions.

        Args:
            messages (Any): Input messages in various formats - string, list of dicts/objects,
                           or message objects with role, content, and optional tool_calls

        Returns:
            Dict[str, Any]: OpenInference-formatted message attributes compatible with
                           Phoenix UI visualization requirements for conversation display
        """
        logger.debug(f"Extracting input messages of type: {type(messages)}")

        # Convert messages to OpenInference Message format
        oi_messages = convert_messages_to_openinference(messages)

        # Use OpenInference helper to convert to proper span attributes
        return get_input_message_attributes(oi_messages)

    def _extract_template_info(
        self, prompt_template: Any, instance: Any
    ) -> Dict[str, Any]:
        """
        Extract prompt template information for comprehensive LLM tracing.

        Captures template metadata including format, variables, and content
        to provide complete visibility into prompt construction and template
        usage within LLM interactions.

        Args:
            prompt_template (Any): Prompt template object containing template data
            instance (Any): LLM client instance with potential template configuration

        Returns:
            Dict[str, Any]: Template information including type, template content,
                           variables, and serialized template data for tracing
        """
        template_info = {}
        template_info["type"] = getattr(
            prompt_template,
            "template_format",
            getattr(instance, "template_format", None),
        )
        template_info["template"] = getattr(prompt_template, "template", None)
        template_info["variables"] = getattr(prompt_template, "variables", None)

        if hasattr(prompt_template, "model_dump") and callable(
            prompt_template.model_dump
        ):
            template_info.update(prompt_template.model_dump())

        return template_info

    def _handle_async_execution(
        self, wrapped: Any, args: Any, kwargs: Any, attributes: Dict[str, Any]
    ) -> Any:
        """
        Handle asynchronous LLM execution with comprehensive span tracing.

        Manages async LLM calls by creating spans with proper attribute handling,
        result processing, and error management within OpenTelemetry context,
        ensuring complete LLM interaction tracing.

        Args:
            wrapped (callable): Original async LLM method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            attributes (Dict[str, Any]): Pre-built span attributes including input messages and tools

        Returns:
            Any: Coroutine that executes the wrapped method with proper span instrumentation,
                 output processing, and comprehensive error handling
        """

        async def async_wrapper():
            with self._tracer.start_as_current_span(
                "ChatCompletion", attributes=attributes
            ) as span:
                try:
                    result = await wrapped(*args, **kwargs)
                    self._set_output_attributes(span, result)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return async_wrapper()

    def _handle_sync_execution(
        self, wrapped: Any, args: Any, kwargs: Any, attributes: Dict[str, Any]
    ) -> Any:
        """
        Handle synchronous LLM execution with comprehensive span tracing.

        Manages sync LLM calls by creating spans with proper attribute handling,
        result processing, and error management within OpenTelemetry context,
        ensuring complete LLM interaction tracing.

        Args:
            wrapped (callable): Original sync LLM method to execute
            args (tuple): Positional arguments for the wrapped method
            kwargs (dict): Keyword arguments for the wrapped method
            attributes (Dict[str, Any]): Pre-built span attributes including input messages and tools

        Returns:
            Any: Result from wrapped method execution with proper span instrumentation,
                 output processing, and comprehensive error handling
        """
        with self._tracer.start_as_current_span(
            "ChatCompletion", attributes=attributes
        ) as span:
            try:
                result = wrapped(*args, **kwargs)
                self._set_output_attributes(span, result)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def _set_output_attributes(self, span: Any, result: Any) -> None:
        """
        Set comprehensive output attributes based on LLM response structure.

        Processes LLM response results to extract and set proper span attributes
        including serialized output, OpenInference-formatted messages, and
        token usage metrics, with comprehensive error handling and fallback.

        Args:
            span (Any): OpenTelemetry span to set attributes on
            result (Any): LLM response result object containing messages, usage, and metadata
        """
        try:
            # Serialize the entire result object for complete LLM response context
            self._set_serialized_output(span, result)

            # Extract structured message attributes for OpenInference compatibility
            message, metadata = process_llm_response(result)

            if message:
                # Set OpenInference message attributes for Phoenix UI display
                self._set_message_attributes(span, message)

                # Set token counts from metadata if available
                if metadata:
                    self._set_token_attributes(span, metadata)

        except Exception as e:
            logger.warning(f"Error setting output attributes: {e}")
            # Fallback to simple string representation
            span.set_attribute(OUTPUT_VALUE, str(result))
            span.record_exception(e)

    def _set_serialized_output(self, span: Any, result: Any) -> None:
        """
        Set serialized output value with multiple serialization strategies.

        Attempts various serialization methods to capture the complete LLM
        response structure, with graceful fallback to string representation
        for comprehensive output tracing.

        Args:
            span (Any): OpenTelemetry span to set OUTPUT_VALUE attribute on
            result (Any): LLM response result object to be serialized
        """
        try:
            if hasattr(result, "model_dump_json") and callable(result.model_dump_json):
                span.set_attribute(OUTPUT_VALUE, result.model_dump_json())
            elif hasattr(result, "model_dump") and callable(result.model_dump):
                result_dict = result.model_dump()
                span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result_dict))
            else:
                # Fallback: serialize as JSON
                span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result))
        except Exception as e:
            logger.debug(f"Error serializing output: {e}")
            span.set_attribute(OUTPUT_VALUE, str(result))

    def _set_message_attributes(self, span: Any, message: Any) -> None:
        """
        Set OpenInference message attributes for proper Phoenix UI display.

        Converts assistant messages with tool_calls to the OpenInference Message
        format that Phoenix UI expects, ensuring tool calls and responses are
        properly visualized in the observability dashboard.

        Args:
            span (Any): OpenTelemetry span to set message attributes on
            message (Any): Assistant message object with potential tool_calls and content
        """
        logger.debug(
            f"Setting output message attributes for role: {getattr(message, 'role', 'assistant')}"
        )

        # Convert message to OpenInference Message format
        oi_message = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None) or "",
        }

        # Convert tool_calls to OpenInference format if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            from ..message_processors import convert_tool_calls_to_openinference

            oi_tool_calls = convert_tool_calls_to_openinference(message.tool_calls)
            if oi_tool_calls:
                oi_message["tool_calls"] = oi_tool_calls
                logger.debug(f"Converted {len(oi_tool_calls)} output tool calls")

        # Use OpenInference helper to convert to proper span attributes
        output_attrs = get_output_message_attributes([oi_message])
        for attr_key, attr_value in output_attrs.items():
            span.set_attribute(attr_key, attr_value)

    def _set_token_attributes(self, span: Any, metadata: Any) -> None:
        """
        Set token count attributes from LLM response metadata.

        Extracts and sets token usage metrics including input tokens, output tokens,
        and total tokens from LLM response metadata for cost tracking and
        performance analysis in observability dashboards.

        Args:
            span (Any): OpenTelemetry span to set token attributes on
            metadata (Any): Response metadata containing usage information and token counts
        """
        token_attrs = extract_token_usage(metadata)
        for attr_key, attr_value in token_attrs.items():
            span.set_attribute(attr_key, attr_value)


# ============================================================================
# Exported Classes
# ============================================================================

__all__ = [
    "LLMWrapper",
]
