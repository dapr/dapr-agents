import logging
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
)

from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, ValidationError

from .structure import StructureHandler
from dapr_agents.types import ToolCall

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StreamHandler:
    """
    Handles streaming of chat completion responses, processing tool calls and content responses.
    """

    @staticmethod
    def process_stream(
        stream: Iterator[Dict[str, Any]],
        llm_provider: str,
        response_format: Optional[Union[Type[T], Type[Iterable[T]]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream chat completion responses.

        Args:
            stream: The response stream from the API.
            llm_provider: The LLM provider to use (e.g., 'openai').
            response_format: The optional Pydantic model or iterable model for validating the response.

        Yields:
            dict: Each processed and validated chunk from the chat completion response.
        """
        logger.info("Streaming response enabled.")

        try:
            if llm_provider == "openai":
                yield from StreamHandler._process_openai_stream(stream, response_format)
            elif llm_provider == "dapr":
                yield from StreamHandler._process_dapr_stream(stream, response_format)
            else:
                yield from stream
        except Exception as e:
            logger.error(f"An error occurred during streaming: {e}")
            raise

    @staticmethod
    def _process_openai_stream(
        stream: Iterator[Dict[str, Any]],
        response_format: Optional[Union[Type[T], Type[Iterable[T]]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Process OpenAI stream for chat completion.

        Args:
            stream: The response stream from the OpenAI API.
            response_format: The optional Pydantic model or iterable model for validating the response.

        Yields:
            dict: Each processed and validated chunk from the chat completion response.
        """
        content_accumulator = ""
        json_extraction_active = False
        json_brace_level = 0
        json_string_buffer = ""
        tool_calls = {}

        for chunk in stream:
            processed_chunk = StreamHandler._process_openai_chunk(chunk)
            chunk_type = processed_chunk["type"]
            chunk_data = processed_chunk["data"]

            if chunk_type == "content":
                content_accumulator += chunk_data
                yield processed_chunk
            elif chunk_type in ["tool_calls", "function_call"]:
                for tool_chunk in chunk_data:
                    tool_call_index = tool_chunk["index"]
                    tool_call_id = tool_chunk["id"]
                    tool_call_function = tool_chunk["function"]
                    tool_call_arguments = tool_call_function["arguments"]

                    if tool_call_id is not None:
                        tool_calls.setdefault(
                            tool_call_index,
                            {
                                "id": tool_call_id,
                                "type": tool_chunk["type"],
                                "function": {
                                    "name": tool_call_function["name"],
                                    "arguments": tool_call_arguments,
                                },
                            },
                        )

                    # Add tool call arguments to current tool calls
                    tool_calls[tool_call_index]["function"][
                        "arguments"
                    ] += tool_call_arguments

                    # Process Iterable model if provided
                    if (
                        response_format
                        and isinstance(response_format, Iterable) is True
                    ):
                        trimmed_character = tool_call_arguments.strip()
                        # Check beginning of List
                        if trimmed_character == "[" and json_extraction_active is False:
                            json_extraction_active = True
                        # Check beginning of a JSON object
                        elif (
                            trimmed_character == "{" and json_extraction_active is True
                        ):
                            json_brace_level += 1
                            json_string_buffer += trimmed_character
                        # Check the end of a JSON object
                        elif (
                            "}" in trimmed_character and json_extraction_active is True
                        ):
                            json_brace_level -= 1
                            json_string_buffer += trimmed_character.rstrip(",")
                            if json_brace_level == 0:
                                yield from StreamHandler._validate_json_object(
                                    response_format, json_string_buffer
                                )
                                # Reset buffers and counts
                                json_string_buffer = ""
                        elif json_extraction_active is True:
                            json_string_buffer += tool_call_arguments

        if content_accumulator:
            yield {"type": "final_content", "data": content_accumulator}

        if tool_calls:
            yield from StreamHandler._get_final_tool_calls(tool_calls, response_format)

    @staticmethod
    def _process_openai_chunk(chunk: ChatCompletionChunk) -> Dict[str, Any]:
        """
        Process OpenAI chat completion chunk.

        Args:
            chunk: The chunk from the OpenAI API.

        Returns:
            dict: Processed chunk.
        """
        try:
            chunk_dict = chunk.model_dump()

            if chunk_dict.get("choices") and len(chunk_dict["choices"]) > 0:
                choice: Dict = chunk_dict["choices"][0]
                delta: Dict = choice.get("delta", {})

                # Process content
                if delta.get("content") is not None:
                    return {"type": "content", "data": delta["content"], "chunk": chunk}

                # Process tool calls
                if delta.get("tool_calls"):
                    return {
                        "type": "tool_calls",
                        "data": delta["tool_calls"],
                        "chunk": chunk,
                    }

                # Process function calls
                if delta.get("function_call"):
                    return {
                        "type": "function_call",
                        "data": delta["function_call"],
                        "chunk": chunk,
                    }

                # Process finish reason
                if choice.get("finish_reason"):
                    return {
                        "type": "finish",
                        "data": choice["finish_reason"],
                        "chunk": chunk,
                    }

            return {}
        except Exception as e:
            logger.error(f"Error handling OpenAI chat completion chunk: {e}")
            raise

    @staticmethod
    def _process_dapr_stream(
        stream: Iterator[Dict[str, Any]],
        response_format: Optional[Union[Type[T], Type[Iterable[T]]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Process Dapr stream for chat completion.

        Args:
            stream: The response stream from the Dapr API.
            response_format: The optional Pydantic model or iterable model for validating the response.

        Yields:
            dict: Each processed and validated chunk from the chat completion response.
        """
        content_accumulator = ""
        final_usage = None

        for chunk in stream:
            processed_chunk = StreamHandler._process_dapr_chunk(chunk)
            chunk_type = processed_chunk["type"]
            chunk_data = processed_chunk["data"]

            if chunk_type == "content":
                content_accumulator += chunk_data
                yield processed_chunk
            elif chunk_type == "usage":
                final_usage = chunk_data
            elif chunk_type == "context_id":
                yield processed_chunk

        # Yield final content and usage if available
        if content_accumulator:
            yield {"type": "final_content", "data": content_accumulator}

        if final_usage:
            yield {"type": "final_usage", "data": final_usage}

    @staticmethod
    def _process_dapr_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Dapr chat completion chunk.

        Args:
            chunk: The chunk from the Dapr API.

        Returns:
            dict: Processed chunk.
        """
        try:
            # Handle content chunks
            if chunk.get("choices") and len(chunk["choices"]) > 0:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                if delta.get("content") is not None:
                    return {"type": "content", "data": delta["content"], "chunk": chunk}

            # Handle usage information
            if chunk.get("usage"):
                return {"type": "usage", "data": chunk["usage"], "chunk": chunk}

            # Handle context ID
            if chunk.get("context_id"):
                return {
                    "type": "context_id",
                    "data": chunk["context_id"],
                    "chunk": chunk,
                }

            return {"type": "unknown", "data": chunk, "chunk": chunk}
        except Exception as e:
            logger.error(f"Error handling Dapr chat completion chunk: {e}")
            raise

    @staticmethod
    def _validate_json_object(
        response_format: Optional[Union[Type[T], Type[Iterable[T]]]],
        json_string_buffer: str,
    ):
        try:
            model_class = get_args(response_format)[0]
            # Return current tool call
            structured_output = StructureHandler.validate_response(
                json_string_buffer, model_class
            )
            if isinstance(structured_output, model_class):
                logger.info("Structured output was successfully validated.")
                yield {"type": "structured_output", "data": structured_output}
        except ValidationError as validation_error:
            logger.error(
                f"Validation error: {validation_error} with JSON: {json_string_buffer}"
            )

    @staticmethod
    def _get_final_tool_calls(
        tool_calls: Dict[int, Any],
        response_format: Optional[Union[Type[T], Type[Iterable[T]]]],
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield final tool calls after processing.

        Args:
            tool_calls: The dictionary of accumulated tool calls.
            response_format: The response model for validation.

        Yields:
            dict: Each processed and validated tool call.
        """
        for tool in tool_calls.values():
            if response_format and isinstance(response_format, Iterable) is False:
                structured_output = StructureHandler.validate_response(
                    tool["function"]["arguments"], response_format
                )
                if isinstance(structured_output, response_format):
                    logger.info("Structured output was successfully validated.")
                    yield {"type": "structured_output", "data": structured_output}
            else:
                tool_call = ToolCall(**tool)
                yield {"type": "final_tool_call", "data": tool_call}
