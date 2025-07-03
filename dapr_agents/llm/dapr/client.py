from dapr_agents.types.llm import DaprInferenceClientConfig
from dapr_agents.llm.base import LLMClientBase
from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput
from dapr.clients.grpc._response import ConversationResponse
from typing import Dict, Any, List, Iterator, Optional
from pydantic import model_validator

import logging

logger = logging.getLogger(__name__)


class DaprInferenceClient:
    def __init__(self):
        self.dapr_client = DaprClient()

    def translate_to_json(self, response: ConversationResponse) -> dict:
        response_dict = {"outputs": []}

        for output in response.outputs:
            output_dict = {
                "result": output.result,
            }

            # Add tool calls if present - use get_tool_calls() method
            tool_calls = None
            if hasattr(output, "get_tool_calls"):
                tool_calls = output.get_tool_calls()
            elif hasattr(output, "tool_calls") and output.tool_calls:
                tool_calls = output.tool_calls

            if tool_calls:
                output_dict["tool_calls"] = []
                for tool_call in tool_calls:
                    # Handle different tool call structures
                    if hasattr(tool_call, "function"):
                        # OpenAI-style structure with nested function
                        tool_call_dict = {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    else:
                        # Dapr SDK structure with direct name/arguments
                        tool_call_dict = {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                    output_dict["tool_calls"].append(tool_call_dict)

            # Add finish reason if present
            if hasattr(output, "finish_reason") and output.finish_reason:
                output_dict["finish_reason"] = output.finish_reason

            response_dict["outputs"].append(output_dict)

        return response_dict

    def chat_completion(
        self,
        llm: str,
        conversation_inputs: List[ConversationInput],
        scrub_pii: bool | None = None,
        temperature: float | None = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        # Extract tools from ConversationInput objects for request-level tools parameter
        tools = []
        for conv_input in conversation_inputs:
            if hasattr(conv_input, "tools") and conv_input.tools:
                tools.extend(conv_input.tools)
                # Clear tools from ConversationInput since we're passing them at request level
                conv_input.tools = None

        # Remove duplicates while preserving order
        unique_tools = []
        seen_names = set()
        for tool in tools:
            if tool.name not in seen_names:
                unique_tools.append(tool)
                seen_names.add(tool.name)

        response = self.dapr_client.converse_alpha1(
            name=llm,
            inputs=conversation_inputs,
            scrub_pii=scrub_pii,
            temperature=temperature,
            parameters=parameters,
            tools=unique_tools if unique_tools else None,  # Pass tools at request level
        )
        output = self.translate_to_json(response)

        return output

    def chat_completion_stream(
        self,
        llm: str,
        conversation_inputs: List[ConversationInput],
        context_id: str | None = None,
        scrub_pii: bool | None = None,
        temperature: float | None = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream chat completion responses using Dapr's converse_stream_alpha1 API.

        Args:
            llm: Name of the LLM component to use
            conversation_inputs: List of conversation inputs
            context_id: Optional context ID for continuing conversation
            scrub_pii: Optional flag to scrub PII from inputs and outputs
            temperature: Optional temperature setting for the LLM

        Yields:
            Dict[str, Any]: JSON-formatted streaming response chunks compatible with common LLM APIs
        """
        logger.info(f"Starting streaming conversation with LLM component: {llm}")

        # Extract tools from ConversationInput objects for request-level tools parameter
        tools = []
        for conv_input in conversation_inputs:
            if hasattr(conv_input, "tools") and conv_input.tools:
                tools.extend(conv_input.tools)
                # Clear tools from ConversationInput since we're passing them at request level
                conv_input.tools = None

        # Remove duplicates while preserving order
        unique_tools = []
        seen_names = set()
        for tool in tools:
            if tool.name not in seen_names:
                unique_tools.append(tool)
                seen_names.add(tool.name)

        logger.info(f"🔧 Extracted {len(unique_tools)} unique tools for request level")

        try:
            # Use converse_stream_alpha1 and transform to JSON format
            chunk_count = 0
            for chunk in self.dapr_client.converse_stream_alpha1(
                name=llm,
                inputs=conversation_inputs,
                context_id=context_id,
                scrub_pii=scrub_pii,
                temperature=temperature,
                parameters=parameters,
                tools=unique_tools
                if unique_tools
                else None,  # Pass tools at request level
            ):
                chunk_count += 1
                logger.debug(f"📦 Received chunk {chunk_count}: type={type(chunk)}")
                logger.debug(f"   • Has chunk attr: {hasattr(chunk, 'chunk')}")
                logger.debug(f"   • Has complete attr: {hasattr(chunk, 'complete')}")
                logger.debug(
                    f"   • Has context_id attr: {hasattr(chunk, 'context_id')}"
                )

                # Debug chunk content
                if hasattr(chunk, "chunk"):
                    logger.debug(f"   • chunk.chunk: {chunk.chunk}")
                    if chunk.chunk and hasattr(chunk.chunk, "content"):
                        logger.debug(
                            f"   • chunk.chunk.content: {repr(chunk.chunk.content)}"
                        )

                if hasattr(chunk, "complete"):
                    logger.debug(f"   • chunk.complete: {chunk.complete}")
                    if chunk.complete:
                        logger.debug(
                            f"   • complete has tool_calls: {hasattr(chunk.complete, 'tool_calls')}"
                        )
                        logger.debug(
                            f"   • complete has usage: {hasattr(chunk.complete, 'usage')}"
                        )
                        if (
                            hasattr(chunk.complete, "tool_calls")
                            and chunk.complete.tool_calls
                        ):
                            logger.debug(
                                f"   • tool_calls count: {len(chunk.complete.tool_calls)}"
                            )

                # Handle streaming chunks (content or tool calls)
                if hasattr(chunk, "chunk") and chunk.chunk:
                    chunk_obj = chunk.chunk

                    # Handle text content
                    if hasattr(chunk_obj, "content") and chunk_obj.content:
                        chunk_content = chunk_obj.content
                        logger.debug(
                            f"   • Streaming content: '{str(chunk_content)[:50]}...'"
                        )

                        chunk_dict = {
                            "choices": [
                                {
                                    "delta": {
                                        "content": chunk_content,
                                        "role": "assistant",
                                    },
                                    "index": 0,
                                    "finish_reason": None,
                                }
                            ],
                            "context_id": getattr(chunk, "context_id", None),
                            "usage": None,
                        }
                        yield chunk_dict

                    # Handle tool calls in chunk.chunk.parts
                    elif hasattr(chunk_obj, "parts") and chunk_obj.parts:
                        logger.debug(
                            f"   • Found {len(chunk_obj.parts)} parts in chunk"
                        )

                        for part in chunk_obj.parts:
                            if hasattr(part, "tool_call") and part.tool_call:
                                tool_call = part.tool_call
                                logger.info(f"🔧 Found tool call: {tool_call.name}")

                                tool_call_chunk = {
                                    "choices": [
                                        {
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "id": tool_call.id,
                                                        "type": tool_call.type,
                                                        "function": {
                                                            "name": tool_call.name,
                                                            "arguments": tool_call.arguments,
                                                        },
                                                    }
                                                ]
                                            },
                                            "index": 0,
                                            "finish_reason": None,
                                        }
                                    ],
                                    "context_id": getattr(chunk, "context_id", None),
                                    "usage": None,
                                }
                                yield tool_call_chunk

                        # Send finish reason for tool calls
                        if (
                            hasattr(chunk_obj, "finish_reason")
                            and chunk_obj.finish_reason == "tool_calls"
                        ):
                            final_tool_chunk = {
                                "choices": [
                                    {
                                        "delta": {},
                                        "index": 0,
                                        "finish_reason": "tool_calls",
                                    }
                                ],
                                "context_id": getattr(chunk, "context_id", None),
                                "usage": None,
                            }
                            yield final_tool_chunk

                # Handle completion chunk (chunk.complete with tool calls and usage)
                elif hasattr(chunk, "complete") and chunk.complete:
                    logger.debug("   • Complete chunk received")
                    complete = chunk.complete

                    # Check for tool calls in complete chunk
                    if hasattr(complete, "tool_calls") and complete.tool_calls:
                        logger.info(
                            f"🔧 Found {len(complete.tool_calls)} tool calls in complete chunk"
                        )

                        # Send tool calls as separate chunks (OpenAI-compatible streaming)
                        for i, tool_call in enumerate(complete.tool_calls):
                            logger.debug(
                                f"   • Tool call {i+1}: {tool_call.function.name}"
                            )

                            tool_call_chunk = {
                                "choices": [
                                    {
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "id": tool_call.id,
                                                    "type": tool_call.type,
                                                    "function": {
                                                        "name": tool_call.function.name,
                                                        "arguments": tool_call.function.arguments,
                                                    },
                                                }
                                            ]
                                        },
                                        "index": 0,
                                        "finish_reason": None,
                                    }
                                ],
                                "context_id": getattr(chunk, "context_id", None),
                                "usage": None,
                            }
                            yield tool_call_chunk

                        # Send final chunk with finish_reason for tool calls
                        final_tool_chunk = {
                            "choices": [
                                {"delta": {}, "index": 0, "finish_reason": "tool_calls"}
                            ],
                            "context_id": getattr(chunk, "context_id", None),
                            "usage": None,
                        }
                        yield final_tool_chunk

                    # Handle usage information
                    if hasattr(complete, "usage") and complete.usage:
                        logger.debug(f"   • Usage info: {complete.usage}")

                        usage_chunk = {
                            "choices": [],
                            "context_id": getattr(chunk, "context_id", None),
                            "usage": {
                                "prompt_tokens": getattr(
                                    complete.usage, "prompt_tokens", 0
                                ),
                                "completion_tokens": getattr(
                                    complete.usage, "completion_tokens", 0
                                ),
                                "total_tokens": getattr(
                                    complete.usage, "total_tokens", 0
                                ),
                            },
                        }
                        yield usage_chunk

                    # If no tool calls, send normal completion
                    if not (hasattr(complete, "tool_calls") and complete.tool_calls):
                        finish_chunk = {
                            "choices": [
                                {
                                    "delta": {},
                                    "index": 0,
                                    "finish_reason": getattr(
                                        complete, "finish_reason", "stop"
                                    ),
                                }
                            ],
                            "context_id": getattr(chunk, "context_id", None),
                            "usage": None,
                        }
                        yield finish_chunk

                # Handle context ID updates
                elif hasattr(chunk, "context_id") and chunk.context_id:
                    logger.debug(f"   • Context ID: {chunk.context_id}")
                    context_chunk = {
                        "choices": [],
                        "context_id": chunk.context_id,
                        "usage": None,
                    }
                    yield context_chunk

                else:
                    logger.debug("   • Unknown chunk type, skipping")

            logger.info(f"✅ Streaming completed, processed {chunk_count} chunks")

        except Exception as e:
            logger.error(f"❌ Error during streaming conversation: {e}")
            raise


class DaprInferenceClientBase(LLMClientBase):
    """
    Base class for managing Dapr Inference API clients.
    Handles client initialization, configuration, and shared logic.
    """

    @model_validator(mode="before")
    def validate_and_initialize(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return values

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes private attributes after validation.
        """
        self._provider = "dapr"

        # Set up the private config and client attributes
        self._config = self.get_config()
        self._client = self.get_client()
        return super().model_post_init(__context)

    def get_config(self) -> DaprInferenceClientConfig:
        """
        Returns the appropriate configuration for the Dapr Conversation API.
        """
        return DaprInferenceClientConfig()

    def get_client(self) -> DaprInferenceClient:
        """
        Initializes and returns the Dapr Inference client.
        """
        return DaprInferenceClient()

    @classmethod
    def from_config(
        cls, client_options: DaprInferenceClientConfig, timeout: float = 1500
    ):
        """
        Initializes the DaprInferenceClientBase using DaprInferenceClientConfig.

        Args:
            client_options: The configuration options for the client.
            timeout: Timeout for requests (default is 1500 seconds).

        Returns:
            DaprInferenceClientBase: The initialized client instance.
        """
        return cls()

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def client(self) -> DaprInferenceClient:
        return self._client
