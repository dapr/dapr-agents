from dapr_agents.llm.dapr.client import DaprInferenceClientBase
from dapr_agents.llm.utils import RequestHandler, ResponseHandler
from dapr_agents.prompt.prompty import Prompty
from dapr_agents.types.message import BaseMessage
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.tool import AgentTool
from dapr_agents.types import (
    ChatCompletion,
    Choice,
    MessageContent,
    ToolCall,
    FunctionCall,
)
from dapr.clients.grpc._request import (
    ConversationInput,
    Tool,
    ContentPart,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from typing import (
    Union,
    Optional,
    Iterable,
    Dict,
    Any,
    List,
    Iterator,
    Type,
    Literal,
    ClassVar,
)
from pydantic import BaseModel
from pathlib import Path
import logging
import os
import json
import time

logger = logging.getLogger(__name__)


class DaprChatClient(DaprInferenceClientBase, ChatClientBase):
    """
    Simplified Dapr Chat Client using ONLY ConversationInput.tools field.
    This eliminates the redundant parameters["tools"] approach.
    """

    SUPPORTED_STRUCTURED_MODES: ClassVar[set] = {"function_call"}

    # Public component name attribute
    component_name: Optional[str] = None

    def _convert_dict_to_chat_completion(self, response_dict: dict) -> ChatCompletion:
        """
        Convert our dict response to a proper ChatCompletion object for Agent compatibility.

        Args:
            response_dict: Dict with keys like 'content', 'tool_calls', 'finish_reason'

        Returns:
            ChatCompletion object that Agent can work with
        """
        # Convert tool calls to proper ToolCall objects
        tool_calls = None
        if response_dict.get("tool_calls"):
            tool_calls = []
            for tc in response_dict["tool_calls"]:
                function_call = FunctionCall(
                    name=tc["function"]["name"], arguments=tc["function"]["arguments"]
                )
                tool_call = ToolCall(
                    id=tc["id"], type=tc["type"], function=function_call
                )
                tool_calls.append(tool_call)

        # Create MessageContent
        message_content = MessageContent(
            content=response_dict.get("content"),
            role="assistant",
            tool_calls=tool_calls,
        )

        # Create Choice
        choice = Choice(
            finish_reason=response_dict.get("finish_reason", "stop"),
            index=0,
            message=message_content,
            logprobs=None,
        )

        # Create ChatCompletion
        chat_completion = ChatCompletion(
            choices=[choice],
            created=int(time.time()),
            id=f"dapr-{int(time.time())}",
            model="dapr-llm",
            object="chat.completion",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

        return chat_completion

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization setup."""
        super().model_post_init(__context)

        # Set the default LLM component: component_name > environment variable > "echo"
        self._llm_component = self.component_name or os.environ.get(
            "DAPR_LLM_COMPONENT_DEFAULT", "echo"
        )

    @classmethod
    def from_prompty(
        cls,
        prompty_source: Union[str, Path],
        timeout: Union[int, float, Dict[str, Any]] = 1500,
    ) -> "DaprChatClient":
        """
        Initializes a DaprChatClient using a Prompty source.

        Args:
            prompty_source (Union[str, Path]): The source of the Prompty file, which can be a path to a file
                or inline Prompty content as a string.
            timeout (Union[int, float, Dict[str, Any]], optional): Timeout for requests, defaults to 1500 seconds.

        Returns:
            DaprChatClient: An instance of DaprChatClient configured with the model settings from the Prompty source.
        """

        # Load the Prompty instance from the provided source
        prompty_instance = Prompty.load(prompty_source)

        # Generate the prompt template from the Prompty instance
        prompt_template = Prompty.to_prompt_template(prompty_instance)

        # Initialize the DaprChatClient based on the Prompty model configuration
        return cls.model_validate(
            {
                "timeout": timeout,
                "prompty": prompty_instance,
                "prompt_template": prompt_template,
            }
        )

    def _convert_tools_to_sdk_format(self, tools: List[AgentTool]) -> List[Tool]:
        """
        Convert AgentTool objects to Python SDK Tool objects.

        Args:
            tools: List of AgentTool objects

        Returns:
            List of Tool objects for the Python SDK
        """
        sdk_tools = []

        for agent_tool in tools:
            # Get the tool definition in the correct format for dapr
            tool_def = agent_tool.to_function_call("dapr")

            # Extract the function definition from the OpenAI-compatible format
            if tool_def.get("type") == "function" and "function" in tool_def:
                func_def = tool_def["function"]
            else:
                # Fallback if it's already in function format
                func_def = tool_def

            # Use the original tool name directly
            tool_name = func_def["name"]

            # Create Tool using the correct constructor format
            sdk_tool = Tool(
                type="function",
                name=tool_name,
                description=func_def["description"],
                parameters=json.dumps(
                    func_def["parameters"]
                ),  # Convert dict to JSON string
            )

            sdk_tools.append(sdk_tool)

        return sdk_tools

    def convert_to_conversation_inputs(
        self, inputs: List[Dict[str, Any]], tools: Optional[List[Tool]] = None
    ) -> List[ConversationInput]:
        """
        Convert input dictionaries to ConversationInput objects.

        Args:
            inputs: List of input dictionaries
            tools: Optional list of Tool objects to attach to user messages

        Returns:
            List of ConversationInput objects
        """
        conversation_inputs = []

        for item in inputs:
            # Handle tool messages with the new format
            if item.get("role") == "tool":
                # Tool result message - use the new parts format
                tool_call_id = item.get("tool_call_id")
                name = item.get("name")
                content = item.get("content")

                if tool_call_id and name and content:
                    conv_input = ConversationInput.from_tool_result_simple(
                        tool_name=name, call_id=tool_call_id, result=content
                    )
                else:
                    # Fallback to old format if missing required fields
                    conv_input = ConversationInput(
                        content=item["content"],
                        role=item.get("role"),
                        scrub_pii=item.get("scrubPII") == "true",
                    )
            elif item.get("role") == "assistant" and item.get("tool_calls"):
                # Assistant message with tool_calls - use the new parts format
                parts = []

                # Add text content if present
                if item.get("content"):
                    parts.append(ContentPart(text=TextContent(text=item["content"])))

                # Add tool calls as parts
                for tool_call in item["tool_calls"]:
                    # Handle the OpenAI-style tool call format
                    tool_call_content = ToolCallContent(
                        id=tool_call["id"],
                        type=tool_call["type"],
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"],
                    )
                    parts.append(ContentPart(tool_call=tool_call_content))

                conv_input = ConversationInput(
                    role="assistant",
                    parts=parts,
                    scrub_pii=item.get("scrubPII") == "true",
                )
            else:
                # Regular message - use legacy format for compatibility
                conv_input = ConversationInput(
                    content=item.get("content"),  # Use .get() to handle missing content
                    role=item.get("role"),
                    scrub_pii=item.get("scrubPII") == "true",
                )

                # ✅ Add tools ONLY to user messages (following SDK best practices)
                if tools and item.get("role") == "user":
                    conv_input.tools = tools

            conversation_inputs.append(conv_input)

        return conversation_inputs

    def generate_raw(
        self,
        messages: Union[
            str,
            Dict[str, Any],
            BaseMessage,
            Iterable[Union[Dict[str, Any], BaseMessage]],
        ] = None,
        input_data: Optional[Dict[str, Any]] = None,
        llm_component: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        structured_mode: Literal["function_call"] = "function_call",
        scrubPII: Optional[bool] = False,
        temperature: Optional[float] = None,
        stream: Optional[bool] = False,
        context_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Generate chat completions and return the raw response with parts.

        This method returns the raw ConversationResponse which includes the parts
        needed for proper multi-turn tool calling conversations.

        Returns:
            The raw ConversationResponse from the Python SDK
        """
        if structured_mode not in self.SUPPORTED_STRUCTURED_MODES:
            raise ValueError(
                f"Invalid structured_mode '{structured_mode}'. Must be one of {self.SUPPORTED_STRUCTURED_MODES}."
            )

        # If input_data is provided, check for a prompt_template
        if input_data:
            if not self.prompt_template:
                raise ValueError(
                    "Inputs are provided but no 'prompt_template' is set. Please set a 'prompt_template' to use the input_data."
                )

            logger.info("Using prompt template to generate messages.")
            messages = self.prompt_template.format_prompt(**input_data)

        # Ensure we have messages at this point
        if not messages:
            raise ValueError("Either 'messages' or 'input_data' must be provided.")

        # Process and normalize the messages
        params = {"inputs": RequestHandler.normalize_chat_messages(messages)}

        # Merge Prompty parameters if available, then override with any explicit kwargs
        if self.prompty:
            params = {**self.prompty.model.parameters.model_dump(), **params, **kwargs}
        else:
            params.update(kwargs)

        # ✅ SIMPLIFIED: Only process AgentTool objects, no dict tools
        sdk_tools = None
        if tools:
            # Filter to only AgentTool objects
            agent_tools = [tool for tool in tools if isinstance(tool, AgentTool)]
            if agent_tools:
                sdk_tools = self._convert_tools_to_sdk_format(agent_tools)
                logger.info(f"Converted {len(agent_tools)} tools to SDK format")

        # Override stream parameter if explicitly provided
        if stream is not None:
            params["stream"] = stream

        # ✅ SIMPLIFIED: Convert inputs with tools attached directly
        inputs = self.convert_to_conversation_inputs(params["inputs"], sdk_tools)

        # ✅ SIMPLIFIED: No tools in parameters - only basic conversation parameters
        conversation_parameters = {}

        # Add other parameters (excluding tools, inputs, and unsupported params)
        for key, value in params.items():
            if key not in [
                "inputs",
                "tools",
                "stream",
            ]:  # Skip inputs, tools, and stream
                conversation_parameters[key] = value

        try:
            if params.get("stream", False):
                raise ValueError("Streaming not supported in generate_raw method")
            else:
                logger.info("Invoking the Dapr Conversation API (raw mode).")
                # Use the raw dapr client directly to get the raw response
                raw_response = self.client.dapr_client.converse_alpha1(
                    name=llm_component or self._llm_component,
                    inputs=inputs,
                    tools=sdk_tools,  # Pass tools at request level
                    scrub_pii=scrubPII,
                    temperature=temperature,
                    parameters=conversation_parameters,
                )
                logger.info("Chat completion completed successfully (raw mode).")
                return raw_response

        except Exception as e:
            logger.error(f"Error in Dapr conversation (raw mode): {e}")
            raise

    def generate(
        self,
        messages: Union[
            str,
            Dict[str, Any],
            BaseMessage,
            Iterable[Union[Dict[str, Any], BaseMessage]],
        ] = None,
        input_data: Optional[Dict[str, Any]] = None,
        llm_component: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        structured_mode: Literal["function_call"] = "function_call",
        scrubPII: Optional[bool] = False,
        temperature: Optional[float] = None,
        stream: Optional[bool] = False,
        context_id: Optional[str] = None,
        **kwargs,
    ) -> Union[Iterator[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate chat completions using simplified tool calling approach.

        Args:
            messages: Either pre-set messages or None if using input_data.
            input_data: Input variables for prompt templates.
            llm_component: Name of the LLM component to use for the request.
            tools: List of tools for the request (AgentTool objects only).
            response_format: Optional Pydantic model for structured response parsing.
            structured_mode: Mode for structured output: "function_call".
            scrubPII: Optional flag to obfuscate sensitive information.
            temperature: Temperature setting for the LLM.
            stream: Whether to stream the response.
            context_id: Optional context ID for continuing an existing conversation.
            **kwargs: Additional parameters for the language model.

        Returns:
            The chat completion response(s).
        """
        if structured_mode not in self.SUPPORTED_STRUCTURED_MODES:
            raise ValueError(
                f"Invalid structured_mode '{structured_mode}'. Must be one of {self.SUPPORTED_STRUCTURED_MODES}."
            )

        # If input_data is provided, check for a prompt_template
        if input_data:
            if not self.prompt_template:
                raise ValueError(
                    "Inputs are provided but no 'prompt_template' is set. Please set a 'prompt_template' to use the input_data."
                )

            logger.info("Using prompt template to generate messages.")
            messages = self.prompt_template.format_prompt(**input_data)

        # Ensure we have messages at this point
        if not messages:
            raise ValueError("Either 'messages' or 'input_data' must be provided.")

        # Process and normalize the messages
        params = {"inputs": RequestHandler.normalize_chat_messages(messages)}

        # Merge Prompty parameters if available, then override with any explicit kwargs
        if self.prompty:
            params = {**self.prompty.model.parameters.model_dump(), **params, **kwargs}
        else:
            params.update(kwargs)

        # ✅ SIMPLIFIED: Only process AgentTool objects, no dict tools
        sdk_tools = None
        if tools:
            # Filter to only AgentTool objects
            agent_tools = [tool for tool in tools if isinstance(tool, AgentTool)]
            if agent_tools:
                sdk_tools = self._convert_tools_to_sdk_format(agent_tools)
                logger.info(f"Converted {len(agent_tools)} tools to SDK format")

        # Override stream parameter if explicitly provided
        if stream is not None:
            params["stream"] = stream

        # ✅ SIMPLIFIED: Convert inputs with tools attached directly
        inputs = self.convert_to_conversation_inputs(params["inputs"], sdk_tools)

        # ✅ SIMPLIFIED: No tools in parameters - only basic conversation parameters
        conversation_parameters = {}

        # Add other parameters (excluding tools, inputs, and unsupported params)
        for key, value in params.items():
            if key not in [
                "inputs",
                "tools",
                "stream",
            ]:  # Skip inputs, tools, and stream
                conversation_parameters[key] = value

        try:
            # Use streaming or non-streaming API based on the stream parameter
            if params.get("stream", False):
                logger.info("Invoking the Dapr Streaming Conversation API.")
                return self._handle_streaming(
                    llm_component,
                    inputs,
                    context_id,
                    scrubPII,
                    temperature,
                    conversation_parameters,
                )
            else:
                logger.info("Invoking the Dapr Conversation API.")
                response = self.client.chat_completion(
                    llm=llm_component or self._llm_component,
                    conversation_inputs=inputs,
                    # context_id is not supported in non-streaming mode
                    scrub_pii=scrubPII,
                    temperature=temperature,
                    parameters=conversation_parameters,  # ✅ NO tools here!
                )
                logger.info("Chat completion completed successfully.")

                # ✅ SIMPLIFIED: Handle tool calls from response
                if response and response.get("outputs"):
                    output = response["outputs"][0]

                    # Extract and return the relevant information
                    result_dict = {
                        "content": output.get("result"),
                        "finish_reason": output.get("finish_reason", "stop"),
                    }

                    # Check for tool calls in the response
                    if "tool_calls" in output and output["tool_calls"]:
                        result_dict["tool_calls"] = output["tool_calls"]
                        logger.info(f"Received {len(output['tool_calls'])} tool calls")
                        for i, tool_call in enumerate(output["tool_calls"]):
                            logger.info(f"Tool call {i+1}: {tool_call}")

                    # Convert to ChatCompletion for Agent compatibility
                    return self._convert_dict_to_chat_completion(result_dict)

                # Fallback for unexpected response format - try to convert anyway
                fallback_dict = {"content": str(response), "finish_reason": "stop"}
                return self._convert_dict_to_chat_completion(fallback_dict)

        except Exception as e:
            logger.error(f"Error in Dapr conversation: {e}")
            raise

    def _handle_streaming(
        self,
        llm_component,
        inputs,
        context_id,
        scrub_pii,
        temperature,
        conversation_parameters,
    ):
        """Handle streaming responses separately to avoid generator issues."""
        response_stream = self.client.chat_completion_stream(
            llm=llm_component or self._llm_component,
            conversation_inputs=inputs,
            context_id=context_id,
            scrub_pii=scrub_pii,
            temperature=temperature,
            parameters=conversation_parameters,  # ✅ NO tools here!
        )
        logger.info("Streaming chat completion started successfully.")

        for chunk in response_stream:
            yield chunk
