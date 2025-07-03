from typing import Dict, Any, Optional, List, Type, Union, Iterable, Literal
from dapr_agents.prompt.prompty import Prompty, PromptyHelper
from dapr_agents.types.message import BaseMessage
from dapr_agents.llm.utils import StructureHandler
from dapr_agents.tool.utils.tool import ToolHelper
from pydantic import BaseModel, ValidationError
from dapr_agents.tool.base import AgentTool

import logging

logger = logging.getLogger(__name__)


class RequestHandler:
    """
    Handles the preparation of requests for language models.
    """

    @staticmethod
    def process_prompty_messages(
        prompty: Prompty, inputs: Dict[str, Any] = {}
    ) -> List[Dict[str, Any]]:
        """
        Process and format messages based on Prompty template and provided inputs.

        Args:
            prompty (Prompty): The Prompty instance containing the template and settings.
            inputs (Dict[str, Any]): Input variables for the Prompty template (default is an empty dictionary).

        Returns:
            List[Dict[str, Any]]: Processed and prepared messages.
        """
        # Prepare inputs and generate messages from Prompty content
        api_type = prompty.model.api
        prepared_inputs = PromptyHelper.prepare_inputs(
            inputs, prompty.inputs, prompty.sample
        )
        messages = PromptyHelper.to_prompt(
            prompty.content, prepared_inputs, api_type=api_type
        )

        return messages

    @staticmethod
    def normalize_chat_messages(
        messages: Union[
            str,
            Dict[str, Any],
            BaseMessage,
            Iterable[Union[Dict[str, Any], BaseMessage]],
        ],
    ) -> List[Dict[str, Any]]:
        """
        Normalize and validate the input messages into a list of dictionaries.

        Args:
            messages (Union[str, Dict[str, Any], BaseMessage, Iterable[Union[Dict[str, Any], BaseMessage]]]):
                Input messages in various formats (string, dict, BaseMessage, or an iterable).

        Returns:
            List[Dict[str, Any]]: A list of normalized message dictionaries with keys 'role' and 'content'.

        Raises:
            ValueError: If the input format is unsupported or if required fields are missing in a dictionary.
        """
        # Initialize an empty list to store the normalized messages
        normalized_messages = []

        # Use a queue to process messages iteratively and handle nested structures
        queue = [messages]

        while queue:
            msg = queue.pop(0)
            if isinstance(msg, str):
                normalized_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, BaseMessage):
                normalized_messages.append(msg.model_dump())
            elif isinstance(msg, dict):
                role = msg.get("role")
                if role not in {"user", "assistant", "tool", "system"}:
                    raise ValueError(
                        f"Unrecognized role '{role}'. Supported roles are 'user', 'assistant', 'tool', or 'system'."
                    )
                normalized_messages.append(msg)
            elif isinstance(msg, Iterable) and not isinstance(msg, (str, dict)):
                queue.extend(msg)
            else:
                raise ValueError(f"Unsupported message format: {type(msg)}")
        return normalized_messages

    @staticmethod
    def process_params(
        params: Dict[str, Any],
        llm_provider: str,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        structured_mode: Literal["json", "function_call"] = "json",
    ) -> Dict[str, Any]:
        """
        Prepare request parameters for the language model.

        Args:
            params: Parameters for the request.
            llm_provider: The LLM provider to use (e.g., 'openai').
            tools: List of tools to include in the request.
            response_format: Either a Pydantic model (for function calling)
                            or a JSON Schema definition/dict (for raw JSON structured output).
            structured_mode: The mode of structured output: 'json' or 'function_call'.
                            Defaults to 'json'.

        Returns:
            Dict[str, Any]: Prepared request parameters.
        """
        if tools:
            logger.info("Tools are available in the request.")
            # Convert AgentTool objects to dict format for the provider
            tool_dicts = []
            for tool in tools:
                if isinstance(tool, AgentTool):
                    tool_dicts.append(
                        ToolHelper.format_tool(tool, tool_format=llm_provider)
                    )
                else:
                    tool_dicts.append(
                        ToolHelper.format_tool(tool, tool_format=llm_provider)
                    )
            params["tools"] = tool_dicts

        if response_format:
            logger.info(f"Structured Mode Activated! Mode={structured_mode}.")
            params = StructureHandler.generate_request(
                response_format=response_format,
                llm_provider=llm_provider,
                structured_mode=structured_mode,
                **params,
            )

        return params

    @staticmethod
    def validate_request(
        request: Union[BaseModel, Dict[str, Any]], request_class: Type[BaseModel]
    ) -> BaseModel:
        """
        Validate and transform a dictionary into a Pydantic object.

        Args:
            request (Union[BaseModel, Dict[str, Any]]): The request data as a dictionary or a Pydantic object.
            request_class (Type[BaseModel]): The Pydantic model class for validation.

        Returns:
            BaseModel: A validated Pydantic object.

        Raises:
            ValueError: If validation fails.
        """
        if isinstance(request, dict):
            try:
                request = request_class(**request)
            except ValidationError as e:
                raise ValueError(f"Validation error: {e}")

        try:
            validated_request = request_class.model_validate(request)
        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")

        return validated_request
