from dapr_agents.types import AgentError, AssistantMessage, ChatCompletion, ToolMessage
from dapr_agents.agent import AgentBase
from typing import List, Optional, Dict, Any, Union
from pydantic import Field, ConfigDict
import logging

logger = logging.getLogger(__name__)

class ToolCallAgent(AgentBase):
    """
    Agent that manages tool calls and conversations using a language model.
    It integrates tools and processes them based on user inputs and task orchestration.
    """

    tool_history: List[ToolMessage] = Field(default_factory=list, description="Executed tool calls during the conversation.")
    tool_choice: Optional[str] = Field(default=None, description="Strategy for selecting tools ('auto', 'required', 'none'). Defaults to 'auto' if tools are provided.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize the agent's settings, such as tool choice and parent setup.
        Sets the tool choice strategy based on provided tools.
        """
        self.tool_choice = self.tool_choice or ('auto' if self.tools else None)
        
        # Proceed with base model setup
        super().model_post_init(__context)

    async def run(self, input_data: Optional[Union[str, Dict[str, Any]]] = None) -> Any:
        """
        Asynchronously executes the agent's main task using the provided input or memory context.

        Args:
            input_data (Optional[Union[str, Dict[str, Any]]]): User input as string or dict.

        Returns:
            Any: The agent's final output.

        Raises:
            AgentError: If user input is invalid or tool execution fails.
        """
        logger.debug(f"Agent run started with input: {input_data if input_data else 'Using memory context'}")

        # Format messages; construct_messages already includes chat history.
        messages = self.construct_messages(input_data or {})
        user_message = self.get_last_user_message(messages)
        
        if input_data and user_message:
            # Add the new user message to memory only if input_data is provided and user message exists
            self.memory.add_message(user_message)

        # Always print the last user message for context, even if no input_data is provided
        if user_message:
            self.text_formatter.print_message(user_message)

        # Process conversation iterations
        return await self.process_iterations(messages)
    
    async def process_response(self, tool_calls: List[dict]) -> None:
        """
        Asynchronously executes tool calls and appends tool results to memory.

        Args:
            tool_calls (List[dict]): Tool calls returned by the LLM.

        Raises:
            AgentError: If a tool execution fails.
        """
        for tool in tool_calls:
            function_name = tool.function.name
            try:
                logger.info(f"Executing {function_name} with arguments {tool.function.arguments}")
                result = await self.tool_executor.run_tool(function_name, **tool.function.arguments_dict)
                tool_message = ToolMessage(tool_call_id=tool.id, name=function_name, content=str(result))
                self.text_formatter.print_message(tool_message)
                self.tool_history.append(tool_message)
            except Exception as e:
                logger.error(f"Error executing tool {function_name}: {e}")
                raise AgentError(f"Error executing tool '{function_name}': {e}") from e
    
    async def process_iterations(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Iteratively drives the agent conversation until a final answer or max iterations.

        Args:
            messages (List[Dict[str, Any]]): Initial conversation messages.

        Returns:
            Any: The final assistant message.

        Raises:
            AgentError: On chat failure or tool issues.
        """
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations} started.")

            messages += self.tool_history

            try:
                response: ChatCompletion = self.llm.generate(
                    messages=messages,
                    tools=self.tools,
                    tool_choice=self.tool_choice,
                )
                response_message = response.get_message()
                self.text_formatter.print_message(response_message)

                if response.get_reason() == "tool_calls":
                    self.tool_history.append(response_message)
                    await self.process_response(response.get_tool_calls())
                else:
                    self.memory.add_message(AssistantMessage(response.get_content()))
                    self.tool_history.clear()
                    return response.get_content()
            except Exception as e:
                logger.error(f"Error during chat generation: {e}")
                raise AgentError(f"Failed during chat generation: {e}") from e

        logger.info("Max iterations reached. Agent has stopped.")
    
    async def run_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Executes a registered tool by name, automatically handling sync or async tools.

        Args:
            tool_name (str): Name of the tool to run.
            *args: Positional arguments passed to the tool.
            **kwargs: Keyword arguments passed to the tool.

        Returns:
            Any: Result from the tool execution.

        Raises:
            AgentError: If the tool is not found or execution fails.
        """
        try:
            return await self.tool_executor.run_tool(tool_name, *args, **kwargs)
        except Exception as e:
            logger.error(f"Agent failed to run tool '{tool_name}': {e}")
            raise AgentError(f"Failed to run tool '{tool_name}': {e}") from e