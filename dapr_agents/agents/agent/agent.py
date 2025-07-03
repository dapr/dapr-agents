from dapr_agents.types import AgentError, AssistantMessage, ChatCompletion, ToolMessage
from dapr_agents.agents.base import AgentBase
from typing import List, Optional, Dict, Any, Union
from pydantic import Field, ConfigDict
import logging
import asyncio
from dapr_agents.types.message import UserMessage
from dapr_agents.types.message import ToolCall

logger = logging.getLogger(__name__)


class Agent(AgentBase):
    """
    Agent that manages tool calls and conversations using a language model.
    It integrates tools and processes them based on user inputs and task orchestration.
    """

    tool_history: List[ToolMessage] = Field(
        default_factory=list, description="Executed tool calls during the conversation."
    )
    tool_choice: Optional[str] = Field(
        default=None,
        description="Strategy for selecting tools ('auto', 'required', 'none'). Defaults to 'auto' if tools are provided.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize the agent's settings, such as tool choice and parent setup.
        Sets the tool choice strategy based on provided tools.
        """
        self.tool_choice = self.tool_choice or ("auto" if self.tools else None)

        # Proceed with base model setup
        super().model_post_init(__context)

    async def run(self, input_data: Optional[Union[str, Dict[str, Any]]] = None) -> Any:
        """Run the agent with the given input with graceful shutdown support."""
        try:
            if self._shutdown_event.is_set():
                print("Shutdown requested. Skipping agent execution.")
                return None

            task = asyncio.create_task(self._run_agent(input_data))
            done, pending = await asyncio.wait(
                [task, asyncio.create_task(self._shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for p in pending:
                p.cancel()

            if self._shutdown_event.is_set():
                print("Shutdown requested during execution. Cancelling agent.")
                task.cancel()
                return None

            if task in done:
                return await task

        except asyncio.CancelledError:
            print("Agent execution was cancelled.")
            return None
        except Exception as e:
            print(f"Error during agent execution: {e}")
            raise

    async def _run_agent(
        self, input_data: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Any:
        """Internal method for running the agent logic (original ToolCallAgent run method)."""
        logger.debug(
            f"Agent run started with input: {input_data if input_data else 'Using memory context'}"
        )

        # Format messages; construct_messages already includes chat history.
        messages = self.construct_messages(input_data or {})
        user_message = self.get_last_user_message(messages)

        if input_data and user_message:
            # Add the new user message to memory only if input_data is provided and user message exists
            user_msg = UserMessage(content=user_message.get("content", ""))
            self.memory.add_message(user_msg)

        # Always print the last user message for context, even if no input_data is provided
        if user_message:
            self.text_formatter.print_message(user_message)

        # Process conversation iterations
        return await self.process_iterations(messages)

    async def process_response(self, tool_calls: List[ToolCall]) -> None:
        """
        Asynchronously executes tool calls and appends tool results to memory.

        Args:
            tool_calls (List[ToolCall]): Tool calls returned by the LLM.

        Raises:
            AgentError: If a tool execution fails.
        """
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            tool_id = tool_call.id
            function_args = (
                tool_call.function.arguments_dict
            )  # Use the property to get dict

            if not function_name:
                logger.error(f"Tool call missing function name: {tool_call}")
                continue

            try:
                logger.info(f"Executing {function_name} with arguments {function_args}")
                result = await self.tool_executor.run_tool(
                    function_name, **function_args
                )
                tool_message = ToolMessage(
                    tool_call_id=tool_id, name=function_name, content=str(result)
                )
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

            # Create a copy of messages for this iteration
            current_messages = messages.copy()

            try:
                response = self.llm.generate(
                    messages=current_messages,
                    tools=self.get_llm_tools(),
                )

                # Handle different response types
                if isinstance(response, ChatCompletion):
                    response_message = response.get_message()
                    if response_message:
                        message_dict = {
                            "role": "assistant",
                            "content": response_message,
                        }
                        self.text_formatter.print_message(message_dict)

                    if response.get_reason() == "tool_calls":
                        tool_calls = response.get_tool_calls()
                        if tool_calls:
                            # Add the assistant message with tool calls to the conversation
                            if response_message:
                                # Extract content from response_message if it's a dict
                                if isinstance(response_message, dict):
                                    content = response_message.get("content", "")
                                    if content is None:
                                        content = ""
                                    tool_calls_data = response_message.get(
                                        "tool_calls", []
                                    )
                                else:
                                    content = (
                                        str(response_message)
                                        if response_message is not None
                                        else ""
                                    )
                                    tool_calls_data = []

                                message_dict = {
                                    "role": "assistant",
                                    "content": content,
                                    "tool_calls": tool_calls_data,
                                }
                                messages.append(message_dict)

                            # Run tools and collect only the results for the current tool calls to prevent LLM errs.
                            # Context: https://github.com/dapr/dapr-agents/pull/139#discussion_r2176117456
                            tool_results = []
                            await self.process_response(tool_calls)
                            for tool_call in tool_calls:
                                # Find the corresponding ToolMessage in self.tool_history
                                tool_msg = next(
                                    (
                                        msg
                                        for msg in self.tool_history
                                        if msg.tool_call_id == tool_call.id
                                    ),
                                    None,
                                )
                                if tool_msg:
                                    tool_message_dict = {
                                        "role": "tool",
                                        "content": tool_msg.content or "",
                                        "tool_call_id": tool_msg.tool_call_id,
                                    }
                                    tool_results.append(tool_message_dict)
                            messages.extend(tool_results)

                            # Continue to next iteration to let LLM process tool results
                            continue
                    else:
                        # Final response - add to memory and return
                        content = response.get_content()
                        if content:
                            self.memory.add_message(AssistantMessage(content=content))
                        self.tool_history.clear()
                        return content
                else:
                    # Handle Dict or Iterator responses (for structured output or streaming)
                    logger.warning(
                        f"Received non-ChatCompletion response: {type(response)}"
                    )
                    if isinstance(response, dict):
                        return response.get("content", str(response))
                    else:
                        return str(response)
            except Exception as e:
                logger.error(f"Error during chat generation: {e}")
                raise AgentError(f"Failed during chat generation: {e}") from e

        logger.info("Max iterations reached. Agent has stopped.")
        return None

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
