import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from dapr_agents.agents.base import AgentBase
from dapr_agents.types import (
    AgentError,
    ToolCall,
    ToolExecutionRecord,
    ToolMessage,
    UserMessage,
    LLMChatResponse,
)

logger = logging.getLogger(__name__)


class Agent(AgentBase):
    """
    Agent that manages tool calls and conversations using a language model.
    It integrates tools and processes them based on user inputs and task orchestration.
    """

    async def run(self, input_data: Optional[Union[str, Dict[str, Any]]] = None) -> Any:
        """
        Runs the agent with the given input, supporting graceful shutdown.
        Uses the _race helper to handle shutdown and cancellation cleanly.

        Args:
            input_data (Optional[Union[str, Dict[str, Any]]]): Input for the agent, can be a string or dict.
        Returns:
            Any: The result of agent execution, or None if shutdown is requested.
        """
        try:
            return await self._race(self._run_agent(input_data))
        except asyncio.CancelledError:
            logger.info("Agent execution was cancelled.")
            return None
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            raise

    async def _race(self, coro) -> Optional[Any]:
        """
        Runs the given coroutine and races it against the agent's shutdown event.
        If shutdown is triggered, cancels the task and returns None.

        Args:
            coro: The coroutine to run (e.g., _run_agent(input_data)).
        Returns:
            Optional[Any]: The result of the coroutine, or None if shutdown is triggered.
        """
        task = asyncio.create_task(coro)
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        done, pending = await asyncio.wait(
            [task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for p in pending:
            p.cancel()
        if self._shutdown_event.is_set():
            logger.info("Shutdown requested during execution. Cancelling agent.")
            task.cancel()
            return None
        return await task

    async def _run_agent(
        self, input_data: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Any:
        """
        Internal method for running the agent logic.
        Formats messages, updates memory, and drives the conversation loop.

        Args:
            input_data (Optional[Union[str, Dict[str, Any]]]): Input for the agent, can be a string or dict.
        Returns:
            Any: The result of the agent's conversation loop.
        """
        logger.debug(
            f"Agent run started with input: {input_data if input_data else 'Using memory context'}"
        )

        # Construct messages using only input_data; chat history handled internally
        messages: List[Dict[str, Any]] = self.construct_messages(input_data or {})
        user_message = self.get_last_user_message(messages)
        # Always work with a copy of the user message for safety
        user_message_copy: Optional[Dict[str, Any]] = (
            dict(user_message) if user_message else None
        )

        if input_data and user_message_copy:
            # Add the new user message to memory only if input_data is provided and user message exists
            user_msg = UserMessage(content=user_message_copy.get("content", ""))
            self.memory.add_message(user_msg)

        # Always print the last user message for context, even if no input_data is provided
        if user_message_copy is not None:
            # Ensure keys are str for mypy
            self.text_formatter.print_message(
                {str(k): v for k, v in user_message_copy.items()}
            )

        # Process conversation iterations and return the result
        return await self.conversation(messages)

    async def execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolMessage]:
        """
        Executes a batch of tool calls in parallel, bounded by max_concurrent, using asyncio.gather.
        Each tool call is executed asynchronously using run_tool, and results are appended to the persistent audit log (tool_history).
        If any tool call fails, the error is propagated and other tasks continue unless you set return_exceptions=True.

        Args:
            tool_calls (List[ToolCall]): List of tool calls returned by the LLM to execute in this batch.
            max_concurrent (int, optional): Maximum number of concurrent tool executions (default: 5).

        Returns:
            List[ToolMessage]: Results for this batch of tool calls, in the same order as input.

        Raises:
            AgentError: If any tool execution fails.
        """
        # Limiting concurrency to avoid overwhelming downstream systems
        max_concurrent = 10
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_and_record(tool_call: ToolCall) -> ToolMessage:
            """
            Executes a single tool call, respecting the concurrency limit.
            Appends the result to the persistent audit log.
            If the function name is missing, returns a ToolMessage with error status and raises AgentError.
            """
            async with semaphore:
                function_name = tool_call.function.name
                tool_id = tool_call.id
                function_args = tool_call.function.arguments_dict

                if not function_name:
                    error_msg = f"Tool call missing function name: {tool_call}"
                    logger.error(error_msg)
                    # Return a ToolExecutionRecord with error status and raise AgentError
                    tool_execution_record = ToolExecutionRecord(
                        tool_call_id="<missing>",
                        tool_name="<missing>",
                        tool_args={},
                        execution_result=error_msg,
                    )
                    self.tool_history.append(tool_execution_record)
                    raise AgentError(error_msg)

                try:
                    logger.debug(
                        f"Executing {function_name} with arguments {function_args}"
                    )
                    result = await self.run_tool(function_name, **function_args)
                    result_str = str(result) if result is not None else ""
                    tool_message = ToolMessage(
                        tool_call_id=tool_id,
                        name=function_name,
                        content=result_str,
                    )
                    # Print the tool message for visibility
                    self.text_formatter.print_message(tool_message)
                    # Add tool message to memory
                    self.memory.add_message(tool_message)
                    # Append tool message to the persistent audit log
                    tool_execution_record = ToolExecutionRecord(
                        tool_call_id=tool_id,
                        tool_name=function_name,
                        tool_args=function_args,
                        execution_result=result_str,
                    )
                    self.tool_history.append(tool_execution_record)
                    return tool_message
                except Exception as e:
                    logger.error(f"Error executing tool {function_name}: {e}")
                    raise AgentError(
                        f"Error executing tool '{function_name}': {e}"
                    ) from e

        # Run all tool calls concurrently, but bounded by max_concurrent
        return await asyncio.gather(*(run_and_record(tc) for tc in tool_calls))

    async def conversation(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Drives the agent conversation iteratively until a final answer or max iterations is reached.
        Handles tool calls, updates memory, and returns the final assistant message.
        Tool results are localized per iteration; persistent audit log is kept for all tool executions.

        Args:
            messages (List[Dict[str, Any]]): Initial conversation messages.
        Returns:
            Any: The final assistant message or None if max iterations reached.
        Raises:
            AgentError: On chat failure or tool issues.
        """
        final_reply = None
        for turn in range(1, self.max_iterations + 1):
            logger.info(f"Iteration {turn}/{self.max_iterations} started.")
            try:
                # Generate response using the LLM
                response: LLMChatResponse = self.llm.generate(
                    messages=messages,
                    tools=self.get_llm_tools(),
                    **(
                        {"tool_choice": self.tool_choice}
                        if self.tool_choice is not None
                        else {}
                    ),
                )
                # Get the first candidate from the response
                response_message = response.get_message()
                # Check if the response contains an assistant message
                if response_message is None:
                    raise AgentError("LLM returned no assistant message")
                else:
                    assistant = response_message
                    self.text_formatter.print_message(assistant)
                    self.memory.add_message(assistant)

                # Handle tool calls response
                if assistant is not None and assistant.has_tool_calls():
                    tool_calls = assistant.get_tool_calls()
                    if tool_calls:
                        messages.append(assistant.model_dump())
                        tool_msgs = await self.execute_tools(tool_calls)
                        messages.extend([tm.model_dump() for tm in tool_msgs])
                        if turn == self.max_iterations:
                            final_reply = assistant
                            logger.info("Reached max turns after tool calls; stopping.")
                            break
                        continue

                # No tool calls => done
                final_reply = assistant
                break

            except Exception as e:
                logger.error(f"Error on turn {turn}: {e}")
                raise AgentError(f"Failed during chat generation: {e}") from e

        # Post-loop
        if final_reply is None:
            logger.warning("No reply generated; hitting max iterations.")
            return None

        logger.info(f"Agent conversation completed after {turn} turns.")
        return final_reply

    async def run_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Executes a single registered tool by name, handling both sync and async tools.
        Used for atomic tool execution, either directly or as part of a batch in execute_tools.

        Args:
            tool_name (str): Name of the tool to run.
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.

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
