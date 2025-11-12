from __future__ import annotations

import asyncio
import json
import logging
import signal
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Sequence, Union

from dapr_agents.agents.base import AgentBase
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentProfileConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    AgentExecutionConfig,
)
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.types import (
    AgentError,
    AssistantMessage,
    LLMChatResponse,
    ToolCall,
    ToolExecutionRecord,
    ToolMessage,
)
from dapr_agents.types.workflow import DaprWorkflowStatus

logger = logging.getLogger(__name__)


class Agent(AgentBase):
    """
    Standalone (non-workflow) agent built on AgentBase.

    Overview:
        Reuses AgentBase for profile/prompting, LLM wiring, memory, and durable state.
        Runs an in-process conversation loop (no Dapr Workflows) while persisting each
        run using the standard AgentWorkflowState schema that AgentBase wires in.
    """

    def __init__(
        self,
        *,
        # Profile / prompt
        profile: Optional[AgentProfileConfig] = None,
        name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        instructions: Optional[Iterable[str]] = None,
        style_guidelines: Optional[Iterable[str]] = None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[PromptTemplateBase] = None,
        # Runtime
        llm: Optional[ChatClientBase] = None,
        tools: Optional[Iterable[Any]] = None,
        memory: Optional[AgentMemoryConfig] = None,
        # Persistence/registry
        state: Optional[AgentStateConfig] = None,
        registry: Optional[AgentRegistryConfig] = None,
        # Behavior / execution
        execution: Optional[AgentExecutionConfig] = None,
        # Misc
        agent_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize behavior + infrastructure for a non-workflow agent.

        Args:
            profile: High-level profile (can be overridden by explicit fields).
            name, role, goal, instructions, style_guidelines, system_prompt: Prompt/profile fields.
            prompt_template: Optional explicit prompt template instance.
            llm: Chat client; defaults to `get_default_llm()`.
            tools: Optional tool callables or `AgentTool` instances.
            memory: Conversation memory config.
            state: Durable state configuration (store/key + optional hooks).
            registry: Team registry configuration.
            execution: Execution dials for the agent run.
            agent_metadata: Extra metadata to store in the registry.
        """
        super().__init__(
            profile=profile,
            name=name,
            role=role,
            goal=goal,
            instructions=instructions,
            style_guidelines=style_guidelines,
            system_prompt=system_prompt,
            state=state,
            memory=memory,
            registry=registry,
            execution=execution,
            agent_metadata=agent_metadata,
            llm=llm,
            tools=tools,
            prompt_template=prompt_template,
        )

        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()

        try:
            self.load_state()
        except Exception:
            logger.debug(
                "Standalone agent state load failed; using defaults.", exc_info=True
            )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    async def run(
        self,
        input_data: Optional[Union[str, Dict[str, Any]]] = None,
        *,
        instance_id: Optional[str] = None,
    ) -> Optional[AssistantMessage]:
        """
        Execute a conversational run in-process.

        Args:
            input_data: Optional user input (string or structured dict).
            instance_id: Optional workflow-like instance id; auto-generated if omitted.

        Returns:
            The final assistant message (if not cancelled), else None.

        Raises:
            AgentError: Propagates structured errors from generation/tooling.
        """
        try:
            return await self._race(
                self._run_agent(input_data=input_data, instance_id=instance_id)
            )
        except asyncio.CancelledError:
            logger.info("Standalone agent run was cancelled.")
            return None
        except Exception as exc:
            logger.exception("Standalone agent run failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Internal runtime loop
    # ------------------------------------------------------------------
    async def _race(
        self, coro: Awaitable[Optional[AssistantMessage]]
    ) -> Optional[AssistantMessage]:
        """Race the agent execution against shutdown signals."""
        task = asyncio.create_task(coro)
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        done, pending = await asyncio.wait(
            [task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for pending_task in pending:
            pending_task.cancel()
        if self._shutdown_event.is_set():
            logger.info(
                "Shutdown requested during execution; cancelling standalone run."
            )
            task.cancel()
            return None
        return await task

    async def _run_agent(
        self,
        *,
        input_data: Optional[Union[str, Dict[str, Any]]],
        instance_id: Optional[str],
    ) -> Optional[AssistantMessage]:
        """One-shot conversational run with tool loop and durable timeline."""
        self.load_state()
        active_instance = instance_id or self._generate_instance_id()

        # Build initial messages with persistent + per-instance history
        chat_history = self._reconstruct_conversation_history(active_instance)
        messages = self.prompting_helper.build_initial_messages(
            user_input=input_data,
            chat_history=chat_history,
        )

        # Keep per-instance system messages in sync with state
        self._sync_system_messages_with_state(active_instance, messages)

        # Print + capture the user's message if present
        user_message = self._get_last_user_message(messages)
        user_message_copy = dict(user_message) if user_message else None
        task_text = user_message_copy.get("content") if user_message_copy else None

        if user_message_copy is not None:
            self.text_formatter.print_message(
                {str(k): v for k, v in user_message_copy.items()}
            )

        # Ensure instance exists (flexible model via _get_entry_container)
        created_instance = active_instance not in (
            getattr(self.workflow_state, "instances", {}) or {}
        )
        self.ensure_instance_exists(
            instance_id=active_instance,
            input_value=task_text or "Triggered without input.",
            triggering_workflow_instance_id=None,
            time=datetime.now(timezone.utc),
        )
        if created_instance:
            self.save_state()

        # Persist the user message into timeline + memory
        self._process_user_message(active_instance, task_text, user_message_copy)

        # Enter the tool/LLM loop
        final_reply = await self._conversation_loop(
            instance_id=active_instance,
            messages=messages,
        )

        return final_reply

    def construct_messages(
        self,
        input_data: Optional[Union[str, Dict[str, Any]]] = None,
        *,
        instance_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build the outbound message list (without running the loop).

        Args:
            input_data: Optional user input.
            instance_id: Optional instance id to load per-instance history.

        Returns:
            List of message dicts suitable for an LLM chat API.
        """
        self.load_state()
        active_instance = instance_id or self._generate_instance_id()
        chat_history = self._reconstruct_conversation_history(active_instance)
        return self.prompting_helper.build_initial_messages(
            user_input=input_data,
            chat_history=chat_history,
        )

    async def _conversation_loop(
        self,
        *,
        instance_id: str,
        messages: List[Dict[str, Any]],
    ) -> Optional[AssistantMessage]:
        """
        Core in-process loop that alternates LLM responses and tool calls.

        Args:
            instance_id: Per-run id used to persist timeline to state.
            messages: Initial outbound messages (system + history + user).

        Returns:
            Final assistant message, or None if cancelled mid-run.

        Raises:
            AgentError: If chat generation fails or no assistant message is produced.
        """
        pending_messages = list(messages)
        final_reply: Optional[AssistantMessage] = None

        last_assistant_dict: Dict[str, Any] | None = None

        try:
            for turn in range(1, self.execution.max_iterations + 1):
                logger.info(
                    "Iteration %d/%d started.", turn, self.execution.max_iterations
                )
                response: LLMChatResponse = self.llm.generate(
                    messages=pending_messages,
                    tools=self.get_llm_tools(),
                    **(
                        {"tool_choice": self.execution.tool_choice}
                        if self.execution.tool_choice is not None
                        else {}
                    ),
                )
                assistant_message = response.get_message()
                if assistant_message is None:
                    raise AgentError("LLM returned no assistant message.")

                assistant_dict = assistant_message.model_dump()
                last_assistant_dict = assistant_dict
                self._save_assistant_message(instance_id, assistant_dict)
                self.text_formatter.print_message(assistant_dict)

                if assistant_message.has_tool_calls():
                    tool_calls = assistant_message.get_tool_calls()
                    if tool_calls:
                        pending_messages.append(assistant_dict)
                        tool_msgs = await self._execute_tool_calls(
                            instance_id, tool_calls
                        )
                        pending_messages.extend(tool_msgs)
                        continue

                final_reply = assistant_message
                break
            else:
                # Max iterations reached without a final reply; append a notice.
                if last_assistant_dict:
                    content = last_assistant_dict.get("content") or ""
                    if content:
                        content = content.rstrip() + "\n\n"
                    content += (
                        "I reached the maximum number of reasoning steps before I could finish. "
                        "Please rephrase or provide more detail so I can try again."
                    )
                    last_assistant_dict["content"] = content
                    final_reply = AssistantMessage(**last_assistant_dict)
                else:
                    final_reply = AssistantMessage(
                        role="assistant",
                        content=(
                            "I reached the maximum number of reasoning steps before I could finish. "
                            "Please rephrase or provide more detail so I can try again."
                        ),
                    )
                message_dict = final_reply.model_dump()
                self._save_assistant_message(instance_id, message_dict)
                self.text_formatter.print_message(message_dict)
                logger.warning(
                    "Standalone agent hit max iterations (%d) without a final response.",
                    self.execution.max_iterations,
                )
        except Exception as exc:
            logger.error("Error during conversation loop: %s", exc)
            final_reply = AssistantMessage(
                role="assistant",
                content=f"Error: {exc}",
            )
            message_dict = final_reply.model_dump()
            self._save_assistant_message(instance_id, message_dict)
            self.text_formatter.print_message(message_dict)
            raise AgentError(f"Failed during chat generation: {exc}") from exc
        finally:
            self._update_instance_completion(instance_id, final_reply)

        return final_reply

    async def _execute_tool_calls(
        self,
        instance_id: str,
        tool_calls: Sequence[ToolCall],
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls concurrently with bounded parallelism.

        Args:
            instance_id: Timeline instance id to append tool results to.
            tool_calls: ToolCall objects from the assistant.

        Returns:
            List of tool message dicts to append to the LLM turn.
        """
        max_concurrent = 10
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_single(tool_call: ToolCall) -> Dict[str, Any]:
            async with semaphore:
                return await self._run_tool_call(instance_id, tool_call)

        return await asyncio.gather(*(run_single(call) for call in tool_calls))

    async def _run_tool_call(
        self, instance_id: str, tool_call: ToolCall
    ) -> Dict[str, Any]:
        """
        Execute one tool call and persist outcome to state + memory.

        Args:
            instance_id: Timeline instance id to append messages/history.
            tool_call: ToolCall describing the function and arguments.

        Returns:
            A tool message dict ({role:"tool", name, content, id})

        Raises:
            AgentError: On invalid call or tool execution failure.
        """
        function_name = tool_call.function.name
        if not function_name:
            error_msg = f"Tool call missing function name: {tool_call}"
            logger.error(error_msg)
            raise AgentError(error_msg)

        function_args = tool_call.function.arguments_dict
        try:
            result = await self.tool_executor.run_tool(function_name, **function_args)
        except Exception as exc:
            logger.error("Error executing tool %s: %s", function_name, exc)
            raise AgentError(f"Error executing tool '{function_name}': {exc}") from exc

        # Safe serialization of tool result
        if isinstance(result, str):
            serialized_result = result
        else:
            try:
                serialized_result = json.dumps(result)
            except Exception:  # noqa: BLE001
                serialized_result = str(result)

        # Build memory + durable messages
        tool_message = ToolMessage(
            tool_call_id=tool_call.id,
            name=function_name,
            content=serialized_result,
        )
        message_dict = tool_message.model_dump()

        history_entry = ToolExecutionRecord(
            tool_call_id=tool_call.id,
            tool_name=function_name,
            tool_args=function_args,
            execution_result=serialized_result,
        )

        # Append to durable timeline using the flexible model/coercer path
        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None
        if entry is not None and hasattr(entry, "messages"):
            # Prefer a custom coercer if configured; otherwise the configured message model, with a safe fallback.
            try:
                if getattr(self, "_message_coercer", None):
                    durable_message = self._message_coercer(message_dict)  # type: ignore[attr-defined]
                else:
                    durable_message = self._message_dict_to_message_model(message_dict)
            except Exception:
                # Last-resort: keep the raw dict so we don't drop tool output.
                durable_message = dict(message_dict)

            entry.messages.append(durable_message)
            if hasattr(entry, "tool_history"):
                entry.tool_history.append(history_entry)
            if hasattr(entry, "last_message"):
                entry.last_message = durable_message

        # Always persist to memory + in-process history
        self.text_formatter.print_message(message_dict)
        self.memory.add_message(tool_message)
        self.tool_history.append(history_entry)
        self.save_state()

        # Return tool message dict so the next LLM turn can see it
        return message_dict

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def _update_instance_completion(
        self,
        instance_id: str,
        final_reply: Optional[AssistantMessage],
    ) -> None:
        """
        Mark an instance as completed/failed with end time and output.

        Args:
            instance_id: Timeline instance id.
            final_reply: AssistantMessage (if any) that ended the loop.
        """
        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None
        if entry is None:
            return

        entry.status = (
            DaprWorkflowStatus.COMPLETED.value
            if final_reply
            else DaprWorkflowStatus.FAILED.value
        )
        if final_reply and hasattr(entry, "output"):
            entry.output = final_reply.content or ""
        entry.end_time = datetime.now(timezone.utc)
        self.save_state()

    def _generate_instance_id(self) -> str:
        """Generate a unique instance id for standalone runs."""
        return f"{self.name}-{uuid.uuid4().hex}"

    # ------------------------------------------------------------------
    # Infrastructure hooks (signals)
    # ------------------------------------------------------------------
    def _setup_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers to allow graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (OSError, ValueError):
            # Not available in some environments (e.g., Windows/subthreads)
            pass

    def _signal_handler(
        self, signum, frame
    ) -> None:  # pragma: no cover - signal handler
        """Signal handler that asks the run loop to stop."""
        logger.info("Received signal %s. Shutting down gracefully...", signum)
        self._shutdown_event.set()
