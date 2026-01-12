from __future__ import annotations

from datetime import timedelta
import json
import logging
from typing import Any, Dict, Iterable, List, Optional
from os import getenv

import dapr.ext.workflow as wf

from dapr_agents.agents.base import AgentBase
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    WorkflowGrpcOptions,
    WorkflowRetryPolicy,
    AgentObservabilityConfig,
)
from dapr_agents.agents.prompting import AgentProfileConfig
from dapr_agents.agents.schemas import (
    AgentTaskResponse,
    BroadcastMessage,
    TriggerAction,
)
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.types import (
    AgentError,
    LLMChatResponse,
    ToolMessage,
    UserMessage,
)
from dapr_agents.types.workflow import DaprWorkflowStatus
from dapr_agents.tool.utils.serialization import serialize_tool_result
from dapr_agents.workflow.decorators import message_router, workflow_entry
from dapr_agents.workflow.utils.grpc import apply_grpc_options
from dapr_agents.workflow.utils.pubsub import broadcast_message, send_message_to_agent

logger = logging.getLogger(__name__)


class DurableAgent(AgentBase):
    """
    Workflow-native durable agent runtime on top of AgentBase.

    Overview:
        Wires your AgentBase behavior into Dapr Workflows for durable, pub/sub-driven runs.
        Persists state using the built-in AgentWorkflowState schema while still honoring
        safe hook overrides (entry_factory, message_coercer, etc.).

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
        # Infrastructure
        pubsub: Optional[AgentPubSubConfig] = None,
        state: Optional[AgentStateConfig] = None,
        registry: Optional[AgentRegistryConfig] = None,
        # Memory / runtime
        memory: Optional[AgentMemoryConfig] = None,
        llm: Optional[ChatClientBase] = None,
        tools: Optional[Iterable[Any]] = None,
        # Behavior / execution
        execution: Optional[AgentExecutionConfig] = None,
        # Misc
        agent_metadata: Optional[Dict[str, Any]] = None,
        workflow_grpc: Optional[WorkflowGrpcOptions] = None,
        runtime: Optional[wf.WorkflowRuntime] = None,
        retry_policy: WorkflowRetryPolicy = WorkflowRetryPolicy(),
        agent_observability: Optional[AgentObservabilityConfig] = None,
    ) -> None:
        """
        Initialize behavior, infrastructure, and workflow runtime.

        Args:
            profile: High-level profile (can be overridden by explicit fields).
            name: Agent name (required if not in `profile`).
            role: Agent role/persona label.
            goal: High-level objective for prompting context.
            instructions: Extra instruction lines for the system prompt.
            style_guidelines: Style directives for the system prompt.
            system_prompt: System prompt override.
            prompt_template: Optional explicit prompt template instance.

            pubsub: Optional Dapr Pub/Sub configuration for triggers/broadcasts.
                If omitted, the agent won't subscribe to any topics and can only be
                triggered directly via AgentRunner.
            state: Durable state configuration (store/key + optional hooks).
            registry: Team registry configuration.
            execution: Execution dials for the agent run.

            memory: Conversation memory config; defaults to in-memory, or Dapr state-backed if available.
            llm: Chat client; defaults to `get_default_llm()`.
            tools: Optional tool callables or `AgentTool` instances.

            agent_metadata: Extra metadata to publish to the registry.
            workflow_grpc: Optional gRPC overrides for the workflow runtime channel.
            runtime: Optional pre-existing workflow runtime to attach to.
            retry_policy: Durable retry policy configuration.
            agent_observability: Observability configuration for tracing/logging.
        """
        super().__init__(
            pubsub=pubsub,
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
            workflow_grpc=workflow_grpc,
            llm=llm,
            tools=tools,
            prompt_template=prompt_template,
            agent_observability=agent_observability,
        )

        apply_grpc_options(self.workflow_grpc_options)

        self._runtime: wf.WorkflowRuntime = runtime or wf.WorkflowRuntime()
        self._runtime_owned = runtime is None
        self._registered = False
        self._started = False

        try:
            retries = int(getenv("DAPR_API_MAX_RETRIES", ""))
        except ValueError:
            retries = retry_policy.max_attempts

        if retries < 1:
            raise (
                ValueError("max_attempts or DAPR_API_MAX_RETRIES must be at least 1.")
            )

        self._retry_policy: wf.RetryPolicy = wf.RetryPolicy(
            max_number_of_attempts=retries,
            first_retry_interval=timedelta(
                seconds=retry_policy.initial_backoff_seconds
            ),
            max_retry_interval=timedelta(seconds=retry_policy.max_backoff_seconds),
            backoff_coefficient=retry_policy.backoff_multiplier,
            retry_timeout=timedelta(seconds=retry_policy.retry_timeout)
            if retry_policy.retry_timeout
            else None,
        )

    # ------------------------------------------------------------------
    # Runtime accessors
    # ------------------------------------------------------------------
    @property
    def runtime(self) -> wf.WorkflowRuntime:
        """Return the underlying workflow runtime."""
        return self._runtime

    @property
    def is_started(self) -> bool:
        """Return True when the workflow runtime has been started."""
        return self._started

    # ------------------------------------------------------------------
    # Workflows / Activities
    # ------------------------------------------------------------------
    @workflow_entry
    @message_router(message_model=TriggerAction)
    def agent_workflow(self, ctx: wf.DaprWorkflowContext, message: dict):
        """
        Primary workflow loop reacting to `TriggerAction` pub/sub messages.

        Args:
            ctx: Dapr workflow context injected by the runtime.
            message: Trigger payload; may include task string and metadata.

        Returns:
            Final assistant message as a dict.

        Raises:
            AgentError: If the loop finishes without producing a final response.
        """
        task = message.get("task")
        metadata = message.get("_message_metadata", {}) or {}

        # Propagate OTel/parent workflow relations if present.
        otel_span_context = message.get("_otel_span_context")
        if "workflow_instance_id" in message:
            metadata["triggering_workflow_instance_id"] = message[
                "workflow_instance_id"
            ]

        trigger_instance_id = metadata.get("triggering_workflow_instance_id")
        source = metadata.get("source") or "direct"

        # Ensure we have the latest durable state for this turn.
        if self.state_store:
            self.load_state()

        # Bootstrap instance entry (flexible to non-`instances` models).
        self.ensure_instance_exists(
            instance_id=ctx.instance_id,
            input_value=task or "Triggered without input.",
            triggering_workflow_instance_id=trigger_instance_id,
            time=ctx.current_utc_datetime,
        )

        if not ctx.is_replaying:
            logger.info("Initial message from %s -> %s", source, self.name)

        # Record initial entry via activity to keep deterministic/replay-friendly I/O.
        yield ctx.call_activity(
            self.record_initial_entry,
            input={
                "instance_id": ctx.instance_id,
                "input_value": task or "Triggered without input.",
                "source": source,
                "triggering_workflow_instance_id": trigger_instance_id,
                "start_time": ctx.current_utc_datetime.isoformat(),
                "trace_context": otel_span_context,
            },
            retry_policy=self._retry_policy,
        )

        final_message: Dict[str, Any] = {}
        turn = 0

        try:
            for turn in range(1, self.execution.max_iterations + 1):
                if not ctx.is_replaying:
                    logger.debug(
                        "Agent %s turn %d/%d (instance=%s)",
                        self.name,
                        turn,
                        self.execution.max_iterations,
                        ctx.instance_id,
                    )

                assistant_response: Dict[str, Any] = yield ctx.call_activity(
                    self.call_llm,
                    input={
                        "task": task,
                        "instance_id": ctx.instance_id,
                        "time": ctx.current_utc_datetime.isoformat(),
                    },
                    retry_policy=self._retry_policy,
                )

                tool_calls = assistant_response.get("tool_calls") or []
                if tool_calls:
                    if not ctx.is_replaying:
                        logger.debug(
                            "Agent %s executing %d tool call(s) on turn %d",
                            self.name,
                            len(tool_calls),
                            turn,
                        )
                    parallel = [
                        ctx.call_activity(
                            self.run_tool,
                            input={
                                "tool_call": tc,
                                "instance_id": ctx.instance_id,
                                "time": ctx.current_utc_datetime.isoformat(),
                                "order": idx,
                            },
                            retry_policy=self._retry_policy,
                        )
                        for idx, tc in enumerate(tool_calls)
                    ]
                    tool_results: List[Dict[str, Any]] = yield wf.when_all(parallel)

                    yield ctx.call_activity(
                        self.save_tool_results,
                        input={
                            "tool_results": tool_results,
                            "instance_id": ctx.instance_id,
                        },
                        retry_policy=self._retry_policy,
                    )

                    task = None  # prepare for next turn
                    continue

                final_message = assistant_response
                if not ctx.is_replaying:
                    logger.debug(
                        "Agent %s produced final response on turn %d (instance=%s)",
                        self.name,
                        turn,
                        ctx.instance_id,
                    )
                break
            else:
                # Loop exhausted without a terminating reply → surface a friendly notice.
                base = final_message.get("content") or ""
                if base:
                    base = base.rstrip() + "\n\n"
                base += (
                    "I reached the maximum number of reasoning steps before I could finish. "
                    "Please rephrase or provide more detail so I can try again."
                )
                final_message = {"role": "assistant", "content": base}
                if not ctx.is_replaying:
                    logger.warning(
                        "Agent %s hit max iterations (%d) without a final response (instance=%s)",
                        self.name,
                        self.execution.max_iterations,
                        ctx.instance_id,
                    )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent %s workflow failed: %s", self.name, exc)
            final_message = {"role": "assistant", "content": f"Error: {str(exc)}"}

        # Optionally broadcast the final message to the team.
        if self.broadcast_topic_name:
            yield ctx.call_activity(
                self.broadcast_message_to_agents,
                input={"message": final_message},
                retry_policy=self._retry_policy,
            )

        # Optionally send a direct response back to the trigger origin.
        if source and trigger_instance_id:
            yield ctx.call_activity(
                self.send_response_back,
                input={
                    "response": final_message,
                    "target_agent": source,
                    "target_instance_id": trigger_instance_id,
                },
                retry_policy=self._retry_policy,
            )

        # Finalize the workflow entry in durable state.
        yield ctx.call_activity(
            self.finalize_workflow,
            input={
                "instance_id": ctx.instance_id,
                "final_output": final_message.get("content", ""),
                "end_time": ctx.current_utc_datetime.isoformat(),
                "triggering_workflow_instance_id": trigger_instance_id,
            },
            retry_policy=self._retry_policy,
        )

        if not ctx.is_replaying:
            verdict = (
                "max_iterations_reached"
                if turn == self.execution.max_iterations
                else "completed"
            )
            logger.info(
                "Workflow %s finalized for agent %s with verdict=%s",
                ctx.instance_id,
                self.name,
                verdict,
            )

        return final_message

    @message_router(message_model=BroadcastMessage, broadcast=True)
    def broadcast_listener(self, ctx: wf.DaprWorkflowContext, message: dict) -> None:
        """
        Handle broadcast messages sent by other agents and store them in memory.

        Args:
            ctx: Dapr workflow context (unused).
            message: Broadcast payload containing content and metadata.
        """
        metadata = message.get("_message_metadata", {}) or {}
        source = metadata.get("source") or "unknown"
        message_content = message.get("content", "")
        if source == self.name:
            logger.debug("Agent %s ignoring self-originated broadcast.", self.name)
            return

        logger.info("Agent %s received broadcast from %s", self.name, source)
        logger.debug("Full broadcast message: %s", message)
        # Store as a user message from the broadcasting agent (kept in persistent memory).
        self.memory.add_message(
            UserMessage(name=source, content=message_content, role="user")
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reconstruct_conversation_history(
        self, instance_id: str
    ) -> List[Dict[str, Any]]:
        """
        Build conversation history from per-instance state.

        Args:
            instance_id: Workflow instance identifier.

        Returns:
            Message history for this specific workflow instance.
        """
        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None

        instance_messages: List[Dict[str, Any]] = []
        if entry and hasattr(entry, "messages"):
            for msg in getattr(entry, "messages"):
                serialized = self._serialize_message(msg)
                if serialized.get("role") != "system":
                    instance_messages.append(serialized)

        if instance_messages:
            return instance_messages

        persistent_memory: List[Dict[str, Any]] = []
        try:
            for msg in self.memory.get_messages():
                try:
                    persistent_memory.append(self._serialize_message(msg))
                except TypeError:
                    logger.debug(
                        "Unsupported memory message type %s; skipping.", type(msg)
                    )
        except Exception:
            logger.debug("Unable to load persistent memory.", exc_info=True)

        return persistent_memory

    # ------------------------------------------------------------------
    # Activities
    # ------------------------------------------------------------------
    def record_initial_entry(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Record the initial entry for a workflow instance.

        Args:
            payload: Keys:
                - instance_id: Workflow instance id.
                - input_value: Initial input value.
                - source: Trigger source string.
                - triggering_workflow_instance_id: Optional parent workflow id.
                - start_time: ISO8601 datetime string.
                - trace_context: Optional tracing context.
        """
        # Load latest state to ensure we have current data before modifying
        if self.state_store:
            self.load_state()

        instance_id = payload.get("instance_id")
        trace_context = payload.get("trace_context")
        input_value = payload.get("input_value", "Triggered without input.")
        source = payload.get("source", "direct")
        triggering_instance = payload.get("triggering_workflow_instance_id")
        start_time = self._coerce_datetime(payload.get("start_time"))

        # Ensure instance exists in durable state
        self.ensure_instance_exists(
            instance_id=instance_id,
            input_value=input_value,
            triggering_workflow_instance_id=triggering_instance,
            time=start_time,
        )

        # Use flexible container accessor (supports custom state layouts)
        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None
        if entry is None:
            return

        entry.input_value = input_value
        entry.source = source
        entry.triggering_workflow_instance_id = triggering_instance
        entry.start_time = start_time
        entry.trace_context = trace_context

        session_id = getattr(self.memory, "session_id", None)
        if session_id is not None and hasattr(entry, "session_id"):
            entry.session_id = str(session_id)

        entry.status = DaprWorkflowStatus.RUNNING.value
        self.save_state()

    def call_llm(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ask the LLM to generate the next assistant message.

        Args:
            payload: Must contain 'instance_id'; may include 'task' and 'time'.

        Returns:
            Assistant message as a dict.

        Raises:
            AgentError: If the LLM call fails or yields no message.
        """
        # Load latest state to ensure we have current data
        if self.state_store:
            self.load_state()

        instance_id = payload.get("instance_id")
        task = payload.get("task")

        chat_history = self._reconstruct_conversation_history(instance_id)
        messages = self.prompting_helper.build_initial_messages(
            user_input=task,
            chat_history=chat_history,
        )

        # Sync current system messages into per-instance state
        self._sync_system_messages_with_state(instance_id, messages)

        # Persist the user's turn (if any) into the instance timeline + memory
        # Only process and print user message if task is provided (initial turn)
        if task:
            user_message = self._get_last_user_message(messages)
            user_copy = dict(user_message) if user_message else None
            self._process_user_message(instance_id, task, user_copy)

            if user_copy is not None:
                self.text_formatter.print_message(
                    {str(k): v for k, v in user_copy.items()}
                )

        tools = self.get_llm_tools()
        generate_kwargs = {
            "messages": messages,
            "tools": tools,
        }
        if tools and self.execution.tool_choice is not None:
            generate_kwargs["tool_choice"] = self.execution.tool_choice

        try:
            response: LLMChatResponse = self.llm.generate(**generate_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM generate failed: %s", exc)
            raise AgentError(str(exc)) from exc

        assistant_message = response.get_message()
        if assistant_message is None:
            raise AgentError("LLM returned no assistant message.")

        as_dict = assistant_message.model_dump()
        self._save_assistant_message(instance_id, as_dict)
        self.text_formatter.print_message(as_dict)
        self.save_state()
        return as_dict

    def run_tool(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single tool call.

        Args:
            payload: Keys 'tool_call', 'instance_id', 'time', 'order'.

        Returns:
            ToolMessage as a dict.

        Raises:
            AgentError: If tool arguments contain invalid JSON.
        """
        tool_call = payload.get("tool_call", {})
        fn_name = tool_call["function"]["name"]
        raw_args = tool_call["function"].get("arguments", "")

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            raise AgentError(f"Invalid JSON in tool args: {exc}") from exc

        async def _execute_tool() -> Any:
            return await self.tool_executor.run_tool(fn_name, **args)

        result = self._run_asyncio_task(_execute_tool())

        # Debug: Log the actual result before serialization
        logger.debug(f"Tool {fn_name} returned: {result} (type: {type(result)})")

        # Serialize the tool result using centralized utility
        serialized_result = serialize_tool_result(result)

        tool_result = ToolMessage(
            content=serialized_result,
            role="tool",
            name=fn_name,
            tool_call_id=tool_call["id"],
        )

        # Print the tool result for visibility
        self.text_formatter.print_message(tool_result)

        return tool_result.model_dump()

    def save_tool_results(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Save tool results to memory in the correct order.

        This activity is called after all parallel tool executions complete.
        It writes all tool results to memory sequentially, ensuring correct
        ordering for OpenAI API compliance.

        Args:
            payload: Keys 'tool_results' (list of tool result dicts) and 'instance_id'.
        """
        if self.state_store:
            self.load_state()

        tool_results_raw: List[Dict[str, Any]] = payload.get("tool_results", [])
        tool_results: List[ToolMessage] = [ToolMessage(**tr) for tr in tool_results_raw]
        instance_id: str = payload.get("instance_id", "")

        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None

        existing_tool_ids: set[str] = set()
        if entry is not None and hasattr(entry, "messages"):
            for msg in getattr(entry, "messages"):
                try:
                    tid = getattr(msg, "tool_call_id", None)
                    if tid:
                        existing_tool_ids.add(tid)
                except Exception:
                    pass

        for tool_result in tool_results:
            tool_call_id = tool_result.tool_call_id

            if tool_call_id in existing_tool_ids:
                logger.debug(f"Tool result {tool_call_id} already in entry, skipping")
                continue

            if entry is not None and hasattr(entry, "messages"):
                tool_message_model = (
                    self._message_coercer(tool_result.model_dump())
                    if getattr(self, "_message_coercer", None)
                    else self._message_dict_to_message_model(tool_result.model_dump())
                )
                entry.messages.append(tool_message_model)
                if hasattr(entry, "last_message"):
                    entry.last_message = tool_message_model

            self.memory.add_message(tool_result)
            logger.debug(f"Added tool result {tool_call_id} to memory")

        self.save_state()

    def broadcast_message_to_agents(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Broadcast a message to all agents via pub/sub (if a broadcast topic is set).

        Args:
            payload: Dict containing 'message' (assistant/user-like dict).
        """
        message = payload.get("message", {})
        if not isinstance(message, dict) or not self.broadcast_topic_name:
            logger.debug(
                "Skipping broadcast because payload is invalid or topic is unset."
            )
            return

        try:
            agents_metadata = self.get_agents_metadata(
                exclude_self=False, exclude_orchestrator=False
            )
        except Exception:  # noqa: BLE001
            logger.exception("Unable to load agents metadata; broadcast aborted.")
            return

        message["role"] = "user"
        message["name"] = self.name
        response_message = BroadcastMessage(**message)

        async def _broadcast() -> None:
            await broadcast_message(
                message=response_message,
                broadcast_topic=self.broadcast_topic_name,
                message_bus=self.message_bus_name,
                source=self.name,
                agents_metadata=agents_metadata,
            )

        try:
            self._run_asyncio_task(_broadcast())
        except Exception:  # noqa: BLE001
            logger.exception("Failed to publish broadcast message.")

    def send_response_back(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Send the final response back to the triggering agent.

        Args:
            payload: Dict containing 'response', 'target_agent', 'target_instance_id'.
        """
        response = payload.get("response", {})
        target_agent = payload.get("target_agent", "")
        target_instance_id = payload.get("target_instance_id", "")
        if not target_agent or not target_instance_id:
            logger.debug(
                "Target agent or instance missing; skipping response publication."
            )
            return

        response["role"] = "user"
        response["name"] = self.name
        response["workflow_instance_id"] = target_instance_id
        agent_response = AgentTaskResponse(**response)

        agents_metadata = self.get_agents_metadata()

        try:
            self._run_asyncio_task(
                send_message_to_agent(
                    source=self.name,
                    target_agent=target_agent,
                    message=agent_response,
                    agents_metadata=agents_metadata,
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to publish response to %s", target_agent)

    def finalize_workflow(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Finalize a workflow instance by setting status, output, and end time.

        Args:
            payload: Dict with 'instance_id', 'final_output', 'end_time',
                     and optional 'triggering_workflow_instance_id'.
        """
        # Load latest state to ensure we have current data before modifying
        if self.state_store:
            self.load_state()

        instance_id = payload.get("instance_id")
        final_output = payload.get("final_output", "")
        end_time = payload.get("end_time", "")
        triggering_workflow_instance_id = payload.get("triggering_workflow_instance_id")

        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None
        if not entry:
            return

        entry.status = (
            DaprWorkflowStatus.COMPLETED.value
            if final_output
            else DaprWorkflowStatus.FAILED.value
        )
        entry.end_time = self._coerce_datetime(end_time)
        if hasattr(entry, "output"):
            entry.output = final_output or ""
        entry.triggering_workflow_instance_id = triggering_workflow_instance_id
        self.save_state()

    # ------------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------------
    def start(
        self,
        runtime: Optional[wf.WorkflowRuntime] = None,
        *,
        auto_register: bool = True,
    ) -> None:
        """
        Start the workflow runtime and register this agent's components.

        Behavior:
        • If a runtime is provided, attach to it (we still consider it not owned).
        • Register workflows once (if not already).
        • Always attempt to start the runtime; treat start() as idempotent:
            - If it's already running, swallow/log the exception and continue.
        • We only call shutdown() later if we own the runtime.
        """
        if self._started:
            raise RuntimeError("Agent has already been started.")

        if runtime is not None:
            self._runtime = runtime
            self._runtime_owned = False
            self._registered = False
            logger.info(
                "Attached injected WorkflowRuntime (owned=%s).", self._runtime_owned
            )

        if auto_register and not self._registered:
            self.register_workflows(self._runtime)
            self._registered = True
            logger.info(
                "Registered workflows/activities on WorkflowRuntime for agent '%s'.",
                self.name,
            )

        # Always try to start; treat as idempotent.
        try:
            self._runtime.start()
            logger.info(
                "WorkflowRuntime started for agent '%s' (owned=%s).",
                self.name,
                self._runtime_owned,
            )
        except Exception as exc:  # noqa: BLE001
            # Most common benign case: runtime already running
            logger.warning(
                "WorkflowRuntime.start() raised for agent '%s' (likely already running): %s",
                self.name,
                exc,
                exc_info=True,
            )

        self._started = True

    def stop(self) -> None:
        """Stop the workflow runtime if it is owned by this instance."""
        if not self._started:
            return

        if self._runtime_owned:
            try:
                self._runtime.shutdown()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Error while shutting down workflow runtime", exc_info=True
                )

        self._started = False

    def register(self, runtime: wf.WorkflowRuntime) -> None:
        """
        Register workflows and activities on a provided runtime.

        Args:
            runtime: An externally-managed workflow runtime to register with.
        """
        self._runtime = runtime
        self._runtime_owned = False
        self.register_workflows(runtime)
        self._registered = True

    def register_workflows(self, runtime: wf.WorkflowRuntime) -> None:
        """
        Register workflows/activities for this agent.

        Args:
            runtime: The Dapr workflow runtime to register with.
        """
        runtime.register_workflow(self.agent_workflow)
        runtime.register_workflow(self.broadcast_listener)
        runtime.register_activity(self.record_initial_entry)
        runtime.register_activity(self.call_llm)
        runtime.register_activity(self.run_tool)
        runtime.register_activity(self.save_tool_results)
        runtime.register_activity(self.broadcast_message_to_agents)
        runtime.register_activity(self.send_response_back)
        runtime.register_activity(self.finalize_workflow)
