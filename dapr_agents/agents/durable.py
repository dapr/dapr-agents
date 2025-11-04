from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import dapr.ext.workflow as wf
from pydantic import TypeAdapter

from dapr_agents.agents.base import AgentBase
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    WorkflowGrpcOptions,
)
from dapr_agents.agents.handoffs import (
    HandoffSpec,
    create_handoff_tool,
    generate_handoff_aliases,
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
    AgentToolExecutorError,
    LLMChatResponse,
    ToolExecutionRecord,
    ToolMessage,
    UserMessage,
)
from dapr_agents.types.workflow import DaprWorkflowStatus
from dapr_agents.workflow.decorators.routers import message_router
from dapr_agents.workflow.runners.agent import workflow_entry
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
        handoffs: Optional[Iterable[HandoffSpec]] = None,
        # Behavior / execution
        execution: Optional[AgentExecutionConfig] = None,
        # Misc
        agent_metadata: Optional[Dict[str, Any]] = None,
        workflow_grpc: Optional[WorkflowGrpcOptions] = None,
        runtime: Optional[wf.WorkflowRuntime] = None,
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
            handoffs: Optional iterable of `HandoffSpec` entries that will be converted into
                handoff tools during initialization.

            agent_metadata: Extra metadata to publish to the registry.
            workflow_grpc: Optional gRPC overrides for the workflow runtime channel.
            runtime: Optional pre-existing workflow runtime to attach to.
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
        )

        apply_grpc_options(self.workflow_grpc_options)

        self._runtime: wf.WorkflowRuntime = runtime or wf.WorkflowRuntime()
        self._runtime_owned = runtime is None
        self._registered = False
        self._started = False

        self.output_type = None
        if profile and getattr(profile, "output_type", None) is not None:
            self.output_type = profile.output_type

        # Maps handoff tool aliases to their downstream targets and spec metadata
        self._handoff_tools: Dict[str, str] = {}
        self._handoff_catalog: Dict[str, HandoffSpec] = {}

        # Ensure tools container is always a list for efficient append operations
        if not isinstance(self.tools, list):
            self.tools = list(self.tools or [])

        normalized_specs: list[HandoffSpec] = []
        if handoffs:
            for spec in handoffs:
                if not isinstance(spec, HandoffSpec):
                    raise TypeError(
                        "handoffs iterable must contain HandoffSpec instances."
                    )
                normalized_specs.append(spec)

        if normalized_specs:

            def metadata_resolver() -> Dict[str, Any]:
                return self.get_agents_metadata(
                    exclude_self=True,
                    exclude_orchestrator=True,
                )

            for spec in normalized_specs:
                self._register_handoff_tool(spec, metadata_resolver)

    # ------------------------------------------------------------------
    # Handoff tool registration
    # ------------------------------------------------------------------

    def _register_handoff_tool(
        self,
        spec: HandoffSpec,
        metadata_resolver: Callable[[], Mapping[str, Any]],
    ) -> None:
        """
        Materialize a handoff tool, register it with the executor, and track aliases.
        """
        tool = create_handoff_tool(
            target_agent_name=spec.agent_name,
            source_agent_name=self.name,
            metadata_resolver=metadata_resolver,
            tool_name=spec.tool_name,
            description=spec.description,
            input_model=spec.input_model,
        )

        try:
            self.tool_executor.register_tool(tool)
        except AgentToolExecutorError:
            logger.warning("Duplicate handoff tool skipped: %s", tool.name)
            return

        self.tools.append(tool)

        alias_keys = {
            alias.lower()
            for alias in generate_handoff_aliases(tool.name, spec.agent_name)
        }
        snapshot = replace(spec)

        for alias in alias_keys:
            existing = self._handoff_tools.get(alias)
            if existing and existing != snapshot.agent_name:
                logger.warning(
                    "Handoff alias '%s' on agent '%s' already mapped to '%s'; replacing with '%s'.",
                    alias,
                    self.name,
                    existing,
                    snapshot.agent_name,
                )
            self._handoff_tools[alias] = snapshot.agent_name
            self._handoff_catalog[alias] = snapshot

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
        expect_response = message.get("expect_response", True)

        # Propagate OTel/parent workflow relations if present.
        otel_span_context = message.get("_otel_span_context")
        if "workflow_instance_id" in message:
            metadata["triggering_workflow_instance_id"] = message[
                "workflow_instance_id"
            ]

        trigger_instance_id = metadata.get("triggering_workflow_instance_id")
        source = metadata.get("source") or "direct"

        # Track if workflow ended via handoff (to skip broadcast)
        is_handoff_exit = False

        # Ensure we have the latest durable state for this turn.
        self.load_state()

        # Bootstrap instance entry.
        self.ensure_instance_exists(
            instance_id=ctx.instance_id,
            input_value=task or "Triggered without input.",
            triggering_workflow_instance_id=trigger_instance_id,
            time=ctx.current_utc_datetime,
        )

        if not ctx.is_replaying:
            logger.info("Initial message from %s -> %s", source, self.name)

        # Record initial entry
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

                # Only include task on first turn; subsequent turns continue from history
                assistant_response: Dict[str, Any] = yield ctx.call_activity(
                    self.call_llm,
                    input={
                        "task": task if turn == 1 else None,
                        "instance_id": ctx.instance_id,
                        "time": ctx.current_utc_datetime.isoformat(),
                    },
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
                        )
                        for idx, tc in enumerate(tool_calls)
                    ]
                    tool_results = yield wf.when_all(parallel)

                    handoff_results = [
                        result
                        for result in tool_results
                        if result and result.get("is_handoff")
                    ]
                    persisted_results = [
                        result
                        for result in tool_results
                        if result and not result.get("is_handoff")
                    ]

                    if persisted_results:
                        yield ctx.call_activity(
                            self.persist_tool_results,
                            input={
                                "instance_id": ctx.instance_id,
                                "tool_results": persisted_results,
                                "time": ctx.current_utc_datetime.isoformat(),
                            },
                        )

                    if handoff_results:
                        if not ctx.is_replaying:
                            logger.info(
                                "Agent %s performing handoff on turn %d to %s",
                                self.name,
                                turn,
                                handoff_results[0].get("agent_name"),
                            )
                        if len(handoff_results) > 1:
                            logger.warning(
                                "Agent %s requested %d handoffs, executing first to %s",
                                self.name,
                                len(handoff_results),
                                handoff_results[0].get("agent_name"),
                            )

                        for idx, handoff_res in enumerate(handoff_results):
                            yield ctx.call_activity(
                                self.record_handoff_tool_message,
                                input={
                                    "instance_id": ctx.instance_id,
                                    "tool_call": handoff_res.get("tool_call"),
                                    "agent_name": handoff_res.get("agent_name"),
                                    "executed": idx == 0,
                                    "tool_args": handoff_res.get("tool_args"),
                                    "handoff_spec": handoff_res.get("handoff_spec"),
                                },
                            )

                        handoff_result = handoff_results[0]

                        final_message = {
                            "role": "assistant",
                            "content": f"Handed off to {handoff_result.get('agent_name')}",
                        }
                        yield ctx.call_activity(
                            self.record_final_assistant_message,
                            input={
                                "instance_id": ctx.instance_id,
                                "message": final_message,
                            },
                        )

                        yield ctx.call_activity(
                            self.handle_handoff,
                            input={
                                "handoff_result": handoff_result,
                                "instance_id": ctx.instance_id,
                                "time": ctx.current_utc_datetime.isoformat(),
                            },
                        )
                        is_handoff_exit = True
                        break

                    task = None
                    continue

                if self._is_final_output(assistant_response):
                    final_message = assistant_response
                    if not ctx.is_replaying:
                        logger.debug(
                            "Agent %s produced final response on turn %d (instance=%s)",
                            self.name,
                            turn,
                            ctx.instance_id,
                        )
                    break

                if not ctx.is_replaying:
                    logger.debug(
                        "Agent %s response did not satisfy output schema on turn %d; retrying.",
                        self.name,
                        turn,
                    )
                task = None
                continue
            else:
                raise AgentError("Workflow ended without generating a final response.")

        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent %s workflow failed: %s", self.name, exc)
            final_message = {"role": "assistant", "content": f"Error: {str(exc)}"}

        # Optionally broadcast the final message to the team (skip if handoff to avoid race conditions).
        if self.broadcast_topic_name and not is_handoff_exit:
            yield ctx.call_activity(
                self.broadcast_message_to_agents,
                input={"message": final_message},
            )

        # Optionally send a direct response back to the trigger origin.
        # Skip if expect_response=False (e.g., handoff chains where source doesn't need response).
        if source and trigger_instance_id and expect_response:
            yield ctx.call_activity(
                self.send_response_back,
                input={
                    "response": final_message,
                    "target_agent": source,
                    "target_instance_id": trigger_instance_id,
                },
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
        instance_id = payload.get("instance_id")
        task = payload.get("task")

        # Reload state to ensure we have the latest tool results from parallel activities
        self.load_state()

        chat_history = self._construct_messages_with_instance_history(instance_id)
        logger.debug(
            f"call_llm: Retrieved {len(chat_history)} messages from history for instance {instance_id}"
        )

        messages = self.prompting_helper.build_initial_messages(
            user_input=task,
            chat_history=chat_history,
        )

        # Debug: Log final message sequence
        logger.debug(f"call_llm: Final message sequence has {len(messages)} messages")

        # Sync current system messages into per-instance state
        self._sync_system_messages_with_state(instance_id, messages)

        # Persist the user's turn (if any) into the instance timeline + memory
        user_message = self._get_last_user_message(messages)
        user_copy = dict(user_message) if user_message else None
        self._process_user_message(instance_id, task, user_copy)

        # Only print user message when task is provided (turn 1)
        # On subsequent turns, user message is already in history and shouldn't be reprinted
        if user_copy is not None and task is not None:
            self.text_formatter.print_message({str(k): v for k, v in user_copy.items()})

        generate_kwargs = {
            "messages": messages,
        }
        tools = self.get_llm_tools()
        if tools:
            generate_kwargs["tools"] = tools
        if tools and self.execution.tool_choice is not None:
            generate_kwargs["tool_choice"] = self.execution.tool_choice
        if self.output_type:
            generate_kwargs["response_format"] = self.output_type

        try:
            llm_result = self.llm.generate(**generate_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM generate failed: %s", exc)
            raise AgentError(str(exc)) from exc

        assistant_response: Dict[str, Any]
        if isinstance(llm_result, LLMChatResponse):
            assistant_message = llm_result.get_message()
            if assistant_message is None:
                raise AgentError("LLM returned no assistant message.")
            assistant_response = assistant_message.model_dump()
        else:
            # Structured output already validated; wrap it into an assistant message.
            if hasattr(llm_result, "model_dump"):
                payload = llm_result.model_dump()
            else:
                payload = llm_result
            assistant_response = {
                "role": "assistant",
                "content": json.dumps(payload, ensure_ascii=False),
            }
            assistant_message = self._message_dict_to_message_model(assistant_response)

        if assistant_message is None:
            raise AgentError("LLM returned no assistant message.")

        self._save_assistant_message(instance_id, assistant_response)
        self.text_formatter.print_message(assistant_response)
        self.save_state()
        return assistant_response

    def _is_final_output(self, assistant_response: Dict[str, Any]) -> bool:
        """
        Determine whether the assistant response represents a final output according to
        the configured output schema.
        """
        if assistant_response.get("tool_calls"):
            return False

        content = assistant_response.get("content")
        if content is None or (isinstance(content, str) and not content.strip()):
            return False

        if not self.output_type or self.output_type is str:
            return True

        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError:
            logger.debug(
                "Agent %s structured output is not valid JSON yet; continuing loop.",
                self.name,
            )
            return False

        try:
            TypeAdapter(self.output_type).validate_python(parsed)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Agent %s structured output failed validation: %s; continuing loop.",
                self.name,
                exc,
            )
            return False

    def run_tool(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single tool call and return results for later persistence.

        Args:
            payload: Keys 'tool_call', 'instance_id', 'time', 'order'.

        Returns:
            Tool execution record as a dict with optional 'is_handoff' flag.

        Raises:
            AgentError: If tool arguments contain invalid JSON.
        """
        tool_call = payload.get("tool_call", {})
        instance_id = payload.get("instance_id")
        fn_name = tool_call["function"]["name"]
        raw_args = tool_call["function"].get("arguments", "")

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            raise AgentError(f"Invalid JSON in tool args: {exc}") from exc

        if not instance_id:
            raise AgentError("Tool payload missing workflow instance id.")

        # Check if this is a handoff tool BEFORE executing it
        normalized_name = fn_name.lower()
        agent_name = self._handoff_tools.get(normalized_name)
        spec = self._handoff_catalog.get(normalized_name)

        if agent_name:
            spec_payload: Dict[str, Any] | None = None
            if spec:
                spec_payload = {
                    "agent_name": spec.agent_name,
                    "tool_name": spec.tool_name or fn_name,
                    "description": spec.description,
                    "default_task": spec.default_task,
                }
            return {
                "is_handoff": True,
                "agent_name": agent_name,
                "tool_call": tool_call,
                "tool_args": args,
                "handoff_spec": spec_payload,
            }

        tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
        if not tool_call_id:
            tool_call_id = f"{fn_name}:{payload.get('order', 0)}"
            logger.debug(
                "run_tool: Missing tool_call.id, synthesizing fallback id '%s' for %s (instance=%s)",
                tool_call_id,
                fn_name,
                instance_id,
            )
        tool_call_id = str(tool_call_id)

        try:
            result = self._run_asyncio_task(
                self.tool_executor.run_tool(fn_name, **args)
            )
        except AgentToolExecutorError as exc:
            raise AgentError(str(exc)) from exc

        if isinstance(result, str):
            serialized_result = result
        else:
            try:
                serialized_result = json.dumps(result)
            except Exception:  # noqa: BLE001
                serialized_result = str(result)

        tool_result = {
            "tool_call_id": tool_call_id,
            "tool_name": fn_name,
            "tool_args": args,
            "execution_result": serialized_result,
        }

        logger.debug(
            "run_tool: Executed %s (call_id=%s) for instance %s",
            fn_name,
            tool_call_id,
            instance_id,
        )

        tool_message_dict = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": fn_name,
            "content": serialized_result,
        }
        self.text_formatter.print_message(tool_message_dict)

        return {
            "is_handoff": False,
            "tool_result": tool_result,
        }

    def persist_tool_results(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Persist tool execution results after parallel tool activities complete.

        Args:
            payload: Dict with 'instance_id', 'tool_results', and optional 'time'.
        """
        instance_id = payload.get("instance_id")
        results = payload.get("tool_results") or []
        if not instance_id or not results:
            return

        self.load_state()

        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None

        existing_runtime_ids = {
            getattr(record, "tool_call_id", None)
            for record in getattr(self, "tool_history", [])
        }

        for result in results:
            tool_data = result.get("tool_result") or {}
            tool_call_id = tool_data.get("tool_call_id")
            tool_name = tool_data.get("tool_name")
            execution_result = tool_data.get("execution_result")

            if not tool_call_id or not tool_name:
                logger.debug(
                    "persist_tool_results: skipping incomplete tool payload %s",
                    result,
                )
                continue

            history_entry = ToolExecutionRecord(**tool_data)
            tool_message = ToolMessage(
                tool_call_id=tool_call_id,
                name=tool_name,
                content=execution_result,
                role="tool",
            )
            agent_message = {
                "id": tool_message.tool_call_id,
                "role": "tool",
                "name": tool_message.name,
                "content": tool_message.content,
                "tool_call_id": tool_message.tool_call_id,  # Include tool_call_id for proper deduplication
                "agent_name": self.name,
            }

            if entry is not None and hasattr(entry, "messages"):
                existing_ids = self._get_existing_message_ids(entry)
                if agent_message["id"] not in existing_ids:
                    tool_message_model = (
                        self._message_coercer(agent_message)
                        if getattr(self, "_message_coercer", None)
                        else self._message_dict_to_message_model(agent_message)
                    )
                    entry.messages.append(tool_message_model)
                    if hasattr(entry, "tool_history"):
                        entry.tool_history.append(history_entry)
                    if hasattr(entry, "last_message"):
                        entry.last_message = tool_message_model

            if history_entry.tool_call_id not in existing_runtime_ids:
                self.tool_history.append(history_entry)
                existing_runtime_ids.add(history_entry.tool_call_id)
                self.memory.add_message(tool_message)

        self.save_state()
        logger.debug(
            "persist_tool_results: persisted %d tool result(s) for instance %s",
            len(results),
            instance_id,
        )

    def record_handoff_tool_message(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Record a synthetic tool message for a handoff to satisfy OpenAI's message structure.

        Args:
            payload: Dict with 'instance_id', 'tool_call', 'agent_name', 'executed'.
        """
        instance_id = payload.get("instance_id")
        tool_call = payload.get("tool_call")
        agent_name = payload.get("agent_name")
        executed = payload.get("executed", False)
        tool_args = payload.get("tool_args") or {}
        if not tool_call or not agent_name or not instance_id:
            return

        function_entry = tool_call.get("function", {})
        if isinstance(function_entry, dict):
            name = function_entry.get("name") or "handoff"
            call_id = function_entry.get("call_id")
        else:
            name = function_entry or "handoff"
            call_id = None

        tool_call_id = (
            tool_call.get("id") or tool_call.get("call_id") or call_id or name
        )

        execution_payload: Dict[str, Any] = {
            "handoff_to": agent_name,
            "executed": executed,
            "tool_args": tool_args,
        }

        content = json.dumps(
            execution_payload,
            ensure_ascii=False,
        )

        tool_message = ToolMessage(
            tool_call_id=str(tool_call_id),
            name=name,
            content=content,
            role="tool",
        )
        agent_message = {
            "id": tool_message.tool_call_id,
            "role": "tool",
            "name": tool_message.name,
            "content": tool_message.content,
            "tool_call_id": tool_message.tool_call_id,  # Required for print_message formatting
            "agent_name": self.name,
        }

        # Print synthetic handoff tool result for user visibility
        self.text_formatter.print_message(agent_message)

        tool_result = {
            "tool_call_id": tool_message.tool_call_id,
            "tool_name": tool_message.name,
            "tool_args": tool_args,
            "execution_result": content,
        }
        history_entry = ToolExecutionRecord(**tool_result)

        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None
        if entry is not None and hasattr(entry, "messages"):
            existing_ids = self._get_existing_message_ids(entry)
            if agent_message["id"] not in existing_ids:
                tool_message_model = (
                    self._message_coercer(agent_message)
                    if getattr(self, "_message_coercer", None)
                    else self._message_dict_to_message_model(agent_message)
                )
                entry.messages.append(tool_message_model)
                if hasattr(entry, "last_message"):
                    entry.last_message = tool_message_model
                if hasattr(entry, "tool_history"):
                    existing_history_ids = {
                        getattr(t, "tool_call_id", None)
                        for t in getattr(entry, "tool_history")
                    }
                    if history_entry.tool_call_id not in existing_history_ids:
                        entry.tool_history.append(history_entry)

        existing_runtime_ids = {
            getattr(record, "tool_call_id", None)
            for record in getattr(self, "tool_history", [])
        }
        if history_entry.tool_call_id not in existing_runtime_ids:
            self.tool_history.append(history_entry)

        self.memory.add_message(tool_message)
        self.save_state()

    def record_final_assistant_message(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Persist a final assistant message generated after a handoff.

        Args:
            payload: Dict with 'instance_id' and 'message' (assistant-like dict).
        """
        instance_id = payload.get("instance_id")
        message = payload.get("message")
        if not instance_id or not isinstance(message, dict):
            return

        # Ensure a defensive copy before mutating in _save_assistant_message
        snapshot = dict(message)
        try:
            self._save_assistant_message(instance_id, snapshot)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to record final assistant message for instance %s", instance_id
            )

    def broadcast_message_to_agents(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Broadcast a message to all agents via pub/sub (if a broadcast topic is set).

        Args:
            payload: Dict containing 'message' (assistant/user-like dict).
        """
        raw_message = payload.get("message", {})
        if not isinstance(raw_message, dict) or not self.broadcast_topic_name:
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

        message = dict(raw_message)
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
        raw_response = payload.get("response", {})
        target_agent = payload.get("target_agent", "")
        target_instance_id = payload.get("target_instance_id", "")
        if not target_agent or not target_instance_id:
            logger.debug(
                "Target agent or instance missing; skipping response publication."
            )
            return

        response = dict(raw_response)
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

    def handle_handoff(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Execute handoff to target agent by triggering their workflow.

        Args:
            payload: Dict with 'handoff_result', 'instance_id', 'time'.
        """
        handoff_result = payload.get("handoff_result", {})
        instance_id = payload.get("instance_id")
        agent_name = handoff_result.get("agent_name")
        handoff_spec = handoff_result.get("handoff_spec") or {}

        if not agent_name:
            logger.warning("Handoff result missing agent_name")
            return

        # Extract handoff context from tool arguments
        tool_args = handoff_result.get("tool_args", {}) or {}
        raw_task = tool_args.get("task")
        handoff_task = raw_task.strip() if isinstance(raw_task, str) else ""

        if not handoff_task:
            fallback_task = handoff_spec.get("default_task")
            if fallback_task:
                handoff_task = fallback_task
            else:
                logger.error(
                    "Handoff tool called without task for instance %s; using default prompt.",
                    instance_id,
                )
                handoff_task = "Continue the conversation."
            tool_args.setdefault("task", handoff_task)

        # Trigger target agent workflow
        agents_metadata = self.get_agents_metadata()
        if agent_name not in agents_metadata:
            logger.error("Target agent %s not found in metadata", agent_name)
            return

        metadata_payload: Dict[str, Any] = {
            "source_agent": self.name,
            "tool_args": tool_args,
        }
        if handoff_spec:
            metadata_payload["handoff_spec"] = handoff_spec

        trigger_message = TriggerAction(
            task=handoff_task,
            workflow_instance_id=instance_id,
            expect_response=False,  # Handoffs don't expect response back to source
            handoff_metadata=metadata_payload,
        )

        async def _trigger_handoff() -> None:
            await send_message_to_agent(
                source=self.name,
                target_agent=agent_name,
                message=trigger_message,
                agents_metadata=agents_metadata,
            )

        try:
            self._run_asyncio_task(_trigger_handoff())
            logger.info("Handed off from %s to %s", self.name, agent_name)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to trigger handoff to %s", agent_name)

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
                logger.info("WorkflowRuntime shut down for agent '%s'.", self.name)
            except Exception as exc:
                logger.warning(
                    "Error while shutting down workflow runtime for agent '%s': %s",
                    self.name,
                    exc,
                    exc_info=True,
                )

        self._started = False

    def __del__(self) -> None:
        """
        Best-effort cleanup during garbage collection.

        Ensures the workflow runtime is properly shut down if owned by this agent.
        This prevents "Invalid file descriptor" errors during interpreter shutdown.
        """
        try:
            if self._started and self._runtime_owned:
                self._runtime.shutdown()
        except Exception:
            pass

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
        runtime.register_activity(self.persist_tool_results)
        runtime.register_activity(self.broadcast_message_to_agents)
        runtime.register_activity(self.send_response_back)
        runtime.register_activity(self.finalize_workflow)
        runtime.register_activity(self.handle_handoff)
        runtime.register_activity(self.record_handoff_tool_message)
        runtime.register_activity(self.record_final_assistant_message)
