from __future__ import annotations

from datetime import timedelta
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, Iterable, List, Optional
from os import getenv

import dapr.ext.workflow as wf
from pydantic import BaseModel

from dapr_agents.agents.orchestrators.llm.prompts import (
    NEXT_STEP_PROMPT,
    PROGRESS_CHECK_PROMPT,
    SUMMARY_GENERATION_PROMPT,
    TASK_INITIAL_PROMPT,
    TASK_PLANNING_PROMPT,
)
from dapr_agents.agents.orchestrators.llm.schemas import (
    IterablePlanStep,
    NextStep,
    ProgressCheckOutput,
    schemas,
)
from dapr_agents.agents.orchestrators.llm.state import PlanStep
from dapr_agents.agents.orchestrators.llm.utils import (
    find_step_in_plan,
    restructure_plan,
    update_step_statuses,
)

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
    AgentWorkflowMessage,
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
    AssistantMessage,
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
        orchestrator: bool = False,
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
            orchestrator: Whether this agent is an orchestrator (affects registration).
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
            orchestrator=orchestrator,
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
                
                if self.orchestrator:
                    agents = yield ctx.call_activity(self._get_available_agents)

                    if turn == 1:
                        init = yield ctx.call_activity(
                            self.call_llm,
                            input={
                                "instance_id": ctx.instance_id,
                                "task": f"{TASK_PLANNING_PROMPT.format(
                                    task=task, agents=agents, plan_schema=schemas.plan
                                )}",
                                "time": ctx.current_utc_datetime.isoformat(),
                                "response_format": IterablePlanStep.model_construct().model_dump_json(),
                            },
                        )

                        content = init.get("content", "{}")
                        try:
                            parsed_content = json.loads(content)
                            plan = parsed_content.get("objects", [])
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse LLM response content: {e}")
                            plan = []

                        if not ctx.is_replaying:
                            logger.info(
                                "Received plan from initialization with %d steps", len(plan)
                            )

                        if not ctx.is_replaying:
                            logger.info(
                                "Broadcasting initial plan with %d steps to all agents",
                                len(plan),
                            )
                        yield ctx.call_activity(
                            self.broadcast_message_to_agents,
                            input={"message": plan},
                        )
                        if not ctx.is_replaying:
                            logger.info("Initial plan broadcast completed")
                        
                        plan_content = json.dumps({"objects": plan}, indent=2)
                        plan_message = {
                            "role": "assistant",
                            "name": self.name,
                            "content": plan_content,
                        }
                        yield ctx.call_activity(
                            self._save_plan_message,
                            input={
                                "instance_id": ctx.instance_id,
                                "plan_message": plan_message,
                                "time": ctx.current_utc_datetime.isoformat(),
                            },
                            retry_policy=self._retry_policy,
                        )
                    
                    else:
                        container = self._get_entry_container()
                        entry = container.get(ctx.instance_id) if container else None
                        messages = getattr(entry, "messages") if entry and hasattr(entry, "messages") else []
                        plan = self.get_plan_from_messages(messages) or []
                        if not ctx.is_replaying:
                            logger.info(
                                "Loaded plan from state with %d steps (turn %d)",
                                len(plan),
                                turn,
                            )
                        if len(plan) == 0:
                            if not ctx.is_replaying:
                                logger.info(
                                    "No plan found in state; ending workflow for agent %s (instance=%s)",
                                    self.name,
                                    ctx.instance_id,
                                )
                            raise AgentError("No plan available; cannot continue orchestration.")
                        
                    next_step = yield ctx.call_activity(
                        self.call_llm,
                        input={
                            "task": f"{NEXT_STEP_PROMPT.format(
                                task=task,
                                agents=agents,
                                plan=plan,
                                next_step_schema=schemas.next_step,
                            )}",
                            "time": ctx.current_utc_datetime.isoformat(),
                            "response_format": NextStep.model_construct().model_dump_json(),
                        },
                    )

                    try:
                        parsed_content = json.loads(next_step.get("content", "{}"))
                        next_agent = parsed_content.get("next_agent")
                        instruction = parsed_content.get("instruction")
                        step_id = parsed_content.get("step")
                        substep_id = parsed_content.get("substep")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse LLM response content: {e}")
                        next_agent = None
                        instruction = None
                        step_id = None
                        substep_id = None

                    if not ctx.is_replaying:
                        logger.info(f"Next step decided: agent={next_agent}, step={step_id}, substep={substep_id}, instruction={instruction}")

                    is_valid = yield ctx.call_activity(
                        self._validate_next_step,
                        input={
                            "instance_id": ctx.instance_id,
                            "plan": self._convert_plan_objects_to_dicts(plan),
                            "step": step_id,
                            "substep": substep_id,
                        },
                    )

                    if is_valid:
                        if not ctx.is_replaying:
                            self.print_interaction(
                                source_agent_name=self.name,
                                target_agent_name=next_agent,
                                message=instruction,
                        )
                            
                        agents = self.list_team_agents(include_self=False, team=self.effective_team())
                        agent_appId = agents[next_agent]["agent"]["appid"]

                        retry_policy = wf.RetryPolicy(
                            max_number_of_attempts=2,
                            first_retry_interval=timedelta(milliseconds=100),
                            max_retry_interval=timedelta(seconds=3),
                        )

                        if not ctx.is_replaying:
                            logger.info(f"Invoking agent {next_agent} at app ID {agent_appId} for step {step_id}, substep {substep_id}")

                        result = yield ctx.call_child_workflow(
                            workflow="agent_workflow",
                            input={"task": instruction },
                            app_id=agent_appId,
                            retry_policy=retry_policy,
                        )

                        if not ctx.is_replaying:
                            self.print_interaction(
                                source_agent_name=result.get("name", ""),
                                target_agent_name=self.name,
                                message=result.get("content", ""),
                            )
                        target = find_step_in_plan(plan, step_id, substep_id)
                        if target:
                            target["status"] = "completed"
                        plan = update_step_statuses(plan)

                        processed = yield ctx.call_activity(
                            self.call_llm,
                            input={
                                "task": f"{PROGRESS_CHECK_PROMPT.format(
                                    task=task,
                                    plan=plan,
                                    step=step_id,
                                    substep=substep_id,
                                    results=result.get("content", ""),
                                    progress_check_schema=schemas.progress_check,
                                )}",
                                "time": ctx.current_utc_datetime.isoformat(),
                                "response_format": ProgressCheckOutput.model_construct().model_dump_json(),
                            }
                        )

                        progress = yield ctx.call_activity(
                            self._parse_progress, 
                            input={
                                "content": processed.get("content", ""),
                                "instance_id": ctx.instance_id,
                                "plan_objects": plan,
                            },
                        )

                        plan = progress["plan"]
                        verdict = progress["verdict"]

                        plan_content = json.dumps({"objects": plan}, indent=2)
                        plan_message = {
                            "role": "assistant",
                            "name": self.name,
                            "content": plan_content,
                        }
                        yield ctx.call_activity(
                            self._save_plan_message,
                            input={
                                "instance_id": ctx.instance_id,
                                "plan_message": plan_message,
                                "time": ctx.current_utc_datetime.isoformat(),
                            },
                            retry_policy=self._retry_policy,
                        )

                    else:
                        verdict = "continue"
                        processed = {
                            "name": self.name,
                            "role": "user",
                            "content": f"Step {step_id}, substep {substep_id} not found. Adjusting workflow…",
                        }
                    
                    if verdict != "continue" or turn == self.execution.max_iterations:
                        final_message = yield ctx.call_activity(
                            self._finalize_workflow_with_summary,
                            input={
                                "instance_id": ctx.instance_id,
                                "task": task or "",
                                "verdict": verdict
                                if verdict != "continue"
                                else "max_iterations_reached",
                                "plan_objects": self._convert_plan_objects_to_dicts(plan),
                                "step_id": step_id,
                                "substep_id": substep_id,
                                "agent": next_agent if is_valid else self.name,
                                "result": processed["content"],
                                "wf_time": ctx.current_utc_datetime.isoformat(),
                            },
                        )
                        if not ctx.is_replaying:
                            logger.info("Workflow %s finalized.", ctx.instance_id)

                        return final_message
                    else:
                        task = processed["content"]
                        continue

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
    
    @staticmethod
    def _convert_plan_objects_to_dicts(plan_objects: List[Any]) -> List[Dict[str, Any]]:
        """
        Converts plan objects (Pydantic models or dictionaries) into dictionaries.

        Args:
            plan_objects (List[Any]): A list of plan objects to convert.

        Returns:
            List[Dict[str, Any]]: The converted plan objects as dictionaries.
        """
        if not plan_objects:
            return []
        return [
            obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)
            for obj in plan_objects
        ]
    
    def get_plan_from_messages(self, messages: List[AgentWorkflowMessage]) -> Optional[List[Dict[str, Any]]]:
        # Find all assistant messages with JSON content starting with {
        plan_messages = [
            m for m in messages 
            if m.role == "assistant" and m.content.strip().startswith("{")
        ]
        
        # Get the LAST (most recent) plan message
        if not plan_messages:
            return None
        
        plan_msg = plan_messages[-1]  # Get the last one
        
        try:
            data = json.loads(plan_msg.content)
            return self._convert_plan_objects_to_dicts([PlanStep(**obj) for obj in data.get("objects", [])])
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Failed to parse plan from message: %s", e)
            return None

    async def update_plan_internal(
        self,
        *,
        instance_id: str,
        plan: List[Dict[str, Any]],
        status_updates: Optional[List[Dict[str, Any]]] = None,
        plan_updates: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply status/structure updates to the plan and persist them.

        Args:
            instance_id: Workflow instance id.
            plan: Current plan snapshot.
            status_updates: Optional status updates, each with `step`, optional `substep`,
                and `status` fields.
            plan_updates: Optional structural updates (see `restructure_plan` utility).

        Returns:
            The updated plan after applying changes.

        Raises:
            ValueError: If a referenced step/substep is not found in the plan.
        """

        logger.debug("Updating plan for instance %s", instance_id)

        # Validate and apply status updates.
        if status_updates:
            logger.info("Applying %d status update(s) to plan", len(status_updates))
            for u in status_updates:
                step_id = u["step"]
                sub_id = u.get("substep")
                new_status = u["status"]

                logger.debug(
                    "Updating step %s/%s to status '%s'",
                    step_id,
                    sub_id,
                    new_status,
                )
                target = find_step_in_plan(plan, step_id, sub_id)
                if not target:
                    msg = f"Step {step_id}/{sub_id} not present in plan."
                    logger.error(msg)
                    raise ValueError(msg)

                # Apply status update
                target["status"] = new_status
                logger.debug(
                    "Successfully updated status of step %s/%s to '%s'",
                    step_id,
                    sub_id,
                    new_status,
                )

        # Apply structural updates while preserving substeps unless explicitly overridden.
        if plan_updates:
            logger.debug("Applying %d plan restructuring update(s)", len(plan_updates))
            plan = restructure_plan(plan, plan_updates)

        # Apply global consistency checks for statuses
        plan = update_step_statuses(plan)

        # Persist the updated plan
        #self.update_workflow_state(instance_id=instance_id, plan=plan)

        logger.debug("Plan successfully updated for instance %s", instance_id)
        return plan

    async def finish_workflow_internal(
        self,
        *,
        instance_id: str,
        plan: List[Dict[str, Any]],
        step: int,
        substep: Optional[float],
        verdict: str,
        summary: str,
        wf_time: Optional[str],
    ) -> None:
        """
        Finalize workflow by updating statuses (if completed) and storing the summary.

        Args:
            instance_id: Workflow instance id.
            plan: Current plan snapshot.
            step: Completed step id.
            substep: Completed substep id (if any).
            verdict: Outcome category (e.g., "completed", "failed", "max_iterations_reached").
            summary: Final summary content to persist.
            wf_time: Workflow timestamp (ISO 8601 string) to set as end time if provided.

        Returns:
            None

        Raises:
            ValueError: If a completed step/substep reference is invalid.
        """

        logger.debug(
            "Finalizing workflow for instance %s with verdict '%s'",
            instance_id,
            verdict,
        )

        status_updates: List[Dict[str, Any]] = []

        if verdict == "completed":
            # Find and validate the step or substep
            step_entry = find_step_in_plan(plan, step, substep)
            if not step_entry:
                msg = f"Step {step}/{substep} not found in plan; cannot mark as completed."
                logger.error(msg)
                raise ValueError(msg)

            # Mark the step or substep as completed
            step_entry["status"] = "completed"
            status_updates.append(
                {"step": step, "substep": substep, "status": "completed"}
            )
            logger.debug("Marked step %s/%s as completed", step, substep)

            # If it's a substep, check if all sibling substeps are completed
            if substep is not None:
                parent_step = find_step_in_plan(
                    plan, step
                )  # Get parent without substep
                if parent_step:
                    # Ensure "substeps" is a valid list before iteration
                    substeps = parent_step.get("substeps", [])
                    if not isinstance(substeps, list):
                        substeps = []

                    all_substeps_completed = all(
                        ss.get("status") == "completed" for ss in substeps
                    )
                    if all_substeps_completed:
                        parent_step["status"] = "completed"
                        status_updates.append({"step": step, "status": "completed"})
                        logger.debug(
                            "All substeps of step %s completed; marked parent as completed",
                            step,
                        )

        # Apply updates in one call if any status changes were made
        if status_updates:
            await self.update_plan_internal(
                instance_id=instance_id,
                plan=plan,
                status_updates=status_updates,
            )

        # Store the final summary and verdict in workflow state
        #self.update_workflow_state(
        #    instance_id=instance_id,
        #    final_output=summary,
        #    wf_time=wf_time,
        #)

        logger.info(
            "Workflow %s finalized with verdict '%s'",
            instance_id,
            verdict,
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
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any], response_format: Optional[str] = None
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
        if response_format is not None:
            generate_kwargs["response_model"] = response_format
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
    # Orchestrator Activities
    # ------------------------------------------------------------------

    def _get_available_agents(self, ctx: wf.WorkflowActivityContext) -> str:
        """
        Return a human-formatted list of available agents (excluding orchestrators).

        Args:
            ctx: The Dapr Workflow context.

        Returns:
            A formatted string listing available agents.
        """
        agents_metadata = self.list_team_agents(
            include_self=False, team=self.effective_team()
        )
        if not agents_metadata:
            return "No available agents to assign tasks."
        lines = []
        for name, meta in agents_metadata.items():
            role = meta["agent"].get("role", "Unknown role")
            goal = meta["agent"].get("goal", "Unknown")
            lines.append(f"- {name}: {role} (Goal: {goal})")
        return "\n".join(lines)

    def _validate_next_step(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> bool:
        """Return True if (step, substep) exists in the plan."""
        step = payload["step"]
        substep = payload.get("substep")
        plan = payload["plan"]
        ok = bool(find_step_in_plan(plan, step, substep))
        if not ok:
            logger.error(
                "Step %s/%s not in plan for instance %s",
                step,
                substep,
                payload.get("instance_id"),
            )
        return ok
    
    def _finalize_workflow_with_summary(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> str:
        """
        Ask the LLM for a final summary and persist the finale (plan status + output + end time).
        """
        instance_id = payload["instance_id"]

        async def _finalize() -> str:
            prompt = SUMMARY_GENERATION_PROMPT.format(
                task=payload["task"],
                verdict=payload["verdict"],
                plan=json.dumps(payload["plan_objects"], indent=2),
                step=payload["step_id"],
                substep=payload["substep_id"]
                if payload["substep_id"] is not None
                else "N/A",
                agent=payload["agent"],
                result=payload["result"],
            )
            summary_resp = self.llm.generate(
                messages=[{"role": "user", "content": prompt}]
            )
            if hasattr(summary_resp, "choices") and summary_resp.choices:
                summary = summary_resp.choices[0].message.content
            elif hasattr(summary_resp, "results") and summary_resp.results:
                # Handle LLMChatResponse with results list
                summary = summary_resp.results[0].message.content
            else:
                # Fallback: try to extract content from the response object
                summary = str(summary_resp)

            await self.finish_workflow_internal(
                instance_id=instance_id,
                plan=list(payload["plan_objects"]),
                step=payload["step_id"],
                substep=payload["substep_id"],
                verdict=payload["verdict"],
                summary=summary,
                wf_time=payload["wf_time"],
            )
            return summary

        return self._run_asyncio_task(_finalize())

    def _parse_progress(self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse progress information from assistant content.

        Args:
            payload: Dict with 'content', 'instance_id', and 'plan_objects'.
        Returns:
            Parsed progress dict or None if not found.
        """

        content = payload["content"]
        plan_objects = list(payload["plan_objects"])
        instance_id = payload["instance_id"]

        if hasattr(content, "choices") and content.choices:
            data = content.choices[0].message.content
            progress = ProgressCheckOutput(**json.loads(data))
        elif isinstance(content, ProgressCheckOutput):
            progress = content
        else:
            try:
                data = json.loads(content) if isinstance(content, str) else content
                progress = ProgressCheckOutput(**data)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.error(f"Failed to parse progress check output: {e}. Content: {content}")
                raise AgentError(f"Invalid progress check format: {e}")

        status_updates = [
            (u.model_dump() if hasattr(u, "model_dump") else u)
            for u in (progress.plan_status_update or [])
        ]
        plan_updates = [
            (u.model_dump() if hasattr(u, "model_dump") else u)
            for u in (progress.plan_restructure or [])
        ]
        async def _update_plan() -> List[Dict[str, Any]]:
            if status_updates or plan_updates:
                updated_plan = await self.update_plan_internal(
                    instance_id=instance_id,
                    plan=plan_objects,
                    status_updates=status_updates,
                    plan_updates=plan_updates,
                )
            else:
                updated_plan = plan_objects
            return updated_plan

        return {
            "plan": self._run_asyncio_task(_update_plan()),
            "verdict": progress.verdict,
            "status_updates": status_updates,
            "plan_updates": plan_updates,
            "status": "success",
        }

    def _save_plan_message(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Save the plan as an assistant message to the workflow state.

        Args:
            payload: Dict with 'instance_id', 'plan_message', and 'time'.
        """
        if self.state_store:
            self.load_state()

        instance_id = payload.get("instance_id")
        plan_message = payload.get("plan_message")

        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None

        if entry is not None and hasattr(entry, "messages"):
            plan_message_model = (
                self._message_coercer(plan_message)
                if getattr(self, "_message_coercer", None)
                else self._message_dict_to_message_model(plan_message)
            )
            entry.messages.append(plan_message_model)
            if hasattr(entry, "last_message"):
                entry.last_message = plan_message_model

        # Also add to memory
        self.memory.add_message(
            AssistantMessage(
                content=plan_message.get("content", ""),
                name=plan_message.get("name")
            )
        )

        self.save_state()
        logger.info(f"Saved plan to memory for instance {instance_id}")

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
        runtime.register_activity(self._get_available_agents)
        runtime.register_activity(self._validate_next_step)
        runtime.register_activity(self._finalize_workflow_with_summary)
        runtime.register_activity(self._parse_progress)
        runtime.register_activity(self._save_plan_message)
