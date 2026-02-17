from __future__ import annotations

from datetime import timedelta
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Literal
from os import getenv

import dapr.ext.workflow as wf

from dapr_agents.agents.orchestration import (
    OrchestrationStrategy,
    AgentOrchestrationStrategy,
    RoundRobinOrchestrationStrategy,
    RandomOrchestrationStrategy,
)
from dapr_agents.agents.orchestrators.llm.prompts import (
    NEXT_STEP_PROMPT,
    PROGRESS_CHECK_PROMPT,
    SUMMARY_GENERATION_PROMPT,
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
    OrchestrationMode,
    AgentConfigurationConfig,
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
        # Memory
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
        configuration: Optional[AgentConfigurationConfig] = None,
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

            memory: Enable long-term conversation memory storage; defaults to false.
            llm: Chat client; defaults to `get_default_llm()`.
            tools: Optional tool callables or `AgentTool` instances.

            agent_metadata: Extra metadata to publish to the registry.
            workflow_grpc: Optional gRPC overrides for the workflow runtime channel.
            runtime: Optional pre-existing workflow runtime to attach to.
            retry_policy: Durable retry policy configuration.
            agent_observability: Observability configuration for tracing/logging.
            configuration: Optional configuration store settings for hot-reloading.
        """
        # Mark orchestrators to filtered out when other orchestrators query for available agents
        if execution and execution.orchestration_mode:
            agent_metadata = dict(agent_metadata or {})
            agent_metadata["orchestrator"] = True

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
            configuration=configuration,
        )

        grpc_options = getattr(self, "workflow_grpc_options", None)
        apply_grpc_options(grpc_options)

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

        self._orchestration_strategy: Optional[OrchestrationStrategy] = None
        # Initialize orchestration strategy if this agent is an orchestrator
        if self.execution.orchestration_mode:
            # Default to "agent" mode if no mode specified

            match self.execution.orchestration_mode:
                case OrchestrationMode.AGENT:
                    self._orchestration_strategy = AgentOrchestrationStrategy()
                case OrchestrationMode.ROUNDROBIN:
                    self._orchestration_strategy = RoundRobinOrchestrationStrategy()
                case OrchestrationMode.RANDOM:
                    self._orchestration_strategy = RandomOrchestrationStrategy()
                case _:
                    raise ValueError(
                        f"Invalid orchestration_mode: {self.execution.orchestration_mode}. "
                        f"Must be one of: 'agent', 'roundrobin', 'random'"
                    )

            # Store orchestrator name for strategy finalization
            self._orchestration_strategy.orchestrator_name = self.name
            self.effective_team = registry.team_name if registry else "default"

            logger.debug(
                f"Initialized orchestrator '{self.name}' with mode '{self.execution.orchestration_mode}'"
            )

    # ------------------------------------------------------------------
    # Runtime accessors
    # ------------------------------------------------------------------
    @property
    def orchestrator(self) -> bool:
        """True if this agent is configured as an orchestrator (has orchestration strategy)."""
        return self._orchestration_strategy is not None

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
        # TODO(@sicoyle): fix this bc do i want state on the obj or just refetch it every time with get instead of load?
        if self.state_store:
            self._infra.get_state(ctx.instance_id)

        if not ctx.is_replaying:
            logger.info("Initial message from %s -> %s", source, self.name)

        # Record initial entry via activity to keep deterministic/replay-friendly I/O.
        yield ctx.call_activity(
            self.record_initial_entry,
            input={
                "instance_id": ctx.instance_id,
                "source": source,
                "triggering_workflow_instance_id": trigger_instance_id,
                "trace_context": otel_span_context,
            },
            retry_policy=self._retry_policy,
        )

        final_message: Dict[str, Any] = {}
        turn = 0

        try:
            # Delegate to orchestration workflow if this agent is an orchestrator
            if self._orchestration_strategy:
                if not ctx.is_replaying:
                    logger.info(
                        "Agent %s delegating to orchestration_workflow (instance=%s)",
                        self.name,
                        ctx.instance_id,
                    )

                final_message = yield ctx.call_child_workflow(
                    workflow=self.orchestration_workflow,
                    input={
                        "task": task,
                        "instance_id": ctx.instance_id,
                        "triggering_workflow_instance_id": trigger_instance_id,
                        "start_time": ctx.current_utc_datetime.isoformat(),
                    },
                    retry_policy=self._retry_policy,
                )

                if not ctx.is_replaying:
                    logger.info(
                        "Orchestration workflow completed (instance=%s)",
                        ctx.instance_id,
                    )

            # Standard agent execution loop
            else:
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

        if self.memory is not None:
            yield ctx.call_activity(
                self.summarize,
                input={},
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

    def orchestration_workflow(self, ctx: wf.DaprWorkflowContext, message: dict):
        """Dedicated orchestration workflow using strategy pattern.

        This is an internal workflow called via call_child_workflow from agent_workflow.

        Args:
            ctx: Dapr workflow context
            message: Input dict with task, instance_id, triggering_workflow_instance_id, start_time

        Returns:
            Final message dict to be returned to caller
        """
        task = message.get("task")
        instance_id = message.get("instance_id")

        if not ctx.is_replaying:
            logger.info(
                f"Orchestration workflow started for instance {instance_id} with task: {task}"
            )

        agents_result = yield ctx.call_activity(
            self._get_available_agents,
            retry_policy=self._retry_policy,
        )

        if isinstance(self._orchestration_strategy, AgentOrchestrationStrategy):
            agents_formatted = agents_result["formatted"]
            plan_prompt = TASK_PLANNING_PROMPT.format(
                task=task, agents=agents_formatted, plan_schema=schemas.plan
            )
            init_response = yield ctx.call_activity(
                self.call_llm,
                input={
                    "instance_id": instance_id,
                    "task": plan_prompt,
                    "time": ctx.current_utc_datetime.isoformat(),
                    "response_format": "IterablePlanStep",
                },
                retry_policy=self._retry_policy,
            )

            content = init_response.get("content", "{}")
            try:
                parsed_content = json.loads(content)
                plan = parsed_content.get("objects", [])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response content: {e}")
                plan = []

            if not ctx.is_replaying:
                logger.info(f"Received plan from initialization with {len(plan)} steps")

            yield ctx.call_activity(
                self.broadcast_message_to_agents,
                input={"message": plan},
                retry_policy=self._retry_policy,
            )

            plan_content = json.dumps({"objects": plan}, indent=2)
            plan_message = {
                "role": "assistant",
                "name": self.name,
                "content": plan_content,
            }
            yield ctx.call_activity(
                self._save_plan_message,
                input={
                    "instance_id": instance_id,
                    "plan_message": plan_message,
                    "time": ctx.current_utc_datetime.isoformat(),
                },
                retry_policy=self._retry_policy,
            )

            orch_state = {
                "task": task,
                "agents": agents_formatted,
                "plan": plan,
                "task_history": [],
                "verdict": None,
            }
        else:
            agents_metadata = agents_result["metadata"]
            orch_state = yield ctx.call_activity(
                self._initialize_orchestration,
                input={
                    "task": task,
                    "agents": agents_metadata,
                    "instance_id": instance_id,
                },
                retry_policy=self._retry_policy,
            )

        for turn in range(1, self.execution.max_iterations + 1):
            if not ctx.is_replaying:
                logger.debug(
                    f"Orchestration turn {turn}/{self.execution.max_iterations} (instance={instance_id})"
                )

            if isinstance(self._orchestration_strategy, AgentOrchestrationStrategy):
                plan = orch_state.get("plan", [])
                agents_formatted = orch_state.get("agents", "")

                if turn > 1:
                    try:
                        entry = self._infra.get_state(instance_id)
                    except Exception:
                        logger.exception(
                            f"Failed to get workflow state for instance_id: {instance_id}"
                        )
                        raise
                    messages = (
                        getattr(entry, "messages", [])
                        if entry and hasattr(entry, "messages")
                        else []
                    )
                    plan = self.get_plan_from_messages(messages) or plan

                    if not ctx.is_replaying:
                        logger.info(
                            f"Loaded plan from state with {len(plan)} steps (turn {turn})"
                        )

                    if len(plan) == 0:
                        raise AgentError(
                            "No plan available; cannot continue orchestration."
                        )

                next_step_prompt = NEXT_STEP_PROMPT.format(
                    task=task,
                    agents=agents_formatted,
                    plan=plan,
                    next_step_schema=schemas.next_step,
                )
                next_step_response = yield ctx.call_activity(
                    self.call_llm,
                    input={
                        "instance_id": instance_id,
                        "task": next_step_prompt,
                        "time": ctx.current_utc_datetime.isoformat(),
                        "response_format": "NextStep",
                    },
                    retry_policy=self._retry_policy,
                )

                try:
                    parsed_content = json.loads(next_step_response.get("content", "{}"))
                    next_agent = parsed_content.get("next_agent")
                    instruction = parsed_content.get("instruction")
                    step_id = parsed_content.get("step")
                    substep_id = parsed_content.get("substep")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response content: {e}")
                    raise AgentError(f"Failed to parse next step from LLM: {e}")

                if not ctx.is_replaying:
                    logger.info(
                        f"Next step decided: agent={next_agent}, step={step_id}, "
                        f"substep={substep_id}, instruction={instruction}"
                    )

                is_valid = yield ctx.call_activity(
                    self._validate_next_step,
                    input={
                        "instance_id": instance_id,
                        "plan": self._convert_plan_objects_to_dicts(plan),
                        "step": step_id,
                        "substep": substep_id,
                    },
                    retry_policy=self._retry_policy,
                )

                if not is_valid:
                    if not ctx.is_replaying:
                        logger.warning(
                            f"Step {step_id}, substep {substep_id} not found in plan; skipping turn"
                        )
                    orch_state["verdict"] = "continue"
                    continue

                action = {
                    "agent": next_agent,
                    "instruction": instruction,
                    "metadata": {"step_id": step_id, "substep_id": substep_id},
                }

                orch_state["current_step_id"] = step_id
                orch_state["current_substep_id"] = substep_id
                orch_state["plan"] = plan

            else:
                action = yield ctx.call_activity(
                    self._select_next_action,
                    input={"state": orch_state, "turn": turn, "task": task},
                    retry_policy=self._retry_policy,
                )

            next_agent = action["agent"]
            instruction = action["instruction"]

            if not ctx.is_replaying:
                logger.info(
                    f"Turn {turn}: Selected agent '{next_agent}' with instruction: {instruction[:100]}..."
                )

            agents_metadata = self.list_team_agents(
                include_self=False, team=self.effective_team
            )
            agent_appid = agents_metadata[next_agent]["agent"]["appid"]

            result = yield ctx.call_child_workflow(
                workflow="agent_workflow",
                input={"task": instruction},
                app_id=agent_appid,
                retry_policy=self._retry_policy,
            )

            if not ctx.is_replaying:
                logger.info(
                    f"Turn {turn}: Agent '{next_agent}' responded: {result.get('content', '')[:100]}..."
                )

            if isinstance(self._orchestration_strategy, AgentOrchestrationStrategy):
                plan = orch_state.get("plan", [])
                step_id = orch_state.get("current_step_id")
                substep_id = orch_state.get("current_substep_id")

                target = find_step_in_plan(plan, step_id, substep_id)
                if target:
                    target["status"] = "completed"
                plan = update_step_statuses(plan)

                progress_prompt = PROGRESS_CHECK_PROMPT.format(
                    task=task,
                    plan=plan,
                    step=step_id,
                    substep=substep_id,
                    results=result.get("content", ""),
                    progress_check_schema=schemas.progress_check,
                )
                progress_response = yield ctx.call_activity(
                    self.call_llm,
                    input={
                        "instance_id": instance_id,
                        "task": progress_prompt,
                        "time": ctx.current_utc_datetime.isoformat(),
                        "response_format": "ProgressCheckOutput",
                    },
                    retry_policy=self._retry_policy,
                )

                progress = yield ctx.call_activity(
                    self._parse_progress,
                    input={
                        "content": progress_response.get("content", ""),
                        "instance_id": instance_id,
                        "plan_objects": plan,
                    },
                    retry_policy=self._retry_policy,
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
                        "instance_id": instance_id,
                        "plan_message": plan_message,
                        "time": ctx.current_utc_datetime.isoformat(),
                    },
                    retry_policy=self._retry_policy,
                )

                orch_state["plan"] = plan
                orch_state["verdict"] = verdict
                orch_state["last_agent"] = next_agent
                orch_state["last_result"] = result.get("content", "")

            else:
                process_result = yield ctx.call_activity(
                    self._process_orchestration_response,
                    input={
                        "state": orch_state,
                        "response": result,
                        "action": action,
                        "task": task,
                    },
                    retry_policy=self._retry_policy,
                )

                orch_state = process_result["updated_state"]
                verdict = process_result.get("verdict", "continue")

            if isinstance(self._orchestration_strategy, AgentOrchestrationStrategy):
                verdict = orch_state.get("verdict", "continue")
                should_continue = (verdict == "continue") and (
                    turn < self.execution.max_iterations
                )
            else:
                should_continue = yield ctx.call_activity(
                    self._should_continue_orchestration,
                    input={"state": orch_state, "turn": turn},
                    retry_policy=self._retry_policy,
                )

            if not should_continue:
                if not ctx.is_replaying:
                    logger.info(f"Orchestration stopping after turn {turn}")
                break

        if isinstance(self._orchestration_strategy, AgentOrchestrationStrategy):
            plan = orch_state.get("plan", [])
            verdict = orch_state.get("verdict", "continue")
            step_id = orch_state.get("current_step_id")
            substep_id = orch_state.get("current_substep_id")
            last_agent = orch_state.get("last_agent", "")
            last_result = orch_state.get("last_result", "")

            if verdict == "continue":
                verdict = "max_iterations_reached"

            summary_prompt = SUMMARY_GENERATION_PROMPT.format(
                task=task,
                verdict=verdict,
                plan=plan,
                step=step_id,
                substep=substep_id,
                agent=last_agent,
                result=last_result,
            )
            summary_response = yield ctx.call_activity(
                self.call_llm,
                input={
                    "instance_id": instance_id,
                    "task": summary_prompt,
                    "time": ctx.current_utc_datetime.isoformat(),
                },
                retry_policy=self._retry_policy,
            )

            final_message = {
                "role": "assistant",
                "content": summary_response.get("content", "Orchestration completed."),
                "name": self.name,
            }

        else:
            final_message = yield ctx.call_activity(
                self._finalize_orchestration,
                input={"state": orch_state, "task": task, "instance_id": instance_id},
                retry_policy=self._retry_policy,
            )

        if not ctx.is_replaying:
            logger.info(f"Orchestration workflow completed for instance {instance_id}")

        return final_message

    # ------------------------------------------------------------------
    # Strategy Delegation Activities
    # ------------------------------------------------------------------

    def _initialize_orchestration(self, ctx: Any, payload: dict) -> dict:
        """Initialize orchestration state via strategy.

        Args:
            ctx: Activity context
            payload: Dict with task, agents, instance_id

        Returns:
            Initial orchestration state from strategy
        """
        if not self._orchestration_strategy:
            raise AgentError("No orchestration strategy configured")

        return self._orchestration_strategy.initialize(
            ctx, payload["task"], payload["agents"]
        )

    def _select_next_action(self, ctx: Any, payload: dict) -> dict:
        """Select next agent and instruction via strategy.

        Args:
            ctx: Activity context
            payload: Dict with state, turn, task

        Returns:
            Action dict with agent, instruction, and metadata
        """
        if not self._orchestration_strategy:
            raise AgentError("No orchestration strategy configured")

        return self._orchestration_strategy.select_next_agent(
            ctx, payload["state"], payload["turn"]
        )

    def _process_orchestration_response(self, ctx: Any, payload: dict) -> dict:
        """Process agent response via strategy.

        Args:
            ctx: Activity context
            payload: Dict with state, response, action, task

        Returns:
            Dict with updated_state and verdict
        """
        if not self._orchestration_strategy:
            raise AgentError("No orchestration strategy configured")

        return self._orchestration_strategy.process_response(
            ctx, payload["state"], payload["response"]
        )

    def _should_continue_orchestration(self, ctx: Any, payload: dict) -> bool:
        """Check if orchestration should continue via strategy.

        Args:
            ctx: Activity context
            payload: Dict with state, turn

        Returns:
            True if should continue, False otherwise
        """
        if not self._orchestration_strategy:
            raise AgentError("No orchestration strategy configured")

        return self._orchestration_strategy.should_continue(
            payload["state"], payload["turn"], self.execution.max_iterations
        )

    def _finalize_orchestration(self, ctx: Any, payload: dict) -> dict:
        """Finalize orchestration and generate summary via strategy.

        Args:
            ctx: Activity context
            payload: Dict with state, task, instance_id

        Returns:
            Final message dict for caller
        """
        if not self._orchestration_strategy:
            raise AgentError("No orchestration strategy configured")

        return self._orchestration_strategy.finalize(ctx, payload["state"])

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
        if source == self.name:
            logger.debug("Agent %s ignoring self-originated broadcast.", self.name)
            return

        logger.info("Agent %s received broadcast from %s", self.name, source)
        logger.debug("Full broadcast message: %s", message)

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

    def get_plan_from_messages(
        self, messages: List[AgentWorkflowMessage]
    ) -> Optional[List[Dict[str, Any]]]:
        # Find all assistant messages with JSON content starting with {
        plan_messages = [
            m
            for m in messages
            if m.role == "assistant" and m.content.strip().startswith("{")
        ]

        # Get the LAST (most recent) plan message
        if not plan_messages:
            return None

        plan_msg = plan_messages[-1]  # Get the last one

        try:
            data = json.loads(plan_msg.content)
            return self._convert_plan_objects_to_dicts(
                [PlanStep(**obj) for obj in data.get("objects", [])]
            )
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
        Input/output/status/timestamps come from Dapr get_workflow.
        We only source, triggering_workflow_instance_id, trace_context.
        """
        instance_id = ctx.workflow_id
        if self.state_store and instance_id:
            self._infra.load_state(instance_id)
        trace_context = payload.get("trace_context")
        source = payload.get("source", "direct")
        triggering_instance = payload.get("triggering_workflow_instance_id")

        try:
            entry = self._infra.get_state(instance_id)
        except Exception:
            logger.exception(
                "Failed to get workflow state for instance_id: %s", instance_id
            )
            raise
        if entry is None:
            return

        entry.source = source
        entry.triggering_workflow_instance_id = triggering_instance
        entry.trace_context = trace_context
        self.save_state(instance_id)

    def call_llm(
        self,
        ctx: wf.WorkflowActivityContext,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ask the LLM to generate the next assistant message.

        Args:
            payload: Must contain 'instance_id'; may include 'task', 'time', and 'response_format'.

        Returns:
            Assistant message as a dict.

        Raises:
            AgentError: If the LLM call fails or yields no message.
        """
        # TODO(@sicoyle): i think i can use the instance_id in teh ctx instead here!!
        instance_id = payload.get("instance_id")
        task = payload.get("task")
        response_format_name = payload.get("response_format")

        response_model = None
        if response_format_name:
            model_map = {
                "IterablePlanStep": IterablePlanStep,
                "NextStep": NextStep,
                "ProgressCheckOutput": ProgressCheckOutput,
            }
            response_model = model_map.get(response_format_name)
            if response_model is None:
                logger.warning(
                    "Unknown response_format '%s', ignoring", response_format_name
                )

        # Load latest workflow state to ensure we have current data
        # TODO(@sicoyle): can i remove these calls????
        if self.state_store:
            self._infra.load_state(instance_id)

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

            # Skip printing for orchestrators' internal LLM calls
            if user_copy is not None and not self.orchestrator:
                self.text_formatter.print_message(
                    {str(k): v for k, v in user_copy.items()}
                )

        tools = self.get_llm_tools()
        generate_kwargs = {
            "messages": messages,
            "tools": tools,
        }
        if response_model is not None:
            generate_kwargs["response_format"] = response_model
        if tools and self.execution.tool_choice is not None:
            generate_kwargs["tool_choice"] = self.execution.tool_choice

        try:
            response = self.llm.generate(**generate_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise AgentError(str(exc)) from exc

        # Handle structured output response (Pydantic model) vs regular chat response
        if response_model is not None:
            # Structured output: response is the Pydantic model itself
            if hasattr(response, "model_dump"):
                # Response is the structured Pydantic object
                content = json.dumps(response.model_dump())
            else:
                # Fallback: try to serialize as-is
                content = json.dumps(response)

            assistant_message = {
                "role": "assistant",
                "content": content,
            }
        else:
            # Regular chat response
            assistant_message = response.get_message()
            if assistant_message is None:
                raise AgentError("LLM returned no assistant message.")
            assistant_message = assistant_message.model_dump()

        self._save_assistant_message(instance_id, assistant_message)
        # Skip printing for orchestrators' internal LLM calls
        if not self.orchestrator:
            self.text_formatter.print_message(assistant_message)
        self.save_state(instance_id)
        return assistant_message

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
        instance_id: str = payload.get("instance_id", "")
        tool_results_raw: List[Dict[str, Any]] = payload.get("tool_results", [])
        tool_results: List[ToolMessage] = [ToolMessage(**tr) for tr in tool_results_raw]

        try:
            entry = self._infra.get_state(instance_id)
        except Exception:
            logger.exception(
                f"Failed to get workflow state for instance_id: {instance_id}"
            )
            raise

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

            logger.debug(f"Added tool result {tool_call_id} to memory")

        self.save_state(instance_id)

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

    def summarize(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Summarize the conversation history and tool calls, then save the summary
        to the configured memory store (keyed by workflow instance id).

        Returns:
            Dict with "content" key holding the summary text, or empty dict if
            no memory/store or no conversation to summarize.
        """
        instance_id = ctx.workflow_id
        try:
            entry = self._infra.get_state(instance_id)
        except Exception:
            logger.exception(
                "Failed to get workflow state for instance_id: %s", instance_id
            )
            raise AgentError(
                f"Failed to get workflow state for instance_id: {instance_id}"
            )

        tool_history = getattr(entry, "tool_history", None) or []
        return self._summarize_conversation(instance_id, entry.messages, tool_history)

    # TODO(@sicoyle): I think we can rm this, but need to double check in follow up PR if dapr captures under the hood triggering workflow instance id.
    def finalize_workflow(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> None:
        """
        Finalize workflow state: persist triggering_workflow_instance_id if provided.
        Status/output/end_time come from Dapr get_workflow; we do not store them here.
        """
        instance_id = payload.get("instance_id")
        if self.state_store:
            self._infra.load_state(instance_id)

        triggering_workflow_instance_id = payload.get("triggering_workflow_instance_id")

        try:
            entry = self._infra.get_state(instance_id)
        except Exception:
            logger.exception(
                "Failed to get workflow state for instance_id: %s", instance_id
            )
            raise

        if not entry:
            logger.warning("Workflow state not found for instance_id: %s", instance_id)
            return

        entry.triggering_workflow_instance_id = triggering_workflow_instance_id
        self.save_state(instance_id)

    # ------------------------------------------------------------------
    # Orchestrator Activities
    # ------------------------------------------------------------------

    def _get_available_agents(self, ctx: wf.WorkflowActivityContext) -> Dict[str, Any]:
        """
        Return available agents metadata and formatted string.

        Args:
            ctx: The Dapr Workflow context.

        Returns:
            Dict with 'metadata' (dict) and 'formatted' (str) keys.
        """
        agents_metadata = self.get_agents_metadata(
            exclude_self=True, exclude_orchestrator=True, team=self.effective_team
        )
        if not agents_metadata:
            return {
                "metadata": {},
                "formatted": "No available agents to assign tasks.",
            }
        lines = []
        for name, meta in agents_metadata.items():
            role = meta["agent"].get("role", "Unknown role")
            goal = meta["agent"].get("goal", "Unknown")
            lines.append(f"- {name}: {role} (Goal: {goal})")
        return {
            "metadata": agents_metadata,
            "formatted": "\n".join(lines),
        }

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

    def _parse_progress(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
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
                logger.error(
                    f"Failed to parse progress check output: {e}. Content: {content}"
                )
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

        instance_id = payload.get("instance_id")
        if self.state_store:
            self._infra.load_state(instance_id)

        plan_message = payload.get("plan_message")

        try:
            entry = self._infra.get_state(instance_id)
        except Exception:
            logger.exception(
                f"Failed to get workflow state for instance_id: {instance_id}"
            )
            raise

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
                content=plan_message.get("content", ""), name=plan_message.get("name")
            ),
            instance_id,
        )

        self.save_state(instance_id)
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

        super().stop()

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
        # Primary entry point
        runtime.register_workflow(self.agent_workflow)
        runtime.register_workflow(self.broadcast_listener)

        # Standard agent activities
        runtime.register_activity(self.record_initial_entry)
        runtime.register_activity(self.call_llm)
        runtime.register_activity(self.run_tool)
        runtime.register_activity(self.save_tool_results)
        runtime.register_activity(self.broadcast_message_to_agents)
        runtime.register_activity(self.send_response_back)
        runtime.register_activity(self.summarize)
        runtime.register_activity(self.finalize_workflow)
        runtime.register_activity(self._get_available_agents)

        # Internal orchestration workflow and activities
        if self._orchestration_strategy:
            runtime.register_workflow(self.orchestration_workflow)

            if isinstance(self._orchestration_strategy, AgentOrchestrationStrategy):
                # Agent-based orchestration activities (plan-based with LLM)
                runtime.register_activity(self._validate_next_step)
                runtime.register_activity(self._parse_progress)
                runtime.register_activity(self._save_plan_message)
            else:
                # RoundRobin and Random orchestration activities
                runtime.register_activity(self._initialize_orchestration)
                runtime.register_activity(self._select_next_action)
                runtime.register_activity(self._process_orchestration_response)
                runtime.register_activity(self._should_continue_orchestration)
                runtime.register_activity(self._finalize_orchestration)
