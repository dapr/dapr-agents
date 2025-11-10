from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import dapr.ext.workflow as wf
from durabletask import task as dt_task

from dapr_agents.agents.configs import AgentExecutionConfig
from dapr_agents.agents.orchestrators.llm.base import LLMOrchestratorBase
from dapr_agents.agents.schemas import (
    AgentTaskResponse,
    BroadcastMessage,
    TriggerAction,
)
from dapr_agents.workflow.decorators.routers import message_router
from dapr_agents.workflow.runners.agent import workflow_entry
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
from dapr_agents.agents.orchestrators.llm.utils import find_step_in_plan
from dapr_agents.workflow.utils.pubsub import broadcast_message

logger = logging.getLogger(__name__)


class LLMOrchestrator(LLMOrchestratorBase):
    """
    LLM-driven orchestrator that dynamically selects the next agent based on context and plan.
    Interacts with agents in a multi-step workflow, using an LLM to decide the next step,
    validates and triggers agents, and handles responses. Ensures steps are executed in order,
    checks for progress, and finalizes the workflow with a summary.
    """

    def __init__(
        self,
        *,
        name: str = "LLMOrchestrator",
        timeout_seconds: int = 60,
        execution: Optional[AgentExecutionConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the orchestrator with the provided configuration parameters.

        Args:
            name (str): Logical name of the orchestrator.
            timeout_seconds (int): Timeout duration for awaiting agent responses (in seconds).
        """
        super().__init__(name=name, execution=execution, **kwargs)
        self.timeout = max(1, int(timeout_seconds))

    def register_workflows(self, runtime: wf.WorkflowRuntime) -> None:
        """
        Registers workflows and activities to the provided Dapr WorkflowRuntime.
        """
        runtime.register_workflow(self.llm_orchestrator_workflow)
        runtime.register_workflow(self.route_agent_response)
        runtime.register_activity(self._broadcast_activity)
        runtime.register_activity(self._validate_next_step)
        runtime.register_activity(self._get_available_agents)
        runtime.register_activity(self._initialize_workflow_with_plan)
        runtime.register_activity(self._generate_next_step)
        runtime.register_activity(self._execute_agent_task_with_progress_tracking)
        runtime.register_activity(self._process_agent_response_with_progress)
        runtime.register_activity(self._finalize_workflow_with_summary)

    @workflow_entry
    @message_router(message_model=TriggerAction)
    def llm_orchestrator_workflow(
        self, ctx: wf.DaprWorkflowContext, message: Dict[str, Any]
    ):
        """
        Orchestrates the workflow, handling up to `self.execution.max_iterations` turns using an LLM to choose the next step/agent.
        """
        task_text: Optional[str] = message.get("task")
        parent_id: Optional[str] = message.get("workflow_instance_id")
        instance_id = ctx.instance_id
        final_summary: Optional[str] = None

        # Ensure the instance exists in the state model
        self.ensure_instance_exists(
            instance_id=instance_id,
            input_value=task_text or "",
            triggering_workflow_instance_id=parent_id,
            time=ctx.current_utc_datetime,
        )

        for turn in range(1, self.execution.max_iterations + 1):
            if not ctx.is_replaying:
                logger.info(
                    "LLM turn %d/%d (instance=%s)",
                    turn,
                    self.execution.max_iterations,
                    instance_id,
                )

            # Discover available agents
            agents = yield ctx.call_activity(self._get_available_agents)

            # Turn 1: initialize plan & broadcast
            if turn == 1:
                init = yield ctx.call_activity(
                    self._initialize_workflow_with_plan,
                    input={
                        "instance_id": instance_id,
                        "task": task_text or "",
                        "agents": agents,
                        "wf_time": ctx.current_utc_datetime.isoformat(),
                    },
                )
                plan = init["plan"]
                if not ctx.is_replaying:
                    logger.info(
                        "Received plan from initialization with %d steps", len(plan)
                    )
                initial_message = init["message"]

                # Broadcast the initial plan to all agents
                if not ctx.is_replaying:
                    logger.info(
                        "Broadcasting initial plan with %d steps to all agents",
                        len(plan),
                    )
                yield ctx.call_activity(
                    self._broadcast_activity,
                    input={"message": initial_message},
                )
                if not ctx.is_replaying:
                    logger.info("Initial plan broadcast completed")
            else:
                plan = list(
                    self.state.get("instances", {}).get(instance_id, {}).get("plan", [])
                )
                if not ctx.is_replaying:
                    logger.info(
                        "Loaded plan from state with %d steps (turn %d)",
                        len(plan),
                        turn,
                    )

            # Fallback: if plan is empty/None, try reading from state
            if not plan:
                plan = list(
                    self.state.get("instances", {}).get(instance_id, {}).get("plan", [])
                )
                if not ctx.is_replaying:
                    logger.warning(
                        "Plan was empty, fallback loaded %d steps from state", len(plan)
                    )

            # Ask LLM for next step/agent
            next_step = yield ctx.call_activity(
                self._generate_next_step,
                input={
                    "task": task_text or "",
                    "agents": agents,
                    "plan": json.dumps(
                        self._convert_plan_objects_to_dicts(plan), indent=2
                    ),
                    "next_step_schema": schemas.next_step,
                },
            )

            next_agent = next_step["next_agent"]
            instruction = next_step["instruction"]
            step_id = next_step.get("step")
            substep_id = next_step.get("substep")

            # Validate the next step
            is_valid = yield ctx.call_activity(
                self._validate_next_step,
                input={
                    "instance_id": instance_id,
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

                result = yield ctx.call_activity(
                    self._execute_agent_task_with_progress_tracking,
                    input={
                        "instance_id": instance_id,
                        "next_agent": next_agent,
                        "step_id": step_id,
                        "substep_id": substep_id,
                        "instruction": instruction,
                        "task": task_text or "",
                        "plan_objects": self._convert_plan_objects_to_dicts(plan),
                    },
                )
                plan = result["plan"]

                # Await response or timeout
                event_task = ctx.wait_for_external_event("AgentTaskResponse")
                timeout_task = ctx.create_timer(timedelta(seconds=self.timeout))
                winner = yield dt_task.when_any([event_task, timeout_task])

                if winner == timeout_task:
                    if not ctx.is_replaying:
                        logger.warning(
                            "Turn %d timed out waiting for agent response (instance=%s)",
                            turn,
                            instance_id,
                        )
                    task_results = {
                        "name": "timeout",
                        "content": "⏰ Timeout occurred. Continuing...",
                    }
                else:
                    task_results = yield event_task
                    # Normalize
                    if hasattr(task_results, "model_dump"):
                        task_results = task_results.model_dump()
                    elif not isinstance(task_results, dict):
                        task_results = dict(task_results)

                    if not ctx.is_replaying:
                        self.print_interaction(
                            source_agent_name=task_results.get("name", "agent"),
                            target_agent_name=self.name,
                            message=task_results.get("content", ""),
                        )
                processed = yield ctx.call_activity(
                    self._process_agent_response_with_progress,
                    input={
                        "instance_id": instance_id,
                        "agent": next_agent,
                        "step_id": step_id,
                        "substep_id": substep_id,
                        "task_results": task_results,
                        "task": task_text or "",
                        "plan_objects": self._convert_plan_objects_to_dicts(plan),
                    },
                )
                plan = processed["plan"]
                verdict = processed["verdict"]
            else:
                verdict = "continue"
                task_results = {
                    "name": self.name,
                    "role": "user",
                    "content": f"Step {step_id}, substep {substep_id} not found. Adjusting workflow…",
                }

            if verdict != "continue" or turn == self.execution.max_iterations:
                final_summary = yield ctx.call_activity(
                    self._finalize_workflow_with_summary,
                    input={
                        "instance_id": instance_id,
                        "task": task_text or "",
                        "verdict": verdict
                        if verdict != "continue"
                        else "max_iterations_reached",
                        "plan_objects": self._convert_plan_objects_to_dicts(plan),
                        "step_id": step_id,
                        "substep_id": substep_id,
                        "agent": next_agent if is_valid else self.name,
                        "result": task_results["content"],
                        "wf_time": ctx.current_utc_datetime.isoformat(),
                    },
                )
                if not ctx.is_replaying:
                    logger.info("Workflow %s finalized.", instance_id)
                return final_summary
            else:
                task_text = task_results["content"]

        raise RuntimeError(f"{self.name} workflow {instance_id} exited without summary")

    @message_router(message_model=AgentTaskResponse)
    def route_agent_response(
        self, ctx: wf.DaprWorkflowContext, message: Dict[str, Any]
    ) -> None:
        """Route AgentTaskResponse messages into the running workflow."""
        instance_id = message.get("workflow_instance_id")
        if not instance_id:
            logger.error("AgentTaskResponse missing workflow_instance_id; ignoring.")
            return
        try:
            self.raise_workflow_event(
                instance_id=instance_id,
                event_name="AgentTaskResponse",
                data=message,
            )
        except RuntimeError:
            return

    # ------------------------------------------------------------------
    # Activities
    # ------------------------------------------------------------------

    def _broadcast_activity(
        self,
        ctx: wf.WorkflowActivityContext,
        payload: Dict[str, Any],
    ) -> None:
        """Broadcast a message to all agents (if a broadcast topic is configured)."""
        message = payload.get("message", {})
        if not isinstance(message, dict):
            logger.warning(
                "Skipping broadcast: payload is not a dict, type=%s",
                type(message).__name__,
            )
            return
        if not self.broadcast_topic_name:
            logger.warning(
                "Skipping broadcast: no broadcast topic configured (broadcast_topic_name=%s)",
                self.broadcast_topic_name,
            )
            return

        logger.info(
            "Broadcasting message from %s to topic %s",
            self.name,
            self.broadcast_topic_name,
        )

        try:
            agents_metadata = self.list_team_agents(
                include_self=False, team=self.effective_team()
            )
            logger.info(
                "Found %d agents to broadcast to: %s",
                len(agents_metadata),
                list(agents_metadata.keys()),
            )
        except Exception:
            logger.exception("Unable to load agents metadata; broadcast aborted.")
            return

        message["role"] = message.get("role", "user")
        message["name"] = self.name
        broadcast_payload = BroadcastMessage(**message)

        async def _broadcast() -> None:
            await broadcast_message(
                message=broadcast_payload,
                broadcast_topic=self.broadcast_topic_name,  # type: ignore[union-attr]
                message_bus=self.message_bus_name,  # type: ignore[union-attr]
                source=self.name,
                agents_metadata=agents_metadata,
            )

        try:
            self._run_asyncio_task(_broadcast())
        except Exception:  # noqa: BLE001
            logger.exception("Failed to publish broadcast message.")

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
            role = meta.get("role", "Unknown role")
            goal = meta.get("goal", "Unknown")
            lines.append(f"- {name}: {role} (Goal: {goal})")
        return "\n".join(lines)

    def _initialize_workflow_with_plan(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate (or rehydrate) a plan.

        Args:
            ctx: The Dapr Workflow context.
            payload: The input containing instance ID, task details, available agents, and workflow time.

        Returns:
            A dictionary containing the plan and message to broadcast.
        """
        instance_id = payload["instance_id"]
        task = payload["task"]
        agents = payload["agents"]
        wf_time = payload["wf_time"]

        # Use flexible container accessor (supports custom state layouts)
        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None

        # Check if THIS instance already has a plan (from a previous turn/replay)
        plan_dicts: List[Dict[str, Any]]
        existing_plan = getattr(entry, "plan", None) if entry else None

        if existing_plan:
            logger.info(
                "Reusing existing plan with %d steps from instance %s",
                len(existing_plan),
                instance_id,
            )
            # Convert Plan Step objects to dicts (existing_plan could be List[PlanStep] or List[dict])
            plan_dicts = self._convert_plan_objects_to_dicts(existing_plan)
        else:
            logger.info("Generating new plan for task: %s", task[:100])
            response = self.llm.generate(
                messages=[
                    {
                        "role": "user",
                        "content": TASK_PLANNING_PROMPT.format(
                            task=task, agents=agents, plan_schema=schemas.plan
                        ),
                    }
                ],
                response_format=IterablePlanStep,
            )
            response_dict = response.model_dump()
            plan_objects = [PlanStep(**d) for d in response_dict.get("objects", [])]
            plan_dicts = self._convert_plan_objects_to_dicts(plan_objects)
            logger.info("Generated new plan with %d steps", len(plan_dicts))
            logger.debug("Plan details: %s", json.dumps(plan_dicts, indent=2))

        # Persist and broadcast
        self.update_workflow_state(
            instance_id=instance_id, plan=plan_dicts, wf_time=wf_time
        )
        logger.info(
            "Persisted plan with %d steps to state for instance %s",
            len(plan_dicts),
            instance_id,
        )

        formatted_message = TASK_INITIAL_PROMPT.format(
            task=task, agents=agents, plan=json.dumps(plan_dicts, indent=2)
        )
        initial_message = {"role": "user", "content": formatted_message}
        logger.info(
            "Returning plan with %d steps from initialization activity", len(plan_dicts)
        )
        return {"plan": plan_dicts, "message": initial_message}

    def _generate_next_step(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ask the LLM which agent acts next, with instruction and step ids."""
        prompt = NEXT_STEP_PROMPT.format(
            task=payload["task"],
            agents=payload["agents"],
            plan=payload["plan"],
            next_step_schema=payload["next_step_schema"],
        )
        resp = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            response_format=NextStep,
            structured_mode="json",
        )
        if hasattr(resp, "choices") and resp.choices:
            data = resp.choices[0].message.content
            return dict(NextStep(**json.loads(data)))
        # Fallback if your LLM client returns a pydantic instance directly
        if isinstance(resp, NextStep):
            return resp.model_dump()
        return dict(resp)

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

    def _execute_agent_task_with_progress_tracking(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Mark step in_progress, persist, and trigger the agent with an InternalTriggerAction.
        """

        async def _execute() -> List[Dict[str, Any]]:
            return await self.execute_with_compensation(
                self.trigger_agent_internal(
                    instance_id=payload["instance_id"],
                    name=payload["next_agent"],
                    step=payload["step_id"],
                    substep=payload["substep_id"],
                    instruction=payload["instruction"],
                    plan=list(payload["plan_objects"]),
                ),
                activity_name="execute_agent_task_with_progress_tracking",
                instance_id=payload["instance_id"],
                step_id=payload["step_id"],
                substep_id=payload["substep_id"],
            )

        updated_plan = self._run_asyncio_task(_execute())
        return {"plan": updated_plan, "status": "agent_triggered"}

    def _process_agent_response_with_progress(
        self, ctx: wf.WorkflowActivityContext, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Append the agent's response, ask LLM for progress verdict/updates, and persist updated plan.
        """
        instance_id = payload["instance_id"]
        agent = payload["agent"]
        step_id = payload["step_id"]
        substep_id = payload["substep_id"]
        task = payload["task"]
        plan_objects = list(payload["plan_objects"])
        task_results = dict(payload["task_results"])

        async def _process() -> Dict[str, Any]:
            try:
                await self.update_task_history_internal(
                    instance_id=instance_id,
                    agent=agent,
                    step=step_id,
                    substep=substep_id,
                    results=task_results,
                    plan=plan_objects,
                )

                progress_prompt = PROGRESS_CHECK_PROMPT.format(
                    task=task,
                    plan=json.dumps(
                        self._convert_plan_objects_to_dicts(plan_objects), indent=2
                    ),
                    step=step_id,
                    substep=substep_id if substep_id is not None else "N/A",
                    results=task_results["content"],
                    progress_check_schema=schemas.progress_check,
                )
                progress_resp = self.llm.generate(
                    messages=[{"role": "user", "content": progress_prompt}],
                    response_format=ProgressCheckOutput,
                    structured_mode="json",
                )
                if hasattr(progress_resp, "choices") and progress_resp.choices:
                    data = progress_resp.choices[0].message.content
                    progress = ProgressCheckOutput(**json.loads(data))
                elif isinstance(progress_resp, ProgressCheckOutput):
                    progress = progress_resp
                else:
                    # Best-effort parse
                    progress = ProgressCheckOutput(
                        **(progress_resp if isinstance(progress_resp, dict) else {})
                    )

                status_updates = [
                    (u.model_dump() if hasattr(u, "model_dump") else u)
                    for u in (progress.plan_status_update or [])
                ]
                plan_updates = [
                    (u.model_dump() if hasattr(u, "model_dump") else u)
                    for u in (progress.plan_restructure or [])
                ]

                if status_updates or plan_updates:
                    updated_plan = await self.update_plan_internal(
                        instance_id=instance_id,
                        plan=plan_objects,
                        status_updates=status_updates,
                        plan_updates=plan_updates,
                    )
                else:
                    updated_plan = plan_objects

                return {
                    "plan": updated_plan,
                    "verdict": progress.verdict,
                    "status_updates": status_updates,
                    "plan_updates": plan_updates,
                    "status": "success",
                }

            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to process agent response: %s", exc)
                await self.rollback_agent_response_processing(
                    instance_id, agent, step_id, substep_id
                )
                self.update_workflow_state(
                    instance_id=instance_id,
                    message={
                        "name": agent,
                        "role": "system",
                        "content": f"Failed to process agent response: {exc}",
                        "step": step_id,
                        "substep": substep_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return {
                    "plan": plan_objects,
                    "verdict": "failed",
                    "status_updates": [],
                    "plan_updates": [],
                    "status": "failed",
                }

        return self._run_asyncio_task(_process())

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
