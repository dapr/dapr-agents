from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

import dapr.ext.workflow as wf

from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    AgentExecutionConfig,
    WorkflowGrpcOptions,
)
from dapr_agents.agents.orchestrators.base import OrchestratorBase
from dapr_agents.agents.orchestrators.llm.configs import build_llm_state_bundle
from dapr_agents.agents.utils.text_printer import ColorTextFormatter
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.utils.defaults import get_default_llm
from dapr_agents.memory import ConversationDaprStateMemory, ConversationListMemory
from dapr_agents.types import AssistantMessage, UserMessage

logger = logging.getLogger(__name__)


class LLMOrchestratorBase(OrchestratorBase):
    """
    A base class for LLM-driven orchestrators. This class manages the memory, LLM client,
    and workflow state, and provides mechanisms for handling state persistence,
    compensation, and broadcasting messages across agents.

    Responsibilities:
        • Memory management (Dapr state-backed or in-memory).
        • Dependency injection for LLM clients.
        • Managing durable workflow state (messages, plan, finalization).
        • Broadcast messages and trigger actions via pub/sub and agent registry.
        • Compensation utilities for maintaining state consistency on failures.
    """

    def __init__(
        self,
        *,
        name: str = "LLMOrchestrator",
        pubsub: Optional[AgentPubSubConfig] = None,
        state: Optional[AgentStateConfig] = None,
        registry: Optional[AgentRegistryConfig] = None,
        execution: Optional[AgentExecutionConfig] = None,
        agent_metadata: Optional[Dict[str, Any]] = None,
        memory: Optional[AgentMemoryConfig] = None,
        llm: Optional[ChatClientBase] = None,
        workflow_grpc: Optional[WorkflowGrpcOptions] = None,
        runtime: Optional[wf.WorkflowRuntime] = None,
        workflow_client: Optional[wf.DaprWorkflowClient] = None,
        final_summary_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Initializes the LLMOrchestrator with the provided configurations.

        Args:
            name (str): Logical orchestrator name.
            pubsub (Optional[AgentPubSubConfig]): Dapr Pub/Sub configuration.
            state (Optional[AgentStateConfig]): State configuration for the orchestrator.
                Schema is automatically set to LLMWorkflowState/LLMWorkflowMessage.
            registry (Optional[AgentRegistryConfig]): Configuration for agent/team registry.
            agent_metadata (Optional[Dict[str, Any]]): Metadata to store alongside the registry entry.
            memory (Optional[AgentMemoryConfig]): Memory configuration for the orchestrator.
            llm (Optional[ChatClientBase]): LLM client instance.
            workflow_grpc (Optional[WorkflowGrpcOptions]): gRPC overrides for the workflow runtime channel.
            runtime (Optional[wf.WorkflowRuntime]): Workflow runtime configuration.
            workflow_client (Optional[wf.DaprWorkflowClient]): Dapr workflow client.
        """
        super().__init__(
            name=name,
            pubsub=pubsub,
            state=state,
            registry=registry,
            execution=execution,
            agent_metadata=agent_metadata,
            workflow_grpc=workflow_grpc,
            runtime=runtime,
            workflow_client=workflow_client,
            default_bundle=build_llm_state_bundle(),
            final_summary_callback=final_summary_callback,
        )

        # Memory wiring setup
        self._memory = memory or AgentMemoryConfig()
        if self._memory.store is None and state is not None:
            self._memory.store = ConversationDaprStateMemory(
                store_name=state.store.store_name,
                session_id=f"{self.name}-session",
            )
        self.memory = self._memory.store or ConversationListMemory()

        # Console formatting
        self._text_formatter = ColorTextFormatter()

        # LLM client initialization
        self.llm = llm or get_default_llm()

        # Initialize state if not present
        if not getattr(self, "state", None):
            self.state = {"instances": {}}
        else:
            self.state.setdefault("instances", {})

    @property
    def text_formatter(self) -> ColorTextFormatter:
        """Returns the text formatter used for console output."""
        return self._text_formatter

    @text_formatter.setter
    def text_formatter(self, formatter: ColorTextFormatter) -> None:
        """Sets a custom text formatter for console output."""
        self._text_formatter = formatter

    @staticmethod
    def _utcnow() -> datetime:
        """Returns the current UTC time as a timezone-aware datetime."""
        return datetime.now(timezone.utc)

    @staticmethod
    def _serialize_message(message: Any) -> Dict[str, Any]:
        """
        Serializes a message-like object into a dictionary for storage.

        Args:
            message (Any): The message object to serialize.

        Returns:
            Dict[str, Any]: The serialized message as a dictionary.

        Raises:
            TypeError: If the message type cannot be serialized.
        """
        if hasattr(message, "model_dump"):
            return message.model_dump()  # type: ignore[no-any-return]
        if isinstance(message, dict):
            return dict(message)
        if hasattr(message, "__dict__"):
            return dict(message.__dict__)
        raise TypeError(
            f"Unsupported message type for serialization: {type(message)!r}"
        )

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

    def _ensure_instance_row(
        self, instance_id: str, *, input_text: Optional[str] = None
    ) -> None:
        """
        Ensures an entry exists for the workflow instance in the state.

        This delegates to ensure_instance_exists() which uses the entry_factory
        from the state schema bundle to create proper model instances.

        Args:
            instance_id (str): The workflow instance ID.
            input_text (Optional[str]): The initial input text (if any) for the workflow.
        """
        container = self._get_entry_container()
        logger.debug(
            "_ensure_instance_row: container type=%s, instance_id=%s, exists=%s",
            type(container).__name__ if container else None,
            instance_id,
            instance_id in container if container else False,
        )
        if container and instance_id not in container:
            logger.debug(
                "_ensure_instance_row: Creating new instance via ensure_instance_exists"
            )
            # Use the parent class method which properly handles entry_factory
            self.ensure_instance_exists(
                instance_id=instance_id,
                input_value=input_text or "",
                triggering_workflow_instance_id=None,
                time=self._utcnow(),
            )
            # Check what was created
            entry = container.get(instance_id)
            logger.debug(
                "_ensure_instance_row: Created entry type=%s",
                type(entry).__name__ if entry else None,
            )

    def update_workflow_state(
        self,
        *,
        instance_id: str,
        message: Optional[Dict[str, Any]] = None,
        final_output: Optional[str] = None,
        plan: Optional[List[Dict[str, Any]]] = None,
        wf_time: Optional[str] = None,
    ) -> None:
        """
        Updates the state of the workflow for a given instance.

        Args:
            instance_id (str): The workflow instance ID.
            message (Optional[Dict[str, Any]]): A message to append to the history.
            final_output (Optional[str]): Final output of the workflow.
            plan (Optional[List[Dict[str, Any]]]): The current plan snapshot.
            wf_time (Optional[str]): Workflow time (ISO 8601 string).
        """
        self._ensure_instance_row(instance_id)

        container = self._get_entry_container()
        if not container or instance_id not in container:
            logger.error(
                "Cannot update state - instance %s not found in container", instance_id
            )
            return

        entry = container[instance_id]
        logger.info(
            "update_workflow_state: entry type=%s, hasattr(plan)=%s",
            type(entry).__name__,
            hasattr(entry, "plan"),
        )

        if plan is not None:
            logger.info(
                "Updating plan: entry type=%s, plan length=%d",
                type(entry).__name__,
                len(plan),
            )
            if hasattr(entry, "plan"):
                logger.debug("Entry is a Pydantic model, setting plan attribute")
                from dapr_agents.agents.orchestrators.llm.state import PlanStep

                entry.plan = [
                    PlanStep(**step_dict) if isinstance(step_dict, dict) else step_dict
                    for step_dict in plan
                ]  # type: ignore[attr-defined]
            else:
                # Fallback for dict-based state
                logger.info("Entry is a dict, setting plan key")
                entry["plan"] = plan

        if message is not None:
            msg = self._serialize_message(message)
            logger.info(
                "Processing message: entry type=%s, hasattr(messages)=%s",
                type(entry).__name__,
                hasattr(entry, "messages"),
            )

            if hasattr(entry, "messages"):
                # Entry is a Pydantic model - need to convert dict to message model
                if self._message_coercer:
                    msg_model = self._message_coercer(msg)
                else:
                    msg_model = self._message_dict_to_message_model(msg)
                logger.debug("Message model type: %s", type(msg_model).__name__)
                entry.messages.append(msg_model)  # type: ignore[attr-defined]
                entry.last_message = msg_model  # type: ignore[attr-defined]
            else:
                # Fallback for dict-based state
                logger.debug("Entry is dict, appending message dict directly")
                if "messages" not in entry:
                    entry["messages"] = []
                entry["messages"].append(msg)
                entry["last_message"] = msg

            try:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "assistant":
                    self.memory.add_message(
                        AssistantMessage(content=content, name=msg.get("name"))
                    )
                elif role == "user":
                    self.memory.add_message(
                        UserMessage(content=content, name=msg.get("name"))
                    )
            except Exception:
                logger.info("Failed to mirror message into memory.", exc_info=True)

        if final_output is not None:
            end_time_value = self._coerce_datetime(wf_time)

            if hasattr(entry, "output"):
                entry.output = final_output  # type: ignore[attr-defined]
                entry.end_time = end_time_value  # type: ignore[attr-defined]
            else:
                # Dict-based state fallback - store as ISO string
                entry["output"] = final_output
                entry["end_time"] = end_time_value.isoformat()

        self.save_state()

    async def rollback_workflow_initialization(self, instance_id: str) -> None:
        """Clears a partially-created plan for an instance."""
        try:
            container = self._get_entry_container()
            entry = container.get(instance_id) if container else None
            if entry:
                if hasattr(entry, "plan"):
                    entry.plan = []  # type: ignore[attr-defined]
                else:
                    entry["plan"] = []
                self.save_state()
                logger.debug("Rolled back workflow initialization for %s", instance_id)
        except Exception:
            logger.exception("Failed to rollback workflow initialization.")

    async def rollback_agent_trigger(
        self, instance_id: str, step_id: int, substep_id: Optional[float]
    ) -> None:
        """Reverts a step from `in_progress` back to `not_started`."""
        from dapr_agents.agents.orchestrators.llm.utils import find_step_in_plan

        try:
            container = self._get_entry_container()
            entry = container.get(instance_id) if container else None
            if entry:
                plan = (
                    getattr(entry, "plan", None)
                    if hasattr(entry, "plan")
                    else entry.get("plan", [])
                )
                step_entry = find_step_in_plan(plan, step_id, substep_id)
                if step_entry and step_entry.get("status") == "in_progress":
                    step_entry["status"] = "not_started"
                    self.update_workflow_state(instance_id=instance_id, plan=plan)
                    logger.debug(
                        "Rolled back agent trigger for %s (%s/%s)",
                        instance_id,
                        step_id,
                        substep_id,
                    )
        except Exception:
            logger.exception("Failed to rollback agent trigger.")

    async def rollback_agent_response_processing(
        self, instance_id: str, agent: str, step_id: int, substep_id: Optional[float]
    ) -> None:
        """Undo the last task history entry and revert `completed` -> `in_progress` if needed."""
        from dapr_agents.agents.orchestrators.llm.utils import find_step_in_plan

        try:
            container = self._get_entry_container()
            entry = container.get(instance_id) if container else None
            if entry:
                hist = (
                    getattr(entry, "task_history", None)
                    if hasattr(entry, "task_history")
                    else entry.get("task_history", [])
                )
                for i in range(len(hist) - 1, -1, -1):
                    t = hist[i]
                    if (
                        t.get("agent") == agent
                        and t.get("step") == step_id
                        and t.get("substep") == substep_id
                    ):
                        hist.pop(i)
                        break
                plan = (
                    getattr(entry, "plan", None)
                    if hasattr(entry, "plan")
                    else entry.get("plan", [])
                )
                step_entry = find_step_in_plan(plan, step_id, substep_id)
                if step_entry and step_entry.get("status") == "completed":
                    step_entry["status"] = "in_progress"
                    self.update_workflow_state(instance_id=instance_id, plan=plan)
            logger.debug(
                "Rolled back response processing for agent=%s step=%s substep=%s",
                agent,
                step_id,
                substep_id,
            )
        except Exception:
            logger.exception("Failed to rollback agent response processing.")

    async def rollback_workflow_finalization(self, instance_id: str) -> None:
        """Clear output and end time if finalization failed."""
        try:
            container = self._get_entry_container()
            entry = container.get(instance_id) if container else None
            if entry:
                if hasattr(entry, "output"):
                    entry.output = None  # type: ignore[attr-defined]
                    entry.end_time = None  # type: ignore[attr-defined]
                else:
                    entry["output"] = None
                    entry["end_time"] = None
                self.save_state()
                logger.info("Rolled back workflow finalization for %s", instance_id)
        except Exception:
            logger.exception("Failed to rollback workflow finalization.")

    async def ensure_workflow_state_consistency(self, instance_id: str) -> None:
        """Ensure that the instance row exists and contains the required keys."""
        try:
            self._ensure_instance_row(instance_id)
            container = self._get_entry_container()
            entry = container.get(instance_id) if container else None
            if entry:
                if hasattr(entry, "plan"):
                    # Pydantic model - fields should already exist
                    pass
                else:
                    # Dict-based state
                    entry.setdefault("plan", [])
                    entry.setdefault("messages", [])
                    entry.setdefault("task_history", [])
            self.save_state()
        except Exception:
            logger.exception("Failed to ensure workflow state consistency.")

    async def compensate_failed_activity(
        self,
        *,
        instance_id: str,
        failed_activity: str,
        activity_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generic compensator for multi-step activities."""
        actions: List[str] = []
        try:
            if failed_activity == "initialize_workflow_with_plan":
                await self.rollback_workflow_initialization(instance_id)
                actions.append("cleared_partial_plan")
            elif failed_activity == "execute_agent_task_with_progress_tracking":
                await self.rollback_agent_trigger(
                    instance_id,
                    activity_context.get("step_id"),
                    activity_context.get("substep_id"),
                )
                actions.append("reverted_step_status")
            elif failed_activity == "process_agent_response_with_progress":
                await self.rollback_agent_response_processing(
                    instance_id,
                    activity_context.get("agent"),
                    activity_context.get("step_id"),
                    activity_context.get("substep_id"),
                )
                actions.append("reverted_response_processing")
            elif failed_activity == "finalize_workflow_with_summary":
                await self.rollback_workflow_finalization(instance_id)
                actions.append("reverted_finalization")

            await self.ensure_workflow_state_consistency(instance_id)
            return {
                "status": "compensated",
                "failed_activity": failed_activity,
                "compensation_actions": actions,
            }
        except Exception as exc:
            logger.exception("Compensation failed for %s", failed_activity)
            return {
                "status": "compensation_failed",
                "failed_activity": failed_activity,
                "error": str(exc),
            }

    async def execute_with_compensation(
        self, activity_coro, *, activity_name: str, instance_id: str, **kwargs: Any
    ) -> Any:
        """Execute an async activity and auto-compensate on failure."""
        try:
            return await activity_coro
        except Exception as exc:
            logger.error("Activity %s failed: %s", activity_name, exc)
            ctx = {"instance_id": instance_id, "error": str(exc), **kwargs}
            result = await self.compensate_failed_activity(
                instance_id=instance_id,
                failed_activity=activity_name,
                activity_context=ctx,
            )
            if result.get("status") != "compensated":
                logger.error("Compensation failed: %s", result)
            raise

    async def update_task_history_internal(
        self,
        *,
        instance_id: str,
        agent: str,
        step: int,
        substep: Optional[float],
        results: Dict[str, Any],
        plan: List[Dict[str, Any]],
    ) -> None:
        """
        Append a task result to workflow messages and task history, then persist plan pointer.

        Args:
            instance_id: Workflow instance id.
            agent: Agent producing the results.
            step: Plan step id.
            substep: Plan substep id (if any).
            results: Message-like result dict from the agent.
            plan: Current plan snapshot (dicts).

        Raises:
            ValueError: If the instance row does not exist (unexpected).
        """
        from dapr_agents.agents.orchestrators.llm.state import TaskResult

        logger.debug(
            "Updating task history for %s at step %s, substep %s (instance=%s)",
            agent,
            step,
            substep,
            instance_id,
        )

        # Store the agent's response in the message history
        self.update_workflow_state(instance_id=instance_id, message=results)

        # Retrieve workflow entry from container
        container = self._get_entry_container()
        entry = container.get(instance_id) if container else None
        if not entry:
            msg = f"No workflow entry for instance {instance_id}"
            raise ValueError(msg)

        # Create a TaskResult object
        task_result = TaskResult(
            agent=agent,
            step=step,
            substep=substep,
            result=results.get("content", ""),
        )

        # Append the result to task history
        if hasattr(entry, "task_history"):
            if not hasattr(entry.task_history, "append"):
                entry.task_history = []  # type: ignore[attr-defined]
            # Store TaskResult model instance directly instead of dict to avoid serialization warnings
            entry.task_history.append(task_result)  # type: ignore[attr-defined]
        else:
            entry.setdefault("task_history", []).append(
                task_result.model_dump(mode="json")
            )

        # Get current plan from entry
        current_plan = (
            getattr(entry, "plan", None)
            if hasattr(entry, "plan")
            else entry.get("plan", plan)
        )

        # Persist state with updated plan
        self.update_workflow_state(instance_id=instance_id, plan=current_plan)

    async def trigger_agent_internal(
        self,
        *,
        instance_id: str,
        name: str,
        step: int,
        substep: Optional[float],
        instruction: str,
        plan: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Mark the referenced (step/substep) as in_progress, persist, and send a directed trigger.

        Args:
            instance_id: Workflow instance id.
            name: Target agent name to trigger.
            step: Plan step id.
            substep: Plan substep id (if any).
            instruction: Instruction to send the target agent.
            plan: Current plan snapshot (dicts).

        Returns:
            The updated plan (with status changes applied).

        Raises:
            ValueError: If the (step, substep) reference does not exist in the plan.
            RuntimeError: If sending the trigger fails due to pub/sub issues.
        """
        from dapr_agents.agents.schemas import TriggerAction
        from dapr_agents.agents.orchestrators.llm.utils import (
            find_step_in_plan,
            update_step_statuses,
        )
        from dapr_agents.workflow.utils.pubsub import send_message_to_agent

        logger.info(
            "Triggering agent %s for step %s/%s (instance=%s)",
            name,
            step,
            substep,
            instance_id,
        )

        # Ensure the step or substep exists
        step_entry = find_step_in_plan(plan, step, substep)
        if not step_entry:
            if substep is not None:
                msg = f"Substep {substep} in Step {step} not found in the current plan."
            else:
                msg = f"Step {step} not found in the current plan."
            raise ValueError(msg)

        # Mark step or substep as "in_progress"
        step_entry["status"] = "in_progress"
        logger.debug("Marked step %s, substep %s as 'in_progress'", step, substep)

        # Apply global status updates to maintain consistency
        updated_plan = update_step_statuses(plan)

        # Save updated plan state
        self.update_workflow_state(instance_id=instance_id, plan=updated_plan)

        # Get agents metadata for pub/sub
        agents_metadata = self.list_team_agents(
            include_self=False, team=self.effective_team()
        )

        # Send message to agent with specific task instruction
        trigger = TriggerAction(task=instruction, workflow_instance_id=instance_id)

        async def _send() -> None:
            await send_message_to_agent(
                source=self.name,
                target_agent=name,
                message=trigger,
                agents_metadata=agents_metadata,
            )

        try:
            import asyncio

            loop = asyncio.get_running_loop()
            await _send() if loop.is_running() else asyncio.run(_send())
        except Exception:
            logger.exception("Failed to send trigger to agent %s", name)
            raise

        return updated_plan

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
        from dapr_agents.agents.orchestrators.llm.utils import (
            find_step_in_plan,
            restructure_plan,
            update_step_statuses,
        )

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
        self.update_workflow_state(instance_id=instance_id, plan=plan)

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
        from dapr_agents.agents.orchestrators.llm.utils import find_step_in_plan

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
        self.update_workflow_state(
            instance_id=instance_id,
            final_output=summary,
            wf_time=wf_time,
        )

        logger.info(
            "Workflow %s finalized with verdict '%s'",
            instance_id,
            verdict,
        )
