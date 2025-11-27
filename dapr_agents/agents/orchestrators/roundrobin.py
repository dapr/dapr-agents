from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, Optional, Callable

import dapr.ext.workflow as wf
from durabletask import task as dt_task

from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    AgentExecutionConfig,
    WorkflowGrpcOptions,
)
from dapr_agents.agents.orchestrators.base import OrchestratorBase
from dapr_agents.agents.schemas import (
    AgentTaskResponse,
    BroadcastMessage,
    TriggerAction,
)
from dapr_agents.workflow.decorators.routers import message_router
from dapr_agents.workflow.decorators import workflow_entry
from dapr_agents.workflow.utils.pubsub import broadcast_message, send_message_to_agent

logger = logging.getLogger(__name__)


class RoundRobinOrchestrator(OrchestratorBase):
    """
    Orchestrator that selects agents in strict round-robin order each turn.
    - Turn 1 broadcasts the initial task (if provided).
    - Each turn triggers the next agent in a deterministic order.
    - Waits for response or times out.
    - Feeds the response content into the next turn; returns last content.
    """

    def __init__(
        self,
        *,
        name: str = "RoundRobinOrchestrator",
        pubsub: Optional[AgentPubSubConfig] = None,
        state: Optional[AgentStateConfig] = None,
        registry: Optional[AgentRegistryConfig] = None,
        execution: Optional[AgentExecutionConfig] = None,
        agent_metadata: Optional[Dict[str, Any]] = None,
        workflow_grpc: Optional[WorkflowGrpcOptions] = None,
        timeout_seconds: int = 60,
        runtime: Optional[wf.WorkflowRuntime] = None,
        final_summary_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(
            name=name,
            pubsub=pubsub,
            state=state,
            registry=registry,
            execution=execution,
            agent_metadata=agent_metadata,
            workflow_grpc=workflow_grpc,
            runtime=runtime,
            final_summary_callback=final_summary_callback,
        )
        self.timeout = max(1, timeout_seconds)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_workflows(self, runtime: wf.WorkflowRuntime) -> None:
        """Register workflows and activities."""
        runtime.register_workflow(self.round_robin_workflow)
        runtime.register_workflow(self.process_agent_response)
        runtime.register_activity(self._process_input_activity)
        runtime.register_activity(self._broadcast_activity)
        runtime.register_activity(self._select_next_speaker_activity)
        runtime.register_activity(self._trigger_agent_activity)

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------
    @workflow_entry
    @message_router(message_model=TriggerAction)
    def round_robin_workflow(self, ctx: wf.DaprWorkflowContext, message: dict):
        """
        Drive round-robin loop for up to `self.execution.max_iterations`.
        """
        task = message.get("task")
        instance_id = ctx.instance_id
        final_output: Optional[str] = None

        for turn in range(1, self.execution.max_iterations + 1):
            if not ctx.is_replaying:
                logger.info(
                    "Round-robin turn %d/%d (instance=%s)",
                    turn,
                    self.execution.max_iterations,
                    instance_id,
                )

            # Turn 1: normalize + broadcast initial task
            if turn == 1 and task is not None:
                initial_message = yield ctx.call_activity(
                    self._process_input_activity,
                    input={"task": task},
                )
                if not ctx.is_replaying:
                    self.print_interaction(
                        source_agent_name=initial_message.get("name", "user"),
                        target_agent_name=self.name,
                        message=initial_message.get("content", ""),
                    )
                yield ctx.call_activity(
                    self._broadcast_activity,
                    input={"message": initial_message},
                )

            # Select next agent (round-robin)
            selected_agent = yield ctx.call_activity(
                self._select_next_speaker_activity,
                input={"turn": turn},
            )
            if not ctx.is_replaying:
                logger.info("Selected '%s' for turn %d", selected_agent, turn)

            # Trigger agent
            if not ctx.is_replaying:
                self.print_interaction(
                    source_agent_name=self.name,
                    target_agent_name=selected_agent,
                    message="TriggerAction",
                )
            yield ctx.call_activity(
                self._trigger_agent_activity,
                input={"name": selected_agent, "instance_id": instance_id},
            )

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
                result = {
                    "name": "timeout",
                    "content": "⏰ Timeout occurred. Continuing...",
                }
            else:
                result = yield event_task
                # Normalize
                if hasattr(result, "model_dump"):
                    result = result.model_dump()
                elif not isinstance(result, dict):
                    result = dict(result)

                if not ctx.is_replaying:
                    self.print_interaction(
                        source_agent_name=result.get("name", "agent"),
                        target_agent_name=self.name,
                        message=result.get("content", ""),
                    )

            if turn == self.execution.max_iterations:
                final_output = result.get("content", "")
                break

            task = result.get("content")

        if final_output is None:
            raise RuntimeError(
                "RoundRobin workflow completed without producing a final output."
            )

        self._invoke_final_summary_callback(final_output)
        return final_output

    @message_router(message_model=AgentTaskResponse)
    def process_agent_response(
        self, ctx: wf.DaprWorkflowContext, message: Dict[str, Any]
    ) -> None:
        """
        Route agent responses back into the workflow via an external event.
        """
        instance_id = (message or {}).get("workflow_instance_id")
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
            return  # already logged

    # ------------------------------------------------------------------
    # Activities
    # ------------------------------------------------------------------
    def _process_input_activity(
        self,
        ctx: wf.WorkflowActivityContext,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize the initial task into a user message dict."""
        task = payload.get("task") or ""
        return {"role": "user", "name": "user", "content": task}

    def _broadcast_activity(
        self,
        ctx: wf.WorkflowActivityContext,
        payload: Dict[str, Any],
    ) -> None:
        """Broadcast a message to all agents (if a broadcast topic is configured)."""
        message = payload.get("message", {})
        if not isinstance(message, dict):
            logger.debug("Skipping broadcast: payload is not a dict.")
            return
        if not self.broadcast_topic_name:
            logger.debug("Skipping broadcast: no broadcast topic configured.")
            return

        try:
            agents_metadata = self.list_team_agents(
                include_self=False, team=self.effective_team()
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
        except Exception:
            logger.exception("Failed to publish broadcast message.")

    def _select_next_speaker_activity(
        self,
        ctx: wf.WorkflowActivityContext,
        payload: Dict[str, Any],
    ) -> str:
        """
        Deterministic round-robin selection based on turn number.

        TODO: This method selects from all registered agents in the team, including
        orchestrators. In the future, we may want to add health checks or availability
        checks to ensure selected agents are actually running and ready to receive triggers.
        """
        turn = int(payload.get("turn", 1))
        try:
            agents_metadata = self.list_team_agents(
                include_self=False, team=self.effective_team()
            )
        except Exception as exc:
            logger.error("Unable to load agents metadata: %s", exc)
            raise

        if not agents_metadata:
            raise ValueError("No agents available for selection.")

        # Deterministic ordering across runs/replay
        names = sorted(agents_metadata.keys())
        idx = (max(1, turn) - 1) % len(names)
        return names[idx]

    def _trigger_agent_activity(
        self,
        ctx: wf.WorkflowActivityContext,
        payload: Dict[str, Any],
    ) -> None:
        """Send a TriggerAction to a specific agent via pub/sub."""
        name = payload.get("name")
        instance_id = payload.get("instance_id")
        if not name or not instance_id:
            logger.debug("Trigger activity missing agent name or instance id.")
            return

        trigger = TriggerAction(workflow_instance_id=instance_id)

        try:
            agents_metadata = self.list_team_agents(
                include_self=False,
                team=self.effective_team(),
            )
        except Exception:
            logger.exception("Unable to load agents metadata for trigger.")
            return

        async def _trigger() -> None:
            await send_message_to_agent(
                source=self.name,
                target_agent=name,
                message=trigger,
                agents_metadata=agents_metadata,
            )

        try:
            self._run_asyncio_task(_trigger())
        except Exception:
            logger.exception("Failed to trigger agent %s", name)
