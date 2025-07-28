import logging
import random
from datetime import timedelta
from typing import Any, Dict, Optional

from dapr.ext.workflow import DaprWorkflowContext
from pydantic import BaseModel, Field

from dapr_agents.types import BaseMessage
from dapr_agents.workflow.decorators import message_router, task, workflow
from dapr_agents.workflow.orchestrators.base import OrchestratorWorkflowBase

logger = logging.getLogger(__name__)


class BroadcastMessage(BaseMessage):
    """
    Represents a broadcast message from an agent.
    """


class AgentTaskResponse(BaseMessage):
    """
    Represents a response message from an agent after completing a task.
    """

    workflow_instance_id: Optional[str] = Field(
        default=None, description="Dapr workflow instance id from source if available"
    )


class TriggerAction(BaseModel):
    """
    Represents a message used to trigger an agent's activity within the workflow.
    """

    task: Optional[str] = Field(
        None,
        description="The specific task to execute. If not provided, the agent will act "
        "based on its memory or predefined behavior.",
    )
    workflow_instance_id: Optional[str] = Field(
        default=None, description="Dapr workflow instance id from source if available"
    )


class RandomOrchestrator(OrchestratorWorkflowBase):
    """
    Implements a random workflow where agents are selected randomly to perform tasks.
    The workflow iterates through conversations, selecting a random agent at each step.

    Runs in a single for-loop, breaking when max_iterations is reached.
    """

    current_speaker: Optional[str] = Field(
        default=None,
        init=False,
        description="Current speaker in the conversation, to avoid immediate repeats when possible.",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes and configures the random workflow service.
        """
        self._workflow_name = "RandomWorkflow"
        super().model_post_init(__context)

    @workflow(name="RandomWorkflow")
    # TODO: add retry policies on activities.
    def main_workflow(self, ctx: DaprWorkflowContext, input: TriggerAction):
        """
        Executes the random workflow in up to `self.max_iterations` turns, selecting
        a different (or same) agent at random each turn.

        Args:
            ctx (DaprWorkflowContext): Workflow context.
            input (TriggerAction): Contains `task`.

        Returns:
            str: The final message content when the workflow terminates.
        """
        # Step 1: Gather initial task and instance ID
        task = input.get("task")
        instance_id = ctx.instance_id
        final_output: Optional[str] = None

        # Single loop from turn 1 to max_iterations inclusive
        for turn in range(1, self.max_iterations + 1):
            if not ctx.is_replaying:
                logger.info(
                    f"Random workflow turn {turn}/{self.max_iterations} "
                    f"(Instance ID: {instance_id})"
                )

            # Step 2: On turn 1, process initial task and broadcast
            if turn == 1:
                message = yield ctx.call_activity(
                    self.process_input, input={"task": task}
                )
                logger.info(f"Initial message from {message['role']} -> {self.name}")
                yield ctx.call_activity(
                    self.broadcast_message_to_agents, input={"message": message}
                )

            # Step 3: Select a random speaker
            random_speaker = yield ctx.call_activity(self.select_random_speaker)
            if not ctx.is_replaying:
                logger.info(f"{self.name} selected {random_speaker} (Turn {turn}).")

            # Step 4: Trigger the agent
            yield ctx.call_activity(
                self.trigger_agent,
                input={"name": random_speaker, "instance_id": instance_id},
            )

            # Step 5: Await for agent response or timeout
            if not ctx.is_replaying:
                logger.debug("Waiting for agent response...")
            event_data = ctx.wait_for_external_event("AgentTaskResponse")
            timeout_task = ctx.create_timer(timedelta(seconds=self.timeout))
            any_results = yield self.when_any([event_data, timeout_task])

            # Step 6: Handle response or timeout
            if any_results == timeout_task:
                if not ctx.is_replaying:
                    logger.warning(
                        f"Turn {turn}: agent response timed out (Instance ID: {instance_id})."
                    )
                result = {
                    "name": "timeout",
                    "content": "⏰ Timeout occurred. Continuing...",
                }
            else:
                result = yield event_data
                if not ctx.is_replaying:
                    logger.info(f"{result['name']} -> {self.name}")

            # Step 7: If this is the last allowed turn, mark final_output and break
            if turn == self.max_iterations:
                if not ctx.is_replaying:
                    logger.info(
                        f"Turn {turn}: max iterations reached (Instance ID: {instance_id})."
                    )
                final_output = result["content"]
                break

            # Otherwise, feed into next turn
            task = result["content"]

        # Sanity check (should never happen)
        if final_output is None:
            raise RuntimeError(
                "RandomWorkflow completed without producing a final_output"
            )

        # Return the final message content
        return final_output

    @task
    async def process_input(self, task: str) -> Dict[str, Any]:
        """
        Wraps the raw task into a UserMessage dict.
        """
        return {"role": "user", "name": self.name, "content": task}

    @task
    async def broadcast_message_to_agents(self, message: Dict[str, Any]):
        """
        Broadcasts a message to all agents (excluding orchestrator).
        """
        task_message = BroadcastMessage(**message)
        await self.broadcast_message(message=task_message, exclude_orchestrator=True)

    @task
    def select_random_speaker(self) -> str:
        """
        Selects a random speaker, avoiding repeats when possible.
        """
        agents = self.get_agents_metadata(exclude_orchestrator=True)
        if not agents:
            logger.error("No agents available for selection.")
            raise ValueError("Agents list is empty.")

        names = list(agents.keys())
        # Avoid repeating previous speaker if more than one agent
        if len(names) > 1 and self.current_speaker in names:
            names.remove(self.current_speaker)

        choice = random.choice(names)
        self.current_speaker = choice
        return choice

    @task
    async def trigger_agent(self, name: str, instance_id: str) -> None:
        """
        Sends a TriggerAction to the selected agent.
        """
        logger.info(f"Triggering agent {name} (Instance ID: {instance_id})")
        await self.send_message_to_agent(
            name=name,
            message=TriggerAction(workflow_instance_id=instance_id),
        )

    @message_router
    async def process_agent_response(self, message: AgentTaskResponse):
        """
        Handles incoming AgentTaskResponse events and re-raises them into the workflow.
        """
        workflow_instance_id = getattr(message, "workflow_instance_id", None)
        if not workflow_instance_id:
            logger.error("Missing workflow_instance_id on AgentTaskResponse; ignoring.")
            return
        # Log the received response
        logger.debug(
            f"{self.name} received response for workflow {workflow_instance_id}"
        )
        logger.debug(f"Full response: {message}")
        # Raise a workflow event with the Agent's Task Response
        self.raise_workflow_event(
            instance_id=workflow_instance_id,
            event_name="AgentTaskResponse",
            data=message,
        )
