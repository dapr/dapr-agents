import logging
from datetime import timedelta
from typing import Any, Dict, Optional

from dapr.ext.workflow import DaprWorkflowContext
from pydantic import BaseModel, Field

from dapr_agents.types import BaseMessage
from dapr_agents.workflow.decorators import message_router, task, workflow
from dapr_agents.workflow.orchestrators.base import OrchestratorWorkflowBase

logger = logging.getLogger(__name__)


class AgentTaskResponse(BaseMessage):
    """
    Represents a response message from an agent after completing a task.
    """

    workflow_instance_id: Optional[str] = Field(
        default=None, description="Dapr workflow instance id from source if available"
    )


class BroadcastMessage(BaseMessage):
    """
    Represents a broadcast message from an agent.
    """


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


class RoundRobinOrchestrator(OrchestratorWorkflowBase):
    """
    Implements a round-robin workflow where agents take turns performing tasks.
    Iterates for up to `self.max_iterations` turns, then returns the last reply.
    """

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes and configures the round-robin workflow.
        """
        self._workflow_name = "RoundRobinWorkflow"
        super().model_post_init(__context)

    @workflow(name="RoundRobinWorkflow")
    # TODO: add retry policies on activities.
    def main_workflow(self, ctx: DaprWorkflowContext, input: TriggerAction) -> str:
        """
        Drives the round-robin loop in up to `max_iterations` turns.

        Args:
            ctx (DaprWorkflowContext): Workflow context.
            input (TriggerAction): Contains the initial `task`.

        Returns:
            str: The final message content when the workflow terminates.
        """
        # Step 1: Extract task and instance ID from input
        task = input.get("task")
        instance_id = ctx.instance_id
        final_output: Optional[str] = None

        # Loop from 1..max_iterations
        for turn in range(1, self.max_iterations + 1):
            if not ctx.is_replaying:
                logger.info(
                    f"Round-robin turn {turn}/{self.max_iterations} "
                    f"(Instance ID: {instance_id})"
                )

            # Step 2: On turn 1, process input and broadcast message
            if turn == 1:
                message = yield ctx.call_activity(
                    self.process_input, input={"task": task}
                )
                if not ctx.is_replaying:
                    logger.info(
                        f"Initial message from {message['role']} -> {self.name}"
                    )
                yield ctx.call_activity(
                    self.broadcast_message_to_agents, input={"message": message}
                )

            # Step 3: Select next speaker in round-robin order
            speaker = yield ctx.call_activity(
                self.select_next_speaker, input={"turn": turn}
            )
            if not ctx.is_replaying:
                logger.info(f"Selected agent {speaker} for turn {turn}")

            # Step 4: Trigger that agent
            yield ctx.call_activity(
                self.trigger_agent,
                input={"name": speaker, "instance_id": instance_id},
            )

            # Step 5: Wait for agent response or timeout
            if not ctx.is_replaying:
                logger.debug("Waiting for agent response...")
            event_data = ctx.wait_for_external_event("AgentTaskResponse")
            timeout_task = ctx.create_timer(timedelta(seconds=self.timeout))
            any_results = yield self.when_any([event_data, timeout_task])

            # Step 6: Handle result or timeout
            if any_results == timeout_task:
                if not ctx.is_replaying:
                    logger.warning(
                        f"Turn {turn}: response timed out "
                        f"(Instance ID: {instance_id})"
                    )
                result = {
                    "name": "timeout",
                    "content": "Timeout occurred. Continuing...",
                }
            else:
                result = yield event_data
                if not ctx.is_replaying:
                    logger.info(f"{result['name']} -> {self.name}")

            # Step 7: If this is the last allowed turn, capture and break
            if turn == self.max_iterations:
                if not ctx.is_replaying:
                    logger.info(
                        f"Turn {turn}: max iterations reached (Instance ID: {instance_id})."
                    )
                final_output = result["content"]
                break

            # Otherwise, feed into next iteration
            task = result["content"]

        # Sanity check: final_output must be set
        if final_output is None:
            raise RuntimeError(
                "RoundRobinWorkflow completed without producing final_output"
            )

        return final_output

    @task
    async def process_input(self, task: str) -> Dict[str, Any]:
        """
        Processes the input message for the workflow.

        Args:
            task (str): The user-provided input task.
        Returns:
            dict: Serialized UserMessage with the content.
        """
        return {"role": "user", "name": self.name, "content": task}

    @task
    async def broadcast_message_to_agents(self, message: Dict[str, Any]):
        """
        Broadcasts a message to all agents.

        Args:
            message (Dict[str, Any]): The message content and additional metadata.
        """
        # Format message for broadcasting
        task_message = BroadcastMessage(**message)
        # Send broadcast message
        await self.broadcast_message(message=task_message, exclude_orchestrator=True)

    @task
    async def select_next_speaker(self, turn: int) -> str:
        """
        Selects the next speaker in round-robin order.

        Args:
            turn (int): The current turn number (1-based).
        Returns:
            str: The name of the selected agent.
        """
        agents_metadata = self.get_agents_metadata(exclude_orchestrator=True)
        if not agents_metadata:
            logger.warning("No agents available for selection.")
            raise ValueError("Agents metadata is empty. Cannot select next speaker.")

        agent_names = list(agents_metadata.keys())
        next_speaker = agent_names[(turn - 1) % len(agent_names)]
        return next_speaker

    @task
    async def trigger_agent(self, name: str, instance_id: str) -> None:
        """
        Triggers the specified agent to perform its activity.

        Args:
            name (str): Name of the agent to trigger.
            instance_id (str): Workflow instance ID for context.
        """
        logger.info(f"Triggering agent {name} (Instance ID: {instance_id})")
        await self.send_message_to_agent(
            name=name,
            message=TriggerAction(workflow_instance_id=instance_id),
        )

    @message_router
    async def process_agent_response(self, message: AgentTaskResponse):
        """
        Processes agent response messages sent directly to the agent's topic.

        Args:
            message (AgentTaskResponse): The agent's response containing task results.

        Returns:
            None: The function raises a workflow event with the agent's response.
        """
        workflow_instance_id = getattr(message, "workflow_instance_id", None)

        if not workflow_instance_id:
            logger.error(
                f"{self.name} received an agent response without a valid workflow_instance_id. Ignoring."
            )
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
