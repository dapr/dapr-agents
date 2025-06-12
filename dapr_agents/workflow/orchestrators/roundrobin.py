from dapr_agents.workflow.messaging.decorator import message_router
from dapr_agents.workflow.orchestrators.base import OrchestratorWorkflowBase
from dapr.ext.workflow import DaprWorkflowContext
from dapr_agents.types import BaseMessage
from dapr_agents.workflow.decorators import workflow, task
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


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
        description="The specific task to execute. If not provided, the agent will act based on its memory or predefined behavior.",
    )
    iteration: Optional[int] = Field(0, description="")
    workflow_instance_id: Optional[str] = Field(
        default=None, description="Dapr workflow instance id from source if available"
    )


class RoundRobinOrchestrator(OrchestratorWorkflowBase):
    """
    Implements a round-robin workflow where agents take turns performing tasks.
    The workflow iterates through conversations by selecting agents in a circular order.

    Uses `continue_as_new` to persist iteration state.
    """

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes and configures the round-robin workflow service.
        Registers tasks and workflows, then starts the workflow runtime.
        """
        self._workflow_name = "RoundRobinWorkflow"
        super().model_post_init(__context)

    @workflow(name="RoundRobinWorkflow")
    # TODO: add retry policies on activities.
    def main_workflow(self, ctx: DaprWorkflowContext, input: TriggerAction):
        """
        Executes a round-robin workflow where agents interact iteratively.

        Steps:
        1. Processes input and broadcasts the initial message.
        2. Iterates through agents, selecting a speaker each round.
        3. Waits for agent responses or handles timeouts.
        4. Updates the workflow state and continues the loop.
        5. Terminates when max iterations are reached.

        Uses `continue_as_new` to persist iteration state.

        Args:
            ctx (DaprWorkflowContext): The workflow execution context.
            input (TriggerAction): The current workflow state containing task and iteration.

        Returns:
            str: The last processed message when the workflow terminates.
        """
        task = input.get("task")
        iteration = input.get("iteration", 0)
        instance_id = ctx.instance_id

        if not ctx.is_replaying:
            logger.info(
                f"Round-robin iteration {iteration + 1} started (Instance ID: {instance_id})."
            )

        # Check Termination Condition
        if iteration >= self.max_iterations:
            logger.info(
                f"Max iterations reached. Ending round-robin workflow (Instance ID: {instance_id})."
            )
            return task

        # First iteration: Process input and broadcast
        if iteration == 0:
            message = yield ctx.call_activity(self.process_input, input={"task": task})
            logger.info(f"Initial message from {message['role']} -> {self.name}")

            # Broadcast initial message
            yield ctx.call_activity(
                self.broadcast_message_to_agents, input={"message": message}
            )

        # Select next speaker
        next_speaker = yield ctx.call_activity(
            self.select_next_speaker, input={"iteration": iteration}
        )

        # Trigger agent
        yield ctx.call_activity(
            self.trigger_agent, input={"name": next_speaker, "instance_id": instance_id}
        )

        # Wait for response or timeout
        logger.info("Waiting for agent response...")
        event_data = ctx.wait_for_external_event("AgentTaskResponse")
        timeout_task = ctx.create_timer(timedelta(seconds=self.timeout))
        any_results = yield self.when_any([event_data, timeout_task])

        if any_results == timeout_task:
            logger.warning(
                f"Agent response timed out (Iteration: {iteration + 1}, Instance ID: {instance_id})."
            )
            task_results = {
                "name": "timeout",
                "content": "Timeout occurred. Continuing...",
            }
        else:
            task_results = yield event_data
            logger.info(f"{task_results['name']} -> {self.name}")

        # Check Iteration
        next_iteration_count = iteration + 1
        if next_iteration_count > self.max_iterations:
            logger.info(
                f"Max iterations reached. Ending round-robin workflow (Instance ID: {instance_id})."
            )
            return task_results["content"]

        # Update for next iteration
        input["task"] = task_results["content"]
        input["iteration"] = next_iteration_count

        # Restart workflow with updated state
        # TODO: would we want this updated to preserve agent state between iterations?
        ctx.continue_as_new(input)

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
        await self.broadcast_message(
            message=BaseMessage(**message), exclude_orchestrator=True
        )

    @task
    async def select_next_speaker(self, iteration: int) -> str:
        """
        Selects the next speaker in round-robin order.

        Args:
            iteration (int): The current iteration number.
        Returns:
            str: The name of the selected agent.
        """
        agents_metadata = self.get_agents_metadata(exclude_orchestrator=True)
        if not agents_metadata:
            logger.warning("No agents available for selection.")
            raise ValueError("Agents metadata is empty. Cannot select next speaker.")

        agent_names = list(agents_metadata.keys())

        # Determine the next agent in the round-robin order
        next_speaker = agent_names[iteration % len(agent_names)]
        logger.info(
            f"{self.name} selected agent {next_speaker} for iteration {iteration}."
        )
        return next_speaker

    @task
    async def trigger_agent(self, name: str, instance_id: str) -> None:
        """
        Triggers the specified agent to perform its activity.

        Args:
            name (str): Name of the agent to trigger.
            instance_id (str): Workflow instance ID for context.
        """
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
        try:
            workflow_instance_id = message.get("workflow_instance_id")

            if not workflow_instance_id:
                logger.error(
                    f"{self.name} received an agent response without a valid workflow_instance_id. Ignoring."
                )
                return

            logger.info(
                f"{self.name} processing agent response for workflow instance '{workflow_instance_id}'."
            )

            # Raise a workflow event with the Agent's Task Response
            self.raise_workflow_event(
                instance_id=workflow_instance_id,
                event_name="AgentTaskResponse",
                data=message,
            )

        except Exception as e:
            logger.error(f"Error processing agent response: {e}", exc_info=True)
