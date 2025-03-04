from dapr_agents.workflow.orchestrators.base import OrchestratorServiceBase
from dapr_agents.types import DaprWorkflowContext, BaseMessage
from dapr_agents.workflow.decorators import workflow, task
from typing import Any, Optional
from dataclasses import dataclass
from datetime import timedelta
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class TriggerActionMessage(BaseModel):
    """
    Represents a message used to trigger an agent's activity within the workflow.
    """
    task: Optional[str] = None

@dataclass
class ChatLoop:
    """
    Represents the state of the chat loop, which is updated after each iteration.

    Attributes:
        message (str): The latest message in the conversation.
        iteration (int): The current iteration of the workflow loop.
    """
    message: str
    iteration: int

class RoundRobinOrchestrator(OrchestratorServiceBase):
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

        self.workflow_name = "RoundRobinWorkflow"
        
        super().model_post_init(__context)
    
    @workflow(name="RoundRobinWorkflow")
    def round_robin_workflow(self, ctx: DaprWorkflowContext, input: ChatLoop):
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
            input (ChatLoop): The current workflow state containing `message` and `iteration`.

        Returns:
            str: The last processed message when the workflow terminates.
        """
        message = input.get("message")
        iteration = input.get("iteration", 0)
        instance_id = ctx.instance_id

        if not ctx.is_replaying:
            logger.info(f"Round-robin iteration {iteration + 1} started (Instance ID: {instance_id}).")

        # Check Termination Condition
        if iteration >= self.max_iterations:
            logger.info(f"Max iterations reached. Ending round-robin workflow (Instance ID: {instance_id}).")
            return message

        # First iteration: Process input and broadcast
        if iteration == 0:
            message_input = yield ctx.call_activity(self.process_input, input={"message": message})
            logger.info(f"Initial message from {message_input['role']} -> {self.name}")

            # Broadcast initial message
            yield ctx.call_activity(self.broadcast_input_message, input=message_input)

        # Select next speaker
        next_speaker = yield ctx.call_activity(self.select_next_speaker, input={"iteration": iteration})

        # Trigger agent
        yield ctx.call_activity(self.trigger_agent, input={"name": next_speaker, "instance_id": instance_id})

        # Wait for response or timeout
        logger.info("Waiting for agent response...")
        event_data = ctx.wait_for_external_event("AgentCompletedTask")
        timeout_task = ctx.create_timer(timedelta(seconds=self.timeout))
        any_results = yield self.when_any([event_data, timeout_task])

        if any_results == timeout_task:
            logger.warning(f"Agent response timed out (Iteration: {iteration + 1}, Instance ID: {instance_id}).")
            task_results = {"name": "timeout", "content": "Timeout occurred. Continuing..."}
        else:
            task_results = yield event_data
            logger.info(f"{task_results['name']} -> {self.name}")

        # Update ChatLoop for next iteration
        input["message"] = task_results
        input["iteration"] = iteration + 1

        # Restart workflow with updated state
        ctx.continue_as_new(input)

    @task
    async def process_input(self, message: str):
        """
        Processes the input message for the workflow.

        Args:
            message (str): The user-provided input message.
        Returns:
            dict: Serialized UserMessage with the content.
        """
        return {"role": "user", "content": message}
    
    @task
    async def broadcast_input_message(self, **kwargs):
        """
        Broadcasts a message to all agents.

        Args:
            **kwargs: The message content and additional metadata.
        """
        message = {key: value for key, value in kwargs.items()}
        await self.broadcast_message(message=BaseMessage(**message))
    
    @task
    async def select_next_speaker(self, iteration: int) -> str:
        """
        Selects the next speaker in round-robin order.

        Args:
            iteration (int): The current iteration number.
        Returns:
            str: The name of the selected agent.
        """
        agents_metadata = await self.get_agents_metadata()
        if not agents_metadata:
            logger.warning("No agents available for selection.")
            raise ValueError("Agents metadata is empty. Cannot select next speaker.")

        agent_names = list(agents_metadata.keys())

        # Determine the next agent in the round-robin order
        next_speaker = agent_names[iteration % len(agent_names)]
        logger.info(f"{self.name} selected agent {next_speaker} for iteration {iteration}.")
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
            message=TriggerActionMessage(task=None),
            workflow_instance_id=instance_id,
        )