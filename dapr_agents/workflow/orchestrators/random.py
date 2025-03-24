from dapr_agents.workflow.orchestrators.base import OrchestratorWorkflowBase
from dapr_agents.types import DaprWorkflowContext, BaseMessage, EventMessageMetadata
from dapr_agents.workflow.decorators import workflow, task
from dapr_agents.workflow.messaging.decorator import message_router
from fastapi.responses import JSONResponse
from fastapi import Response, status
from typing import Any, Optional, Dict, Any
from datetime import timedelta
from pydantic import BaseModel, Field
import random
import logging


logger = logging.getLogger(__name__)

class AgentTaskResponse(BaseMessage):
    """
    Represents a response message from an agent after completing a task.
    """
    workflow_instance_id: Optional[str] = Field(default=None, description="Dapr workflow instance id from source if available")

class TriggerAction(BaseModel):
    """
    Represents a message used to trigger an agent's activity within the workflow.
    """
    task: Optional[str] = Field(None, description="The specific task to execute. If not provided, the agent can act based on its memory or predefined behavior.")
    iteration: Optional[int] = Field(default=0, description="The current iteration of the workflow loop.")

class RandomOrchestrator(OrchestratorWorkflowBase):
    """
    Implements a random workflow where agents are selected randomly to perform tasks.
    The workflow iterates through conversations, selecting a random agent at each step.

    Uses `continue_as_new` to persist iteration state.
    """

    current_speaker: Optional[str] = Field(default=None, init=False, description="Current speaker in the conversation.")

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes and configures the random workflow service.
        Registers tasks and workflows, then starts the workflow runtime.
        """
        self._workflow_name = "RandomWorkflow"
        
        super().model_post_init(__context)

    @workflow(name="RandomWorkflow")
    def main_workflow(self, ctx: DaprWorkflowContext, input: TriggerAction):
        """
        Executes a random workflow where agents are selected randomly for interactions.
        Uses `continue_as_new` to persist iteration state.

        Args:
            ctx (DaprWorkflowContext): The workflow execution context.
            input (TriggerAction): The current workflow state containing `message` and `iteration`.

        Returns:
            str: The last processed message when the workflow terminates.
        """
        # Step 0: Retrieving Loop Context
        task = input.get("task")
        iteration = input.get("iteration")
        instance_id = ctx.instance_id

        if not ctx.is_replaying:
            logger.info(f"Random workflow iteration {iteration + 1} started (Instance ID: {instance_id}).")
        
        # First iteration: Process input and broadcast
        if iteration == 0:
            message = yield ctx.call_activity(self.process_input, input={"task": task})
            logger.info(f"Initial message from {message['role']} -> {self.name}")

            # Step 1: Broadcast initial message
            yield ctx.call_activity(self.broadcast_message_to_agents, input={"message": message})

        # Step 2: Select a random speaker
        random_speaker = yield ctx.call_activity(self.select_random_speaker, input={"iteration": iteration})

        # Step 3: Trigger agent
        yield ctx.call_activity(self.trigger_agent, input={"name": random_speaker, "instance_id": instance_id})

        # Step 4: Wait for response or timeout
        logger.info("Waiting for agent response...")
        event_data = ctx.wait_for_external_event("AgentTaskResponse")
        timeout_task = ctx.create_timer(timedelta(seconds=self.timeout))
        any_results = yield self.when_any([event_data, timeout_task])

        if any_results == timeout_task:
            logger.warning(f"Agent response timed out (Iteration: {iteration + 1}, Instance ID: {instance_id}).")
            task_results = {"name": "timeout", "content": "Timeout occurred. Continuing..."}
        else:
            task_results = yield event_data
            logger.info(f"{task_results['name']} -> {self.name}")
        
        # Step 5: Check Iteration
        next_iteration_count = iteration + 1
        if next_iteration_count > self.max_iterations:
            logger.info(f"Max iterations reached. Ending random workflow (Instance ID: {instance_id}).")
            return task_results['content']
        
        # Update ChatLoop for next iteration
        input["task"] = task_results['content']
        input["iteration"] = iteration + 1
        
        # Restart workflow with updated state
        ctx.continue_as_new(input)

    @task
    async def process_input(self, task: str):
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
        await self.broadcast_message(message=BaseMessage(**message), exclude_orchestrator=True)

    @task
    def select_random_speaker(self, iteration: int) -> str:
        """
        Selects a random speaker, ensuring that a different agent is chosen if possible.

        Args:
            iteration (int): The current iteration number.
        Returns:
            str: The name of the randomly selected agent.
        """
        agents_metadata = self.get_agents_metadata(exclude_orchestrator=True)
        if not agents_metadata:
            logger.warning("No agents available for selection.")
            raise ValueError("Agents metadata is empty. Cannot select a random speaker.")

        agent_names = list(agents_metadata.keys())

        # Handle single-agent scenarios
        if len(agent_names) == 1:
            random_speaker = agent_names[0]
            logger.info(f"Only one agent available: {random_speaker}. Using the same agent.")
            return random_speaker

        # Select a random speaker, avoiding repeating the previous speaker when possible
        previous_speaker = getattr(self, "current_speaker", None)
        if previous_speaker in agent_names and len(agent_names) > 1:
            agent_names.remove(previous_speaker)

        random_speaker = random.choice(agent_names)
        self.current_speaker = random_speaker
        logger.info(f"{self.name} randomly selected agent {random_speaker} (Iteration: {iteration}).")
        return random_speaker

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
            message=TriggerAction(task=None),
            workflow_instance_id=instance_id,
        )
    
    @message_router
    async def process_agent_response(self, message: AgentTaskResponse, metadata: EventMessageMetadata) -> Response:
        """
        Processes agent response messages sent directly to the agent's topic.

        Args:
            message (AgentTaskResponse): The agent's response containing task results.
            metadata (EventMessageMetadata): Metadata associated with the message, including headers.
        
        Returns:
            Response: A JSON response confirming the workflow event was successfully triggered.
        """
        agent_response = (message).model_dump()
        workflow_instance_id = metadata.headers.get("workflow_instance_id")
        event_name = metadata.headers.get("event_name", "AgentTaskResponse")

        # Raise a workflow event with the Agent's Task Response!
        self.raise_workflow_event(instance_id=workflow_instance_id, event_name=event_name, data=agent_response)

        return JSONResponse(content={"message": "Workflow event triggered successfully."}, status_code=status.HTTP_200_OK)