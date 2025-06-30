import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dapr.ext.workflow import DaprWorkflowContext
from dapr_agents.workflow.decorators import task, workflow
from dapr_agents.workflow.messaging.decorator import message_router
from dapr_agents.workflow.orchestrators.base import OrchestratorWorkflowBase
from dapr_agents.workflow.orchestrators.llm.schemas import (
    BroadcastMessage,
    TriggerAction,
    NextStep,
    AgentTaskResponse,
    ProgressCheckOutput,
    schemas,
)
from dapr_agents.workflow.orchestrators.llm.prompts import (
    TASK_INITIAL_PROMPT,
    TASK_PLANNING_PROMPT,
    NEXT_STEP_PROMPT,
    PROGRESS_CHECK_PROMPT,
    SUMMARY_GENERATION_PROMPT,
)
from dapr_agents.workflow.orchestrators.llm.state import (
    LLMWorkflowState,
    LLMWorkflowEntry,
    LLMWorkflowMessage,
    PlanStep,
    TaskResult,
)
from dapr_agents.workflow.orchestrators.llm.utils import (
    update_step_statuses,
    restructure_plan,
    find_step_in_plan,
)

logger = logging.getLogger(__name__)


class LLMOrchestrator(OrchestratorWorkflowBase):
    """
    Implements an agentic workflow where an LLM dynamically selects the next speaker.
    The workflow iterates through conversations, updating its state and persisting messages.

    Uses the `continue_as_new` pattern to restart the workflow with updated input at each iteration.
    """

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes and configures the LLM-based workflow service.
        """

        # Initializes local LLM Orchestrator State
        self.state = LLMWorkflowState()

        # Set main workflow name
        self._workflow_name = "LLMWorkflow"

        super().model_post_init(__context)

    @message_router
    @workflow(name="LLMWorkflow")
    # TODO: set retry policies on the activities!
    # TODO: utilize prompt verdict value of failed as we do not currently use.
    # https://github.com/dapr/dapr-agents/pull/136#discussion_r2175751545
    def main_workflow(self, ctx: DaprWorkflowContext, message: TriggerAction):
        """
        Executes an LLM-driven agentic workflow where the next agent is dynamically selected
        based on task progress. The workflow iterates through execution cycles, updating state,
        handling agent responses, and determining task completion.

        Args:
            ctx (DaprWorkflowContext): The workflow execution context.
            message (TriggerAction): The current workflow state containing `message`, `iteration`, and `verdict`.

        Returns:
            str: The final processed message when the workflow terminates.

        Raises:
            RuntimeError: If the LLM determines the task is `failed`.
        """
        # Step 0: Retrieve iteration messages
        task = message.get("task")
        iteration = message.get("iteration")

        # Step 1:
        # Ensure 'instances' and the instance_id entry exist
        instance_id = ctx.instance_id
        self.state.setdefault("instances", {}).setdefault(
            instance_id, LLMWorkflowEntry(input=task).model_dump(mode="json")
        )
        # Retrieve the plan (will always exist after initialization)
        plan = self.state["instances"][instance_id].get("plan", [])

        if not ctx.is_replaying:
            logger.info(
                f"Workflow iteration {iteration + 1} started (Instance ID: {instance_id})."
            )

        # Step 2: Retrieve available agents
        agents = yield ctx.call_activity(self.get_agents_metadata_as_string)

        # Step 3: First iteration setup
        if iteration == 0:
            if not ctx.is_replaying:
                logger.info(f"Initial message from User -> {self.name}")

            # Generate the plan using a language model
            plan = yield ctx.call_activity(
                self.generate_plan,
                input={"task": task, "agents": agents, "plan_schema": schemas.plan},
            )

            # Prepare initial message with task, agents and plan context
            initial_message = yield ctx.call_activity(
                self.prepare_initial_message,
                input={
                    "instance_id": instance_id,
                    "task": task,
                    "agents": agents,
                    "plan": plan,
                },
            )

            # broadcast initial message to all agents
            yield ctx.call_activity(
                self.broadcast_message_to_agents,
                input={"instance_id": instance_id, "task": initial_message},
            )

        # Step 4: Identify agent and instruction for the next step
        next_step = yield ctx.call_activity(
            self.generate_next_step,
            input={
                "task": task,
                "agents": agents,
                "plan": plan,
                "next_step_schema": schemas.next_step,
            },
        )

        # Extract Additional Properties from NextStep
        next_agent = next_step["next_agent"]
        instruction = next_step["instruction"]
        step_id = next_step.get("step", None)
        substep_id = next_step.get("substep", None)

        # Step 5: Validate Step Before Proceeding
        valid_step = yield ctx.call_activity(
            self.validate_next_step,
            input={
                "instance_id": instance_id,
                "plan": plan,
                "step": step_id,
                "substep": substep_id,
            },
        )

        if valid_step:
            # Step 6: Broadcast Task to all Agents
            yield ctx.call_activity(
                self.broadcast_message_to_agents,
                input={"instance_id": instance_id, "task": instruction},
            )

            # Step 7: Trigger next agent
            plan = yield ctx.call_activity(
                self.trigger_agent,
                input={
                    "instance_id": instance_id,
                    "name": next_agent,
                    "step": step_id,
                    "substep": substep_id,
                },
            )

            # Step 8: Wait for agent response or timeout
            if not ctx.is_replaying:
                logger.info(f"Waiting for {next_agent}'s response...")

            event_data = ctx.wait_for_external_event("AgentTaskResponse")
            timeout_task = ctx.create_timer(timedelta(seconds=self.timeout))
            any_results = yield self.when_any([event_data, timeout_task])

            if any_results == timeout_task:
                logger.warning(
                    f"Agent response timed out (Iteration: {iteration + 1}, Instance ID: {instance_id})."
                )
                task_results = {
                    "name": self.name,
                    "role": "user",
                    "content": f"Timeout occurred. {next_agent} did not respond on time. We need to try again...",
                }
            else:
                task_results = yield event_data
                if not ctx.is_replaying:
                    logger.info(f"{task_results['name']} sent a response.")

            # Step 9: Save the task execution results to chat and task history
            yield ctx.call_activity(
                self.update_task_history,
                input={
                    "instance_id": instance_id,
                    "agent": next_agent,
                    "step": step_id,
                    "substep": substep_id,
                    "results": task_results,
                },
            )

            # Step 10: Check progress
            progress = yield ctx.call_activity(
                self.check_progress,
                input={
                    "task": task,
                    "plan": plan,
                    "step": step_id,
                    "substep": substep_id,
                    "results": task_results["content"],
                    "progress_check_schema": schemas.progress_check,
                },
            )

            if not ctx.is_replaying:
                logger.info(f"Tracking Progress: {progress}")

            verdict = progress["verdict"]
            status_updates = progress.get("plan_status_update", [])
            plan_updates = progress.get("plan_restructure", [])

        else:
            logger.warning(
                f"Step {step_id}, Substep {substep_id} not found in plan for instance {instance_id}. Recovering..."
            )

            # Recovery Task: No updates, just iterate again
            verdict = "continue"
            status_updates = []
            plan_updates = []
            task_results = {
                "name": "orchestrator",
                "role": "user",
                "content": f"Step {step_id}, Substep {substep_id} does not exist in the plan. Adjusting workflow...",
            }

        # Step 11: Process progress suggestions and next iteration count
        next_iteration_count = iteration + 1
        if verdict != "continue" or next_iteration_count > self.max_iterations:
            if next_iteration_count >= self.max_iterations:
                verdict = "max_iterations_reached"

            if not ctx.is_replaying:
                logger.info(f"Workflow ending with verdict: {verdict}")

            # Generate final summary based on execution
            summary = yield ctx.call_activity(
                self.generate_summary,
                input={
                    "task": task,
                    "verdict": verdict,
                    "plan": plan,
                    "step": step_id,
                    "substep": substep_id,
                    "agent": next_agent,
                    "result": task_results["content"],
                },
            )

            # Finalize the workflow properly
            yield ctx.call_activity(
                self.finish_workflow,
                input={
                    "instance_id": instance_id,
                    "plan": plan,
                    "step": step_id,
                    "substep": substep_id,
                    "verdict": verdict,
                    "summary": summary,
                },
            )

            if not ctx.is_replaying:
                logger.info(
                    f"Workflow {instance_id} has been finalized with verdict: {verdict}"
                )

            return summary

        if status_updates or plan_updates:
            yield ctx.call_activity(
                self.update_plan,
                input={
                    "instance_id": instance_id,
                    "plan": plan,
                    "status_updates": status_updates,
                    "plan_updates": plan_updates,
                },
            )

        # Step 12: Update TriggerAction state and continue workflow
        message["task"] = task_results["content"]
        message["iteration"] = next_iteration_count

        # Restart workflow with updated TriggerAction state
        ctx.continue_as_new(message)

    @task
    def get_agents_metadata_as_string(self) -> str:
        """
        Retrieves and formats metadata about available agents.

        Returns:
            str: A formatted string listing the available agents and their roles.
        """
        agents_metadata = self.get_agents_metadata(exclude_orchestrator=True)
        if not agents_metadata:
            return "No available agents to assign tasks."

        # Format agent details into a readable string
        agent_list = "\n".join(
            [
                f"- {name}: {metadata.get('role', 'Unknown role')} (Goal: {metadata.get('goal', 'Unknown')})"
                for name, metadata in agents_metadata.items()
            ]
        )

        return agent_list

    @task(description=TASK_PLANNING_PROMPT)
    async def generate_plan(
        self, task: str, agents: str, plan_schema: str
    ) -> List[PlanStep]:
        """
        Generates a structured execution plan for the given task.

        Args:
            task (str): The description of the task to be executed.
            agents (str): Formatted list of available agents and their roles.
            plan_schema (sr): Schema of the plan

        Returns:
            List[Dict[str, Any]]: A list of steps for the overall plan.
        """
        pass

    @task
    async def prepare_initial_message(
        self, instance_id: str, task: str, agents: str, plan: List[Dict[str, Any]]
    ) -> str:
        """
        Initializes the workflow entry and sends the first task briefing to all agents.

        Args:
            instance_id (str): The ID of the workflow instance.
            task (str): The initial input message describing the task.
            agents (str): The formatted list of available agents.
            plan (List[Dict[str, Any]]): The structured execution plan generated beforehand.
        """
        # Format Initial Message with the Plan
        formatted_message = TASK_INITIAL_PROMPT.format(
            task=task, agents=agents, plan=plan
        )

        # Save initial plan using update_workflow_state for consistency
        await self.update_workflow_state(instance_id=instance_id, plan=plan)

        # Return formatted prompt
        return formatted_message

    @task
    async def broadcast_message_to_agents(self, instance_id: str, task: str):
        """
        Saves message to workflow state and broadcasts it to all registered agents.

        Args:
            instance_id (str): Workflow instance ID for context.
            task (str): A task to append to the workflow state and broadcast to all agents.
        """
        # Ensure message is a string
        if not isinstance(task, str):
            raise ValueError("Message must be a string.")

        # Store message in workflow state
        await self.update_workflow_state(
            instance_id=instance_id,
            message={"name": self.name, "role": "user", "content": task},
        )

        # Format message for broadcasting
        task_message = BroadcastMessage(name=self.name, role="user", content=task)

        # Send broadcast message
        await self.broadcast_message(message=task_message, exclude_orchestrator=True)

    @task(description=NEXT_STEP_PROMPT, include_chat_history=True)
    async def generate_next_step(
        self, task: str, agents: str, plan: str, next_step_schema: str
    ) -> NextStep:
        """
        Determines the next agent to respond in a workflow.

        Args:
            task (str): The current task description.
            agents (str): A list of available agents.
            plan (str): The structured execution plan.
            next_step_schema (str): The next step schema.

        Returns:
            Dict: A structured response with the next agent, an instruction, and step ids.
        """
        pass

    @task
    async def validate_next_step(
        self,
        instance_id: str,
        plan: List[Dict[str, Any]],
        step: int,
        substep: Optional[float],
    ) -> bool:
        """
        Validates if the next step exists in the current execution plan.

        Args:
            instance_id (str): The workflow instance ID.
            plan (List[Dict[str, Any]]): The current execution plan.
            step (int): The step number.
            substep (Optional[float]): The substep number.

        Returns:
            bool: True if the step exists, False if it does not.
        """
        step_entry = find_step_in_plan(plan, step, substep)
        if not step_entry:
            logger.error(
                f"Step {step}, Substep {substep} not found in plan for instance {instance_id}."
            )
            return False
        return True

    @task
    async def trigger_agent(
        self, instance_id: str, name: str, step: int, substep: Optional[float]
    ) -> List[dict[str, Any]]:
        """
        Updates step status and triggers the specified agent to perform its activity.

        Args:
            instance_id (str): Workflow instance ID for context.
            name (str): Name of the agent to trigger.
            step (int): The step number associated with the task.
            substep (Optional[float]): The substep number, if applicable.

        Returns:
            List[Dict[str, Any]]: The updated execution plan.
        """
        logger.info(
            f"Triggering agent {name} for step {step}, substep {substep} (Instance ID: {instance_id})"
        )

        # Get the workflow entry from self.state
        workflow_entry = self.state["instances"].get(instance_id)
        if not workflow_entry:
            raise ValueError(f"No workflow entry found for instance_id: {instance_id}")

        plan = workflow_entry["plan"]

        # Ensure step or substep exists
        step_entry = find_step_in_plan(plan, step, substep)
        if not step_entry:
            if substep is not None:
                raise ValueError(
                    f"Substep {substep} in Step {step} not found in the current plan."
                )
            raise ValueError(f"Step {step} not found in the current plan.")

        # Mark step or substep as "in_progress"
        step_entry["status"] = "in_progress"
        logger.info(f"Marked step {step}, substep {substep} as 'in_progress'")

        # Apply global status updates to maintain consistency
        updated_plan = update_step_statuses(plan)

        # Save updated plan state
        await self.update_workflow_state(instance_id=instance_id, plan=updated_plan)

        # Send message to agent
        await self.send_message_to_agent(
            name=name, message=TriggerAction(workflow_instance_id=instance_id)
        )

        return updated_plan

    @task
    async def update_task_history(
        self,
        instance_id: str,
        agent: str,
        step: int,
        substep: Optional[float],
        results: Dict[str, Any],
    ):
        """
        Updates the task history for a workflow instance by recording the results of an agent's execution.

        Args:
            instance_id (str): The unique workflow instance ID.
            agent (str): The name of the agent who performed the task.
            step (int): The step number associated with the task.
            substep (Optional[float]): The substep number, if applicable.
            results (Dict[str, Any]): The result or response generated by the agent.

        Raises:
            ValueError: If the instance ID does not exist in the workflow state.
        """

        logger.info(
            f"Updating task history for {agent} at step {step}, substep {substep} (Instance ID: {instance_id})"
        )

        # Store the agent's response in the message history
        await self.update_workflow_state(instance_id=instance_id, message=results)

        # Retrieve Workflow state
        workflow_entry = self.state["instances"].get(instance_id)
        if not workflow_entry:
            raise ValueError(f"No workflow entry found for instance_id: {instance_id}")

        # Create a TaskResult object
        task_result = TaskResult(
            agent=agent, step=step, substep=substep, result=results["content"]
        )

        # Append the result to task history
        workflow_entry["task_history"].append(task_result.model_dump(mode="json"))

        # Persist state
        await self.update_workflow_state(
            instance_id=instance_id, plan=workflow_entry["plan"]
        )

    @task(description=PROGRESS_CHECK_PROMPT, include_chat_history=True)
    async def check_progress(
        self,
        task: str,
        plan: str,
        step: int,
        substep: Optional[float],
        results: str,
        progress_check_schema: str,
    ) -> ProgressCheckOutput:
        """
        Evaluates the current plan's progress and determines necessary updates.

        Args:
            task (str): The current task description.
            plan (str): The structured execution plan.
            step (int): The step number associated with the task.
            substep (Optional[float]): The substep number, if applicable.
            results (str): The result or response generated by the agent.
            progress_check_schema (str): The schema of the progress check

        Returns:
            ProgressCheckOutput: The plan update details, including status changes and restructuring if needed.
        """
        pass

    @task
    async def update_plan(
        self,
        instance_id: str,
        plan: List[Dict[str, Any]],
        status_updates: Optional[List[Dict[str, Any]]] = None,
        plan_updates: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Updates the execution plan based on status changes and/or plan restructures.

        Args:
            instance_id (str): The workflow instance ID.
            plan (List[Dict[str, Any]]): The current execution plan.
            status_updates (Optional[List[Dict[str, Any]]]): List of updates for step statuses.
            plan_updates (Optional[List[Dict[str, Any]]]): List of full step modifications.

        Raises:
            ValueError: If a specified step or substep is not found.
        """
        logger.info(f"Updating plan for instance {instance_id}")

        # Step 1: Apply status updates directly to `plan`
        if status_updates:
            for update in status_updates:
                step_id = update["step"]
                substep_id = update.get("substep")
                new_status = update["status"]

                step_entry = find_step_in_plan(plan, step_id, substep_id)
                if not step_entry:
                    error_msg = f"Step {step_id}, Substep {substep_id} not found in the current plan."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Apply status update
                step_entry["status"] = new_status
                logger.info(
                    f"Updated status of step {step_id}, substep {substep_id} to '{new_status}'"
                )

        # Step 2: Apply plan restructuring updates (if provided)
        if plan_updates:
            plan = restructure_plan(plan, plan_updates)
            logger.info(f"Applied restructuring updates for {len(plan_updates)} steps.")

        # Step 3: Apply global consistency checks for statuses
        plan = update_step_statuses(plan)

        # Save to state and update workflow
        await self.update_workflow_state(instance_id=instance_id, plan=plan)

        logger.info(f"Plan successfully updated for instance {instance_id}")

    @task(description=SUMMARY_GENERATION_PROMPT, include_chat_history=True)
    async def generate_summary(
        self,
        task: str,
        verdict: str,
        plan: str,
        step: int,
        substep: Optional[float],
        agent: str,
        result: str,
    ) -> str:
        """
        Generates a structured summary of task execution based on conversation history, execution results, and the task plan.

        Args:
            task (str): The original task description.
            verdict (str): The overall task status (e.g., "continue", "completed", or "failed").
            plan (str): The structured execution plan detailing task progress.
            step (int): The step number associated with the most recent action.
            substep (Optional[float]): The substep number, if applicable.
            agent (str): The name of the agent who executed the last action.
            result (str): The response or outcome generated by the agent.

        Returns:
            str: A concise but informative summary of task progress and results, structured for user readability.
        """
        pass

    @task
    async def finish_workflow(
        self,
        instance_id: str,
        plan: List[Dict[str, Any]],
        step: int,
        substep: Optional[float],
        verdict: str,
        summary: str,
    ):
        """
        Finalizes the workflow by updating the plan, marking the provided step/substep as completed if applicable,
        and storing the summary and verdict.

        Args:
            instance_id (str): The workflow instance ID.
            plan (List[Dict[str, Any]]): The current execution plan.
            step (int): The step that was last worked on.
            substep (Optional[float]): The substep that was last worked on (if applicable).
            verdict (str): The final workflow verdict (`completed`, `failed`, or `max_iterations_reached`).
            summary (str): The generated summary of the workflow execution.

        Returns:
            None
        """
        status_updates = []

        if verdict == "completed":
            # Find and validate the step or substep
            step_entry = find_step_in_plan(plan, step, substep)
            if not step_entry:
                raise ValueError(
                    f"Step {step}, Substep {substep} not found in the current plan. Cannot mark as completed."
                )

            # Mark the step or substep as completed
            step_entry["status"] = "completed"
            status_updates.append(
                {"step": step, "substep": substep, "status": "completed"}
            )

            # If it's a substep, check if all sibling substeps are completed
            parent_step = find_step_in_plan(
                plan, step
            )  # Retrieve parent without `substep`
            if parent_step:
                # Ensure "substeps" is a valid list before iteration
                if not isinstance(parent_step.get("substeps"), list):
                    parent_step["substeps"] = []

                all_substeps_completed = all(
                    ss.get("status") == "completed" for ss in parent_step["substeps"]
                )
                if all_substeps_completed:
                    parent_step["status"] = "completed"
                    status_updates.append({"step": step, "status": "completed"})

        # Apply updates in one call
        if status_updates:
            await self.update_plan(
                instance_id=instance_id, plan=plan, status_updates=status_updates
            )

        # Store the final summary and verdict in workflow state
        await self.update_workflow_state(instance_id=instance_id, final_output=summary)

    # TODO: this should be a compensating activity called in the event of an error from any other activity.
    async def update_workflow_state(
        self,
        instance_id: str,
        message: Optional[Dict[str, Any]] = None,
        final_output: Optional[str] = None,
        plan: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Updates the workflow state with a new message, execution plan, or final output.

        Args:
            instance_id (str): The unique identifier of the workflow instance.
            message (Optional[Dict[str, Any]]): A structured message to be added to the workflow state.
            final_output (Optional[str]): The final result of the workflow execution.
            plan (Optional[List[Dict[str, Any]]]): The execution plan associated with the workflow instance.

        Raises:
            ValueError: If the workflow instance ID is not found in the local state.
        """
        workflow_entry = self.state["instances"].get(instance_id)
        if not workflow_entry:
            raise ValueError(
                f"No workflow entry found for instance_id {instance_id} in local state."
            )

        # Only update the provided fields
        if plan is not None:
            workflow_entry["plan"] = plan
        if message is not None:
            serialized_message = LLMWorkflowMessage(**message).model_dump(mode="json")

            # Update workflow state messages
            workflow_entry["messages"].append(serialized_message)
            workflow_entry["last_message"] = serialized_message

            # Update the local chat history
            self.memory.add_message(message)

        if final_output is not None:
            workflow_entry["output"] = final_output
            workflow_entry["end_time"] = datetime.now().isoformat()

        # Persist updated state
        self.save_state()

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
