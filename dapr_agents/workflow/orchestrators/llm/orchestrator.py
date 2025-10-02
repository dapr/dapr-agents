import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dapr.ext.workflow import DaprWorkflowContext
from pydantic import Field

from dapr_agents.workflow.decorators import message_router, task, workflow
from dapr_agents.workflow.orchestrators.base import OrchestratorWorkflowBase
from dapr_agents.workflow.orchestrators.llm.prompts import (
    NEXT_STEP_PROMPT,
    PROGRESS_CHECK_PROMPT,
    SUMMARY_GENERATION_PROMPT,
    TASK_INITIAL_PROMPT,
    TASK_PLANNING_PROMPT,
)
from dapr_agents.workflow.orchestrators.llm.schemas import (
    AgentTaskResponse,
    BroadcastMessage,
    IterablePlanStep,
    NextStep,
    ProgressCheckOutput,
    TriggerAction,
    InternalTriggerAction,
    schemas,
)
from dapr_agents.workflow.orchestrators.llm.state import (
    LLMWorkflowEntry,
    LLMWorkflowMessage,
    LLMWorkflowState,
    PlanStep,
    TaskResult,
)
from dapr_agents.workflow.orchestrators.llm.utils import (
    find_step_in_plan,
    restructure_plan,
    update_step_statuses,
)
from dapr_agents.memory import ConversationDaprStateMemory

logger = logging.getLogger(__name__)


class LLMOrchestrator(OrchestratorWorkflowBase):
    """
    Implements an agentic workflow where an LLM dynamically selects the next speaker.
    The workflow iterates through conversations, updating its state and persisting messages.

    Uses the `continue_as_new` pattern to restart the workflow with updated input at each iteration.
    """

    workflow_instance_id: Optional[str] = Field(
        default=None,
        description="The current workflow instance ID for this orchestrator.",
    )
    memory: ConversationDaprStateMemory = Field(
        default_factory=lambda: ConversationDaprStateMemory(
            store_name="workflowstatestore", session_id="orchestrator_session"
        ),
        description="Persistent memory with session-based state hydration.",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes and configures the LLM-based workflow service.
        """

        # Call OrchestratorWorkflowBase's model_post_init first to initialize state store and other dependencies
        # This will properly load state from storage if it exists
        super().model_post_init(__context)

        self._workflow_name = "OrchestratorWorkflow"
        # TODO(@Sicoyle): fix this later!!
        self._is_orchestrator = True  # Flag for PubSub deduplication to prevent orchestrator workflows from being triggered multiple times

        if not self.state:
            logger.debug("No state found, initializing empty state")
            self.state = {"instances": {}}
        else:
            logger.debug(f"State loaded successfully: {self.state}")

        # Load the current workflow instance ID from state using session_id)
        if self.state and self.state.get("instances"):
            logger.debug(f"Found {len(self.state['instances'])} instances in state")

            current_session_id = self.memory.session_id
            for instance_id, instance_data in self.state["instances"].items():
                stored_workflow_name = instance_data.get("workflow_name")
                stored_session_id = instance_data.get("session_id")
                logger.debug(
                    f"Instance {instance_id}: workflow_name={stored_workflow_name}, session_id={stored_session_id}, current_workflow_name={self._workflow_name}, current_session_id={current_session_id}"
                )
                if (
                    stored_workflow_name == self._workflow_name
                    and stored_session_id == current_session_id
                ):
                    self.workflow_instance_id = instance_id
                    logger.debug(
                        f"Loaded current workflow instance ID from state using session_id: {instance_id}"
                    )
                    break
        else:
            logger.debug("No instances found in state or state is empty")

        # Sync workflow state with Dapr runtime after loading
        # This ensures our database reflects the actual state of resumed workflows
        self._sync_workflow_state_after_startup()

    def _convert_plan_objects_to_dicts(
        self, plan_objects: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert plan objects to dictionaries for JSON serialization.
        Handles both Pydantic models and regular dictionaries.
        """
        result = []
        for obj in plan_objects:
            if hasattr(obj, "model_dump"):
                # Pydantic model
                result.append(obj.model_dump())
            elif isinstance(obj, dict):
                # Already a dictionary
                result.append(obj)
            else:
                # Fallback: try to convert to dict
                result.append(dict(obj) if hasattr(obj, "__dict__") else obj)
        return result

    def _does_workflow_exist(self, instance_id: str) -> bool:
        """
        Check if a workflow instance exists and is accessible via the Dapr client.

        This function attempts to retrieve the workflow metadata from Dapr. A successful
        response indicates the workflow exists in Dapr's state store, while failures
        (e.g., not found errors) indicate the workflow is no longer accessible.

        Args:
            instance_id (str): The workflow instance ID to check

        Returns:
            bool: True if the workflow exists and is accessible, False if not found or on error
        """
        try:
            # Use Dapr client to get workflow instance status
            response = self._dapr_client.get_workflow(instance_id=instance_id)
            # If we get a response, the workflow exists and is accessible
            return response is not None
        except Exception as e:
            logger.debug(f"Workflow {instance_id} not found or not accessible: {e}")
            return False

    @message_router
    async def handle_external_trigger(self, message: TriggerAction):
        """
        Handle external TriggerAction messages from end users.
        This starts a new orchestrator workflow.
        """
        try:
            logger.info(f"External trigger received: {message.task}")
            await self.run_and_monitor_workflow_async(
                workflow="OrchestratorWorkflow", input=message
            )
        except Exception as e:
            logger.error(f"Error handling external trigger: {e}", exc_info=True)

    @workflow(name="OrchestratorWorkflow")
    # TODO: set retry policies on the activities!
    # TODO: utilize prompt verdict value of failed as we do not currently use.
    # https://github.com/dapr/dapr-agents/pull/136#discussion_r2175751545
    def main_workflow(self, ctx: DaprWorkflowContext, message: TriggerAction):
        """
        Executes an LLM-driven agentic workflow where the next agent is dynamically selected
        based on task progress. Runs for up to `self.max_iterations` turns, then summarizes.

        Args:
            ctx (DaprWorkflowContext): The workflow execution context.
            message (TriggerAction): Contains the current `task`.

        Returns:
            str: The final summary when the workflow terminates.

        Raises:
            RuntimeError: If the workflow ends unexpectedly without a final summary.
        """
        # Step 1: Retrieve initial task and ensure state entry exists
        task = message.get("task")
        instance_id = ctx.instance_id
        self.state.setdefault("instances", {}).setdefault(
            instance_id, LLMWorkflowEntry(input=task).model_dump(mode="json")
        )
        # Initialize plan as empty list - it will be set after turn 1
        plan = []
        final_summary: Optional[str] = None

        # Single loop from turn 1 to max_iterations
        for turn in range(1, self.max_iterations + 1):
            if not ctx.is_replaying:
                logger.debug(
                    f"Workflow turn {turn}/{self.max_iterations} (Instance ID: {instance_id})"
                )

            # Get available agents
            agents = yield ctx.call_activity(self.get_available_agents)

            # On turn 1, atomically generate plan and broadcast task
            if turn == 1:
                if not ctx.is_replaying:
                    logger.info(f"Initial message from User -> {self.name}")

                init_result = yield ctx.call_activity(
                    self.initialize_workflow_with_plan,
                    input={
                        "instance_id": instance_id,
                        "task": task,
                        "agents": agents,
                        "wf_time": ctx.current_utc_datetime.isoformat(),
                    },
                )
                logger.info(f"Workflow initialized with plan: {init_result['status']}")
                plan = init_result["plan"]

            # Determine next step and dispatch
            # Plan is now always a list of dictionaries after turn 1
            plan_objects = plan if plan else []

            # If plan is empty, read from workflow state
            if not plan_objects:
                plan_objects = self.state["instances"][instance_id].get("plan", [])
                plan = plan_objects
            next_step = yield ctx.call_activity(
                self.generate_next_step,
                input={
                    "task": task,
                    "agents": agents,
                    "plan": json.dumps(
                        self._convert_plan_objects_to_dicts(plan_objects), indent=2
                    ),
                    "next_step_schema": schemas.next_step,
                },
            )
            # Additional Properties from NextStep
            next_agent = next_step["next_agent"]
            instruction = next_step["instruction"]
            step_id = next_step.get("step", None)
            substep_id = next_step.get("substep", None)

            # Validate Step Before Proceeding
            valid_step = yield ctx.call_activity(
                self.validate_next_step,
                input={
                    "instance_id": instance_id,
                    "plan": self._convert_plan_objects_to_dicts(plan_objects),
                    "step": step_id,
                    "substep": substep_id,
                },
            )

            if valid_step:
                # Atomically execute agent task and mark step as in_progress
                execution_result = yield ctx.call_activity(
                    self.execute_agent_task_with_progress_tracking,
                    input={
                        "instance_id": instance_id,
                        "next_agent": next_agent,
                        "step_id": step_id,
                        "substep_id": substep_id,
                        "instruction": instruction,
                        "task": task,
                        "plan_objects": self._convert_plan_objects_to_dicts(
                            plan_objects
                        ),
                    },
                )
                plan_objects = execution_result["plan"]

                # Wait for agent response or timeout
                if not ctx.is_replaying:
                    logger.debug(f"Waiting for {next_agent}'s response...")

                event_data = ctx.wait_for_external_event("AgentTaskResponse")
                timeout_task = ctx.create_timer(timedelta(seconds=self.timeout))
                any_results = yield self.when_any([event_data, timeout_task])

                # Handle Agent Response or Timeout
                if any_results == timeout_task:
                    if not ctx.is_replaying:
                        logger.warning(
                            f"Agent response timed out (Iteration: {turn}, Instance ID: {instance_id})."
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

                # Atomically process agent response, update history, check progress, and update plan
                response_result = yield ctx.call_activity(
                    self.process_agent_response_with_progress,
                    input={
                        "instance_id": instance_id,
                        "agent": next_agent,
                        "step_id": step_id,
                        "substep_id": substep_id,
                        "task_results": task_results,
                        "task": task,
                        "plan_objects": self._convert_plan_objects_to_dicts(
                            plan_objects
                        ),
                    },
                )

                # Update local variables with results
                plan_objects = response_result["plan"]
                verdict = response_result["verdict"]
                if not ctx.is_replaying:
                    logger.debug(f"Progress verdict: {verdict}")
                    logger.debug(f"Status updates: {response_result['status_updates']}")
                    logger.debug(f"Plan updates: {response_result['plan_updates']}")

                # Update the plan variable to reflect the current state
                plan = plan_objects
            else:
                if not ctx.is_replaying:
                    logger.warning(
                        f"Invalid step {step_id}/{substep_id} in plan for instance {instance_id}. Retrying..."
                    )
                # Recovery Task: No updates, just iterate again
                verdict = "continue"
                task_results = {
                    "name": self.name,
                    "role": "user",
                    "content": f"Step {step_id}, Substep {substep_id} does not exist in the plan. Adjusting workflow...",
                }

            # Process progress suggestions and next iteration count
            if verdict != "continue" or turn == self.max_iterations:
                if not ctx.is_replaying:
                    finale = (
                        "max_iterations_reached"
                        if turn == self.max_iterations
                        else verdict
                    )
                    logger.info(f"Ending workflow with verdict: {finale}")

                # Atomically generate summary and finalize workflow
                final_summary = yield ctx.call_activity(
                    self.finalize_workflow_with_summary,
                    input={
                        "instance_id": instance_id,
                        "task": task,
                        "verdict": verdict,
                        "plan_objects": self._convert_plan_objects_to_dicts(
                            plan_objects
                        ),
                        "step_id": step_id,
                        "substep_id": substep_id,
                        "agent": next_agent,
                        "result": task_results["content"],
                        "wf_time": ctx.current_utc_datetime.isoformat(),
                    },
                )

                # Return the final summary - this should terminate the workflow
                if not ctx.is_replaying:
                    logger.info(f"Workflow {instance_id} finalized.")
                return final_summary
            else:
                # --- PREPARE NEXT TURN ---
                task = task_results["content"]

        # Should never reach here
        raise RuntimeError(f"OrchestratorWorkflow {instance_id} exited without summary")

    @task
    def get_available_agents(self) -> str:
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

    @task(description=NEXT_STEP_PROMPT)
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
            NextStep: A structured response with the next agent, an instruction, and step ids.
        """
        # Use the original prompt template
        prompt = NEXT_STEP_PROMPT.format(
            task=task, agents=agents, plan=plan, next_step_schema=next_step_schema
        )

        # Call LLM with prompt
        response = self.llm.generate(
            inputs=[{"role": "user", "content": prompt}],
            response_format=NextStep,
            structured_mode="json",
        )

        # Parse the response
        if hasattr(response, "choices") and response.choices:
            # If it's still a raw response, parse it
            next_step_data = response.choices[0].message.content
            logger.debug(f"Next step generation response: {next_step_data}")
            next_step_dict = json.loads(next_step_data)
            return NextStep(**next_step_dict)
        else:
            # If it's already a Pydantic model
            return response

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

    # ============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS REQUIRED BY OrchestratorWorkflowBase
    # ============================================================================

    async def broadcast_message_to_agents(self, **kwargs) -> None:
        """
        Broadcast a message to all registered agents.
        Required by OrchestratorWorkflowBase abstract method.
        """
        instance_id = kwargs.get("instance_id")
        message = kwargs.get("message")
        if instance_id and message:
            await self.broadcast_message_to_agents_internal(instance_id, message)

    async def trigger_agent(self, name: str, instance_id: str, **kwargs) -> None:
        """
        Trigger a specific agent to perform an action.
        Required by OrchestratorWorkflowBase abstract method.
        """
        step = kwargs.get("step")
        substep = kwargs.get("substep")
        instruction = kwargs.get("instruction")
        plan = kwargs.get("plan", [])

        if step is not None and instruction:
            await self.trigger_agent_internal(
                instance_id=instance_id,
                name=name,
                step=step,
                substep=substep,
                instruction=instruction,
                plan=plan,
            )

    # NOTE: The remaining @task decorated functions handle specific workflow activities

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
        logger.debug(f"Updating plan for instance {instance_id}")

        # Step 1: Apply status updates directly to `plan`
        if status_updates:
            logger.info(f"Applying {len(status_updates)} status updates to plan")
            for update in status_updates:
                step_id = update["step"]
                substep_id = update.get("substep")
                new_status = update["status"]

                logger.info(
                    f"Updating step {step_id}, substep {substep_id} to '{new_status}'"
                )
                step_entry = find_step_in_plan(plan, step_id, substep_id)
                if not step_entry:
                    error_msg = f"Step {step_id}, Substep {substep_id} not found in the current plan."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Apply status update
                step_entry["status"] = new_status
                logger.info(
                    f"Successfully updated status of step {step_id}, substep {substep_id} to '{new_status}'"
                )

        # Step 2: Apply plan restructuring updates (if provided)
        if plan_updates:
            plan = restructure_plan(plan, plan_updates)
            logger.debug(
                f"Applied restructuring updates for {len(plan_updates)} steps."
            )

        # Step 3: Apply global consistency checks for statuses
        plan = update_step_statuses(plan)

        # Save to state and update workflow
        await self.update_workflow_state(instance_id=instance_id, plan=plan)

        logger.info(f"Plan successfully updated for instance {instance_id}")

    @task
    async def finish_workflow(
        self,
        instance_id: str,
        plan: List[Dict[str, Any]],
        step: int,
        substep: Optional[float],
        verdict: str,
        summary: str,
        wf_time: str,
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
        await self.update_workflow_state(
            instance_id=instance_id, wf_time=wf_time, final_output=summary
        )

    @task
    async def initialize_workflow_with_plan(
        self, instance_id: str, task: str, agents: str, wf_time: str
    ) -> Dict[str, Any]:
        """
        Atomically generates a plan and broadcasts it to all agents.
        If a plan already exists in state, it will be reused (state hydration).

        Args:
            instance_id (str): The workflow instance ID.
            task (str): The task description.
            agents (str): Formatted list of available agents.
            wf_time (str): Workflow timestamp.

        Returns:
            Dict containing the generated plan and broadcast status
        """
        try:
            # Look for existing plan using session_id
            existing_plan = None
            for stored_instance_id, instance_data in self.state.get(
                "instances", {}
            ).items():
                stored_session_id = instance_data.get("session_id")
                if stored_session_id == self.memory.session_id:
                    existing_plan = instance_data.get("plan", [])
                    logger.debug(
                        f"Found existing plan for session_id {self.memory.session_id} in instance {stored_instance_id}"
                    )
                    break

            if existing_plan:
                logger.debug(
                    f"Found existing plan in workflow state, reusing it: {len(existing_plan)} steps"
                )
                plan_objects = existing_plan
            else:
                # Generate new plan using the LLM
                logger.debug(
                    "No existing plan found in workflow state, generating new plan"
                )
                response = self.llm.generate(
                    messages=[
                        {
                            "role": "user",
                            "content": TASK_PLANNING_PROMPT.format(
                                task=task,
                                agents=agents,
                                plan_schema=schemas.plan,
                            ),
                        }
                    ],
                    response_format=IterablePlanStep,
                    structured_mode="json",
                )

                # Parse the response - now we get a Pydantic model directly
                if hasattr(response, "choices") and response.choices:
                    # If it's still a raw response, parse it
                    plan_data = response.choices[0].message.content
                    logger.debug(f"Plan generation response: {plan_data}")
                    plan_dict = json.loads(plan_data)
                    # Convert raw dictionaries to Pydantic models
                    plan_objects = [
                        PlanStep(**step_dict)
                        for step_dict in plan_dict.get("objects", [])
                    ]
                else:
                    # If it's already a Pydantic model
                    plan_objects = (
                        response.objects if hasattr(response, "objects") else []
                    )
                    logger.debug(f"Plan generation response (Pydantic): {plan_objects}")

            # Format and broadcast message
            plan_dicts = self._convert_plan_objects_to_dicts(plan_objects)
            formatted_message = TASK_INITIAL_PROMPT.format(
                task=task, agents=agents, plan=json.dumps(plan_dicts, indent=2)
            )

            if not existing_plan:
                await self.update_workflow_state(
                    instance_id=instance_id, plan=plan_dicts, wf_time=wf_time
                )

                # Store the workflow instance ID for session-based state rehydration
                self.workflow_instance_id = instance_id
                logger.debug(f"Stored workflow instance ID: {instance_id}")

            # Broadcast to agents
            await self.broadcast_message_to_agents_internal(
                instance_id=instance_id, message=formatted_message
            )

            return {"plan": plan_dicts, "broadcast_sent": True, "status": "success"}

        except Exception as e:
            logger.error(f"Failed to initialize workflow: {e}")
            # Rollback: clear any partial state
            await self.rollback_workflow_initialization(instance_id)
            raise

    @task
    async def execute_agent_task_with_progress_tracking(
        self,
        instance_id: str,
        next_agent: str,
        step_id: int,
        substep_id: Optional[float],
        instruction: str,
        task: str,
        plan_objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Atomically executes agent task and marks step as in_progress.

        Args:
            instance_id (str): The workflow instance ID.
            next_agent (str): The agent to trigger.
            step_id (int): The step number.
            substep_id (Optional[float]): The substep number.
            instruction (str): The instruction for the agent.
            task (str): The current task description.
            plan_objects (List[Dict[str, Any]]): The current plan.

        Returns:
            Dict containing updated plan and status
        """
        try:
            # Trigger agent and mark step as in_progress
            updated_plan = await self.trigger_agent_internal(
                instance_id=instance_id,
                name=next_agent,
                step=step_id,
                substep=substep_id,
                instruction=instruction,
                plan=plan_objects,
            )

            return {"plan": updated_plan, "status": "agent_triggered"}

        except Exception as e:
            logger.error(f"Failed to execute agent task: {e}")
            # Rollback: revert step status
            await self.rollback_agent_trigger(instance_id, step_id, substep_id)
            raise

    @task
    async def process_agent_response_with_progress(
        self,
        instance_id: str,
        agent: str,
        step_id: int,
        substep_id: Optional[float],
        task_results: Dict[str, Any],
        task: str,
        plan_objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Atomically processes agent response, updates history, checks progress, and updates plan.

        Args:
            instance_id (str): The workflow instance ID.
            agent (str): The agent name.
            step_id (int): The step number.
            substep_id (Optional[float]): The substep number.
            task_results (Dict[str, Any]): The agent's response.
            task (str): The current task description.
            plan_objects (List[Dict[str, Any]]): The current plan.

        Returns:
            Dict containing updated plan, verdict, and status updates
        """
        try:
            # Step 1: Update task history
            await self.update_task_history_internal(
                instance_id=instance_id,
                agent=agent,
                step=step_id,
                substep=substep_id,
                results=task_results,
                plan=plan_objects,
            )

            # Step 2: Check progress using LLM directly
            progress_response = self.llm.generate(
                messages=[
                    {
                        "role": "user",
                        "content": PROGRESS_CHECK_PROMPT.format(
                            task=task,
                            plan=json.dumps(
                                self._convert_plan_objects_to_dicts(plan_objects),
                                indent=2,
                            ),
                            step=step_id,
                            substep=substep_id if substep_id is not None else "N/A",
                            results=task_results["content"],
                            progress_check_schema=schemas.progress_check,
                        ),
                    }
                ],
                response_format=ProgressCheckOutput,
                structured_mode="json",
            )

            # Parse the response - now we get a Pydantic model directly
            if hasattr(progress_response, "choices") and progress_response.choices:
                # If it's still a raw response, parse it
                progress_data = progress_response.choices[0].message.content
                logger.debug(f"Progress check response: {progress_data}")
                progress_dict = json.loads(progress_data)
                progress = ProgressCheckOutput(**progress_dict)
            else:
                # If it's already a Pydantic model
                progress = progress_response
                logger.debug(f"Progress check response (Pydantic): {progress}")

            # Step 3: Apply plan updates atomically
            verdict = progress.verdict
            status_updates = progress.plan_status_update or []
            plan_updates = progress.plan_restructure or []

            # Convert Pydantic models to dictionaries for JSON serialization
            status_updates_dicts = [
                update.model_dump() if hasattr(update, "model_dump") else update
                for update in status_updates
            ]
            plan_updates_dicts = [
                update.model_dump() if hasattr(update, "model_dump") else update
                for update in plan_updates
            ]

            if status_updates or plan_updates:
                updated_plan = await self.update_plan_internal(
                    instance_id=instance_id,
                    plan=plan_objects,
                    status_updates=status_updates_dicts,
                    plan_updates=plan_updates_dicts,
                )
            else:
                updated_plan = plan_objects

            return {
                "plan": updated_plan,
                "verdict": verdict,
                "status_updates": status_updates_dicts,
                "plan_updates": plan_updates_dicts,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Failed to process agent response: {e}")
            # Rollback: revert task history and plan changes
            await self.rollback_agent_response_processing(
                instance_id, agent, step_id, substep_id
            )

            # Save failure state to workflow state
            from datetime import timezone

            await self.update_workflow_state(
                instance_id=instance_id,
                message={
                    "name": agent,
                    "role": "system",
                    "content": f"Failed to process agent response: {str(e)}",
                    "step": step_id,
                    "substep": substep_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Return a failure response to prevent workflow from continuing
            return {
                "plan": plan_objects,
                "verdict": "failed",
                "status_updates": [],
                "plan_updates": [],
                "status": "failed",
            }

    @task
    async def finalize_workflow_with_summary(
        self,
        instance_id: str,
        task: str,
        verdict: str,
        plan_objects: List[Dict[str, Any]],
        step_id: int,
        substep_id: Optional[float],
        agent: str,
        result: str,
        wf_time: str,
    ) -> str:
        """
        Atomically generates summary and finalizes workflow.

        Args:
            instance_id (str): The workflow instance ID.
            task (str): The original task description.
            verdict (str): The final verdict.
            plan_objects (List[Dict[str, Any]]): The current plan.
            step_id (int): The last step worked on.
            substep_id (Optional[float]): The last substep worked on.
            agent (str): The last agent that worked.
            result (str): The last result.
            wf_time (str): Workflow timestamp.

        Returns:
            Final summary string
        """
        try:
            # Step 1: Generate summary using LLM directly
            summary_response = self.llm.generate(
                messages=[
                    {
                        "role": "user",
                        "content": SUMMARY_GENERATION_PROMPT.format(
                            task=task,
                            verdict=verdict,
                            plan=json.dumps(
                                self._convert_plan_objects_to_dicts(plan_objects),
                                indent=2,
                            ),
                            step=step_id,
                            substep=substep_id if substep_id is not None else "N/A",
                            agent=agent,
                            result=result,
                        ),
                    }
                ],
            )

            # Parse the response - handle both raw responses and direct content
            if hasattr(summary_response, "choices") and summary_response.choices:
                # If it's still a raw response, parse it
                summary = summary_response.choices[0].message.content
                logger.debug(f"Summary generation response: {summary}")
            else:
                # If it's already processed content
                summary = str(summary_response)
                logger.debug(f"Summary generation response (processed): {summary}")

            # Step 2: Finalize workflow with plan updates
            await self.finish_workflow_internal(
                instance_id=instance_id,
                plan=plan_objects,
                step=step_id,
                substep=substep_id,
                verdict=verdict,
                summary=summary,
                wf_time=wf_time,
            )

            return summary

        except Exception as e:
            logger.error(f"Failed to finalize workflow: {e}")
            # Rollback: ensure workflow state is consistent
            await self.rollback_workflow_finalization(instance_id)
            raise

    # TODO: this should be a compensating activity called in the event of an error from any other activity.
    async def update_workflow_state(
        self,
        instance_id: str,
        message: Optional[Dict[str, Any]] = None,
        final_output: Optional[str] = None,
        plan: Optional[List[Dict[str, Any]]] = None,
        wf_time: Optional[str] = None,
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
            if wf_time is not None:
                workflow_entry["end_time"] = wf_time

        # Store workflow instance ID, workflow name, and session_id for session-based state rehydration
        workflow_entry["workflow_instance_id"] = instance_id
        workflow_entry["workflow_name"] = self._workflow_name
        workflow_entry["session_id"] = self.memory.session_id

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

        except Exception as e:
            logger.exception(f"Error processing agent response: {e}", exc_info=True)

    async def broadcast_message_to_agents_internal(
        self, instance_id: str, message: str
    ) -> None:
        """
        Internal helper for broadcasting messages to agents.
        """
        logger.info(f"Broadcasting message to all agents (Instance ID: {instance_id})")

        # Create broadcast message
        broadcast_msg = BroadcastMessage(content=message, name=self.name, role="user")

        # Add workflow instance ID to metadata
        broadcast_msg._message_metadata = {
            "workflow_instance_id": instance_id,
            "source": self.name,
            "type": "BroadcastMessage",
        }

        # Send to beacon_channel topic
        await self.send_message_to_agent(name="beacon_channel", message=broadcast_msg)

    async def trigger_agent_internal(
        self,
        instance_id: str,
        name: str,
        step: int,
        substep: Optional[float],
        instruction: str,
        plan: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Internal helper for triggering agents and updating plan status.
        """
        logger.info(
            f"Triggering agent {name} for step {step}, substep {substep} (Instance ID: {instance_id})"
        )

        # Get the workflow entry from self.state
        workflow_entry = self.state["instances"].get(instance_id)
        if not workflow_entry:
            raise ValueError(f"No workflow entry found for instance_id: {instance_id}")

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
        logger.debug(f"Marked step {step}, substep {substep} as 'in_progress'")

        # Apply global status updates to maintain consistency
        updated_plan = update_step_statuses(plan)

        # Save updated plan state
        await self.update_workflow_state(instance_id=instance_id, plan=updated_plan)

        # Send message to agent with specific task instruction
        await self.send_message_to_agent(
            name=name,
            message=InternalTriggerAction(
                task=instruction, workflow_instance_id=instance_id
            ),
        )

        return updated_plan

    async def update_task_history_internal(
        self,
        instance_id: str,
        agent: str,
        step: int,
        substep: Optional[float],
        results: Dict[str, Any],
        plan: List[Dict[str, Any]],
    ):
        """
        Internal helper for updating task history.
        """
        logger.debug(
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

    async def update_plan_internal(
        self,
        instance_id: str,
        plan: List[Dict[str, Any]],
        status_updates: Optional[List[Dict[str, Any]]] = None,
        plan_updates: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Internal helper for updating the execution plan.
        """
        logger.debug(f"Updating plan for instance {instance_id}")

        # Step 1: Apply status updates directly to `plan`
        if status_updates:
            logger.info(f"Applying {len(status_updates)} status updates to plan")
            for update in status_updates:
                step_id = update["step"]
                substep_id = update.get("substep")
                new_status = update["status"]

                logger.info(
                    f"Updating step {step_id}, substep {substep_id} to '{new_status}'"
                )
                step_entry = find_step_in_plan(plan, step_id, substep_id)
                if not step_entry:
                    error_msg = f"Step {step_id}, Substep {substep_id} not found in the current plan."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Apply status update
                step_entry["status"] = new_status
                logger.info(
                    f"Successfully updated status of step {step_id}, substep {substep_id} to '{new_status}'"
                )

        # Step 2: Apply plan restructuring updates (if provided)
        if plan_updates:
            plan = restructure_plan(plan, plan_updates)
            logger.debug(
                f"Applied restructuring updates for {len(plan_updates)} steps."
            )

        # Step 3: Apply global consistency checks for statuses
        plan = update_step_statuses(plan)

        # Save to state and update workflow
        await self.update_workflow_state(instance_id=instance_id, plan=plan)

        logger.info(f"Plan successfully updated for instance {instance_id}")
        return plan

    async def finish_workflow_internal(
        self,
        instance_id: str,
        plan: List[Dict[str, Any]],
        step: int,
        substep: Optional[float],
        verdict: str,
        summary: str,
        wf_time: str,
    ):
        """
        Internal helper for finalizing workflow.
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
            await self.update_plan_internal(
                instance_id=instance_id, plan=plan, status_updates=status_updates
            )

        # Store the final summary and verdict in workflow state
        await self.update_workflow_state(
            instance_id=instance_id, wf_time=wf_time, final_output=summary
        )

    # ============================================================================
    # ROLLBACK AND COMPENSATION METHODS
    # ============================================================================

    async def rollback_workflow_initialization(self, instance_id: str):
        """
        Rollback workflow initialization by clearing partial state.
        """
        try:
            if instance_id in self.state["instances"]:
                # Clear the plan if it was partially created
                self.state["instances"][instance_id]["plan"] = []
                self.save_state()
                logger.info(f"Rolled back workflow initialization for {instance_id}")
        except Exception as e:
            logger.error(f"Failed to rollback workflow initialization: {e}")

    async def rollback_agent_trigger(
        self, instance_id: str, step_id: int, substep_id: Optional[float]
    ):
        """
        Rollback agent trigger by reverting step status.
        """
        try:
            workflow_entry = self.state["instances"].get(instance_id)
            if workflow_entry and "plan" in workflow_entry:
                plan = workflow_entry["plan"]
                step_entry = find_step_in_plan(plan, step_id, substep_id)
                if step_entry and step_entry["status"] == "in_progress":
                    step_entry["status"] = "not_started"
                    await self.update_workflow_state(instance_id=instance_id, plan=plan)
                    logger.info(
                        f"Rolled back agent trigger for step {step_id}, substep {substep_id}"
                    )
        except Exception as e:
            logger.error(f"Failed to rollback agent trigger: {e}")

    async def rollback_agent_response_processing(
        self, instance_id: str, agent: str, step_id: int, substep_id: Optional[float]
    ):
        """
        Rollback agent response processing by reverting changes.
        """
        try:
            workflow_entry = self.state["instances"].get(instance_id)
            if workflow_entry:
                # Remove the last task result if it was added
                if "task_history" in workflow_entry and workflow_entry["task_history"]:
                    # Find and remove the last entry for this agent/step
                    task_history = workflow_entry["task_history"]
                    for i in range(len(task_history) - 1, -1, -1):
                        task = task_history[i]
                        if (
                            task.get("agent") == agent
                            and task.get("step") == step_id
                            and task.get("substep") == substep_id
                        ):
                            task_history.pop(i)
                            break

                # Revert step status if it was changed
                if "plan" in workflow_entry:
                    plan = workflow_entry["plan"]
                    step_entry = find_step_in_plan(plan, step_id, substep_id)
                    if step_entry and step_entry["status"] == "completed":
                        step_entry["status"] = "in_progress"
                        await self.update_workflow_state(
                            instance_id=instance_id, plan=plan
                        )

                logger.info(
                    f"Rolled back agent response processing for {agent} at step {step_id}, substep {substep_id}"
                )
        except Exception as e:
            logger.error(f"Failed to rollback agent response processing: {e}")

    async def rollback_workflow_finalization(self, instance_id: str):
        """
        Rollback workflow finalization to ensure consistent state.
        """
        try:
            workflow_entry = self.state["instances"].get(instance_id)
            if workflow_entry:
                # Clear final output if it was set
                if "output" in workflow_entry:
                    workflow_entry["output"] = None
                if "end_time" in workflow_entry:
                    workflow_entry["end_time"] = None

                self.save_state()
                logger.info(f"Rolled back workflow finalization for {instance_id}")
        except Exception as e:
            logger.error(f"Failed to rollback workflow finalization: {e}")

    # ============================================================================
    # COMPENSATION ACTIVITY FOR FAILED COMBINED ACTIVITIES
    # ============================================================================

    @task
    async def compensate_failed_activity(
        self, instance_id: str, failed_activity: str, activity_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compensates for a failed combined activity by rolling back changes and restoring state.

        Args:
            instance_id (str): The workflow instance ID.
            failed_activity (str): The name of the failed activity.
            activity_context (Dict[str, Any]): Context about the failed activity.

        Returns:
            Dict containing compensation status and any recovery actions taken.
        """
        try:
            logger.warning(
                f"Compensating for failed activity: {failed_activity} (Instance: {instance_id})"
            )

            compensation_actions = []

            if failed_activity == "initialize_workflow_with_plan":
                await self.rollback_workflow_initialization(instance_id)
                compensation_actions.append("cleared_partial_plan")

            elif failed_activity == "execute_agent_task_with_progress_tracking":
                step_id = activity_context.get("step_id")
                substep_id = activity_context.get("substep_id")
                if step_id is not None:
                    await self.rollback_agent_trigger(instance_id, step_id, substep_id)
                    compensation_actions.append("reverted_step_status")

            elif failed_activity == "process_agent_response_with_progress":
                agent = activity_context.get("agent")
                step_id = activity_context.get("step_id")
                substep_id = activity_context.get("substep_id")
                if agent and step_id is not None:
                    await self.rollback_agent_response_processing(
                        instance_id, agent, step_id, substep_id
                    )
                    compensation_actions.append("reverted_response_processing")

            elif failed_activity == "finalize_workflow_with_summary":
                await self.rollback_workflow_finalization(instance_id)
                compensation_actions.append("reverted_finalization")

            # Ensure workflow state is consistent after compensation
            await self.ensure_workflow_state_consistency(instance_id)

            return {
                "status": "compensated",
                "failed_activity": failed_activity,
                "compensation_actions": compensation_actions,
                "instance_id": instance_id,
            }

        except Exception as e:
            logger.error(f"Failed to compensate for activity {failed_activity}: {e}")
            return {
                "status": "compensation_failed",
                "failed_activity": failed_activity,
                "error": str(e),
                "instance_id": instance_id,
            }

    async def ensure_workflow_state_consistency(self, instance_id: str):
        """
        Ensures workflow state is consistent after compensation.
        """
        try:
            workflow_entry = self.state["instances"].get(instance_id)
            if not workflow_entry:
                logger.warning(
                    f"No workflow entry found for {instance_id} during consistency check"
                )
                return

            # Ensure plan exists and is valid
            if "plan" not in workflow_entry or not workflow_entry["plan"]:
                workflow_entry["plan"] = []

            # Ensure task_history exists
            if "task_history" not in workflow_entry:
                workflow_entry["task_history"] = []

            # Ensure messages exists
            if "messages" not in workflow_entry:
                workflow_entry["messages"] = []

            # Save the consistent state
            self.save_state()
            logger.info(f"Ensured workflow state consistency for {instance_id}")

        except Exception as e:
            logger.error(f"Failed to ensure workflow state consistency: {e}")

    # ============================================================================
    # ERROR HANDLING WRAPPER FOR COMBINED ACTIVITIES
    # ============================================================================

    async def execute_with_compensation(
        self, activity_func, activity_name: str, instance_id: str, **kwargs
    ) -> Any:
        """
        Executes a combined activity with automatic compensation on failure.

        Args:
            activity_func: The activity function to execute.
            activity_name: The name of the activity for logging and compensation.
            instance_id: The workflow instance ID.
            **kwargs: Arguments to pass to the activity function.

        Returns:
            The result of the activity function.

        Raises:
            Exception: If the activity fails and compensation also fails.
        """
        try:
            return await activity_func(**kwargs)
        except Exception as e:
            logger.error(f"Activity {activity_name} failed: {e}")

            # Prepare context for compensation
            activity_context = {"instance_id": instance_id, "error": str(e), **kwargs}

            # Attempt compensation
            compensation_result = await self.compensate_failed_activity(
                instance_id=instance_id,
                failed_activity=activity_name,
                activity_context=activity_context,
            )

            if compensation_result["status"] == "compensated":
                logger.info(f"Successfully compensated for failed {activity_name}")
            else:
                logger.error(
                    f"Compensation failed for {activity_name}: {compensation_result}"
                )

            # Re-raise the original exception
            raise
