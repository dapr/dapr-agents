import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from dapr.ext.workflow import DaprWorkflowContext  # type: ignore
from pydantic import Field, model_validator

from dapr_agents.agents.base import AgentBase
from dapr_agents.agents.durableagent.state import DurableAgentWorkflowState
from dapr_agents.types import (
    AgentError,
    AssistantMessage,
    LLMChatResponse,
    ToolExecutionRecord,
    ToolMessage,
    UserMessage,
)
from dapr_agents.types.workflow import DaprWorkflowStatus
from dapr_agents.workflow.agentic import AgenticWorkflow
from dapr_agents.workflow.decorators import message_router, task, workflow
from dapr_agents.memory import ConversationDaprStateMemory

from .schemas import (
    AgentTaskResponse,
    BroadcastMessage,
    InternalTriggerAction,
    TriggerAction,
)
from .state import (
    DurableAgentMessage,
    DurableAgentWorkflowEntry,
)

logger = logging.getLogger(__name__)


# TODO(@Sicoyle): Clear up the lines between DurableAgent and AgentWorkflow
class DurableAgent(AgenticWorkflow, AgentBase):
    """
    A conversational AI agent that responds to user messages, engages in discussions,
    and dynamically utilizes external tools when needed.

    The DurableAgent follows an agentic workflow, iterating on responses based on
    contextual understanding, reasoning, and tool-assisted execution. It ensures
    meaningful interactions by selecting the right tools, generating relevant responses,
    and refining outputs through iterative feedback loops.
    """

    agent_topic_name: Optional[str] = Field(
        default=None,
        description="The topic name dedicated to this specific agent, derived from the agent's name if not provided.",
    )
    agent_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata about the agent, including name, role, goal, instructions, and topic name.",
    )
    workflow_instance_id: Optional[str] = Field(
        default=None,
        description="The current workflow instance ID for this agent.",
    )
    memory: ConversationDaprStateMemory = Field(
        default_factory=lambda: ConversationDaprStateMemory(
            store_name="workflowstatestore", session_id="durable_agent_session"
        ),
        description="Persistent memory with session-based state hydration.",
    )

    @model_validator(mode="before")
    def set_agent_and_topic_name(cls, values: dict):
        # Set name to role if name is not provided
        if not values.get("name") and values.get("role"):
            values["name"] = values["role"]

        # Derive agent_topic_name from agent name
        if not values.get("agent_topic_name") and values.get("name"):
            values["agent_topic_name"] = values["name"]

        return values

    def model_post_init(self, __context: Any) -> None:
        """Initializes the workflow with agentic execution capabilities."""
        # Call AgenticWorkflow's model_post_init first to initialize state store and other dependencies
        # This will properly load state from storage if it exists
        super().model_post_init(__context)

        # Name of main Workflow
        # TODO: can this be configurable or dynamic? Would that make sense?
        self._workflow_name = "AgenticWorkflow"

        # Initialize state structure if it doesn't exist
        if not self.state:
            self.state = {"instances": {}}

        # Load the current workflow instance ID from state using session_id
        logger.debug(f"State after loading: {self.state}")
        if self.state and self.state.get("instances"):
            logger.debug(f"Found {len(self.state['instances'])} instances in state")
            for instance_id, instance_data in self.state["instances"].items():
                stored_workflow_name = instance_data.get("workflow_name")
                stored_session_id = instance_data.get("session_id")
                logger.debug(
                    f"Instance {instance_id}: workflow_name={stored_workflow_name}, session_id={stored_session_id}, current_workflow_name={self._workflow_name}, current_session_id={self.memory.session_id}"
                )
                if (
                    stored_workflow_name == self._workflow_name
                    and stored_session_id == self.memory.session_id
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

        # Register the agentic system
        self._agent_metadata = {
            "name": self.name,
            "role": self.role,
            "goal": self.goal,
            "instructions": self.instructions,
            "topic_name": self.agent_topic_name,
            "pubsub_name": self.message_bus_name,
            "orchestrator": False,
        }

        self.register_agentic_system()

    async def run(self, input_data: Union[str, Dict[str, Any]]) -> Any:
        """
        Fire up the workflow, wait for it to complete, then return the final serialized_output.
        Dapr automatically handles resuming any incomplete workflows when the runtime starts.

        Args:
            input_data (Union[str, Dict[str, Any]]): The input for the workflow. Can be a string (task) or a dict.
        Returns:
            Any: The final output from the workflow execution.
        """
        logger.debug(f"DurableAgent.run() called with input: {input_data}")

        # Set up signal handlers for graceful shutdown when using run() method
        self.setup_signal_handlers()

        # Prepare input payload for workflow
        if isinstance(input_data, dict):
            input_payload = input_data
        else:
            input_payload = {"task": input_data}

        try:
            result = await self.run_and_monitor_workflow_async(
                workflow=self._workflow_name,
                input=input_payload,
            )
            return result
        except asyncio.CancelledError:
            logger.warning("Workflow execution was cancelled")
            raise

    @message_router
    @workflow(name="AgenticWorkflow")
    def tool_calling_workflow(self, ctx: DaprWorkflowContext, message: TriggerAction):
        """
        Executes a tool-calling workflow, determining the task source (either an agent or an external user).
        This uses Dapr Workflows to run the agent in a ReAct-style loop until it generates a final answer or reaches max iterations,
        calling tools as needed.

        Args:
            ctx (DaprWorkflowContext): The workflow context for the current execution, providing state and control methods.
            message (TriggerAction): The trigger message containing the task, iteration, and metadata for workflow execution.

        Returns:
            Dict[str, Any]: The final response message when the workflow completes, or None if continuing to the next iteration.
        """
        # Step 1: pull out task + metadata + span context from workflow input through .start, .run(), pubsub invocation
        if isinstance(message, dict):
            task = message.get("task", None)
            metadata = message.get("_message_metadata", {}) or {}
            # Extract OpenTelemetry span context if present
            otel_span_context = message.get("_otel_span_context", None)
            # Extract workflow_instance_id from TriggerAction if present from orchestrator
            if "workflow_instance_id" in message:
                metadata["triggering_workflow_instance_id"] = message[
                    "workflow_instance_id"
                ]
        else:  # This is for if triggered by an orchestrator
            task = getattr(message, "task", None)
            metadata = getattr(message, "_message_metadata", {}) or {}
            # Extract OpenTelemetry span context if present
            otel_span_context = getattr(message, "_otel_span_context", None)
            # Extract workflow_instance_id from TriggerAction if present from orchestrator
            if hasattr(message, "workflow_instance_id"):
                metadata["triggering_workflow_instance_id"] = getattr(
                    message, "workflow_instance_id"
                )
            # Extract source from TriggerAction if present from orchestrator
            if hasattr(message, "source"):
                metadata["source"] = getattr(message, "source")

        triggering_workflow_instance_id = metadata.get(
            "triggering_workflow_instance_id"
        )
        source = self.get_source_or_default(metadata.get("source"))

        # Store workflow instance ID for observability layer to use
        # The observability layer will handle AGENT span creation for resumed workflows
        if otel_span_context:
            # New workflow - store the provided span context (observability layer handles this)
            from dapr_agents.observability.context_storage import store_workflow_context

            instance_context_key = f"__workflow_context_{ctx.instance_id}__"
            store_workflow_context(instance_context_key, otel_span_context)

        # Load the latest state from database to ensure we have up-to-date instance data
        self.load_state()

        if not ctx.is_replaying:
            logger.debug(f"Initial message from {source} -> {self.name}")

        yield ctx.call_activity(
            self.record_initial_entry,
            input={
                "instance_id": ctx.instance_id,
                "input": task or "Triggered without input.",
                "source": source,
                "triggering_workflow_instance_id": triggering_workflow_instance_id,
                "start_time": ctx.current_utc_datetime.isoformat(),
                "trace_context": otel_span_context,
            },
        )

        try:
            for turn in range(1, self.max_iterations + 1):
                if not ctx.is_replaying:
                    logger.debug(
                        f"Workflow turn {turn}/{self.max_iterations} (Instance ID: {ctx.instance_id})"
                    )

                # Generate Response with LLM and atomically save the assistant's response message
                response_message: dict = yield ctx.call_activity(
                    self.call_llm,
                    input={
                        "task": task,
                        "instance_id": ctx.instance_id,
                        "time": ctx.current_utc_datetime.isoformat(),
                    },
                )

                # Handle tool calls response
                tool_calls = response_message.get("tool_calls") or []
                if tool_calls:
                    if not ctx.is_replaying:
                        logger.debug(
                            f"Turn {turn}: executing {len(tool_calls)} tool call(s)"
                        )
                    # fan‑out parallel tool executions
                    parallel = [
                        ctx.call_activity(
                            self.run_tool,
                            input={
                                "tool_call": tc,
                                "instance_id": ctx.instance_id,
                                "time": ctx.current_utc_datetime.isoformat(),
                                "execution_order": i,  # Add ordering information
                            },
                        )
                        for i, tc in enumerate(tool_calls)
                    ]
                    yield self.when_all(parallel)

                    # Prepare for next turn: clear task so that call_llm() uses memory/history
                    task = None
                    continue  # bump to next turn

                # No tool calls → this is your final answer
                break  # exit loop
            else:
                raise AgentError("Workflow ended without producing a final response")

        except Exception as e:
            logger.exception("Workflow error", exc_info=e)
            err_msg = {
                "role": "assistant",
                "content": f"⚠️ Unexpected error: {e}",
            }
            self._save_assistant_message(ctx.instance_id, err_msg)

        # Get the last message from state (this will be the final response)
        final_msg = self._get_last_message_from_state(ctx.instance_id)
        if not final_msg:
            final_msg = {"role": "assistant", "content": "No response generated"}

        # Broadcast the final response if a broadcast topic is set
        if self.broadcast_topic_name:
            yield ctx.call_activity(
                self.broadcast_message_to_agents,
                input={"message": final_msg},
            )

        # Respond to source agent if available
        if source and triggering_workflow_instance_id:
            yield ctx.call_activity(
                self.send_response_back,
                input={
                    "response": final_msg,
                    "target_agent": source,
                    "target_instance_id": triggering_workflow_instance_id,
                },
            )

        yield ctx.call_activity(
            self.finalize_workflow,
            input={
                "instance_id": ctx.instance_id,
                "final_output": final_msg.get("content", ""),
                "time": ctx.current_utc_datetime.isoformat(),
                "triggering_workflow_instance_id": triggering_workflow_instance_id,
            },
        )

        # Set verdict for the workflow instance
        if not ctx.is_replaying:
            verdict = (
                "max_iterations_reached" if turn == self.max_iterations else "completed"
            )
            logger.info(f"Workflow {ctx.instance_id} finalized: {verdict}")

        # Return the final response message
        return final_msg

    @message_router
    @workflow(name="AgenticWorkflow")
    def internal_trigger_workflow(
        self, ctx: DaprWorkflowContext, message: InternalTriggerAction
    ):
        """
        Handles InternalTriggerAction messages by treating them the same as TriggerAction.
        This prevents self-triggering loops while allowing orchestrators to trigger agents.

        Args:
            ctx (DaprWorkflowContext): The workflow context for the current execution.
            message (InternalTriggerAction): The internal trigger message from an orchestrator.

        Returns:
            Dict[str, Any]: The final response message when the workflow completes.
        """
        # Convert InternalTriggerAction to TriggerAction format and delegate to the main workflow
        trigger_message = TriggerAction(
            task=message.task,
            workflow_instance_id=message.workflow_instance_id,
            source="orchestrator",  # Default source for internal triggers
        )
        return self.tool_calling_workflow(ctx, trigger_message)

    def get_source_or_default(self, source: str):
        # Set default source if not provided (for direct run() calls)
        if not source:
            source = "direct"
        return source

    @task
    def record_initial_entry(
        self,
        instance_id: str,
        input: str,
        source: Optional[str],
        triggering_workflow_instance_id: Optional[str],
        start_time: str,  # required to be passed in using the workflow context for deterministic timestamp
        output: str = "",
        trace_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Records the initial workflow entry for a new workflow instance.
        Args:
            instance_id (str): The workflow instance ID.
            input (str): The input task for the workflow.
            source (Optional[str]): The source of the workflow trigger.
            triggering_workflow_instance_id (Optional[str]): The workflow instance ID of the triggering workflow.
            output (str): The output for the workflow entry (default: "").
            start_time (Optional[str]): The start time in ISO format (default: None, will use current time).
            trace_context (Optional[Dict[str, Any]]): OpenTelemetry trace context for workflow resumption.
        """
        # Convert datetime to string for JSON serialization
        if start_time:
            if isinstance(start_time, str):
                start_time_str = start_time
            else:
                start_time_str = start_time.isoformat()
        else:
            start_time_str = datetime.now(timezone.utc).isoformat()

        entry = {
            "input": input,
            "source": source,
            "workflow_instance_id": instance_id,
            "triggering_workflow_instance_id": triggering_workflow_instance_id,
            "workflow_name": self._workflow_name,
            "session_id": self.memory.session_id,
            "start_time": start_time_str,
            "trace_context": trace_context,
            "status": DaprWorkflowStatus.RUNNING.value,
            "messages": [],
            "tool_history": [],
            "end_time": None,
        }
        if "instances" not in self.state:
            self.state["instances"] = {}
        self.state["instances"][instance_id] = entry

    # Note: This is only really needed bc of the in-memory storage solutions.
    # With persistent storage, this is not needed as we rehydrate the conversation state from the database upon app restart.
    def _ensure_instance_exists(
        self,
        instance_id: str,
        input: str,
        triggering_workflow_instance_id: Optional[str] = None,
        time: Optional[datetime] = None,
    ) -> None:
        """Ensure the instance entry exists in the state."""
        if instance_id not in self.state.get("instances", {}):
            if "instances" not in self.state:
                self.state["instances"] = {}

            # Handle time parameter - it might be a datetime object or a string
            if time:
                if isinstance(time, str):
                    start_time = time
                else:
                    start_time = time.isoformat()
            else:
                start_time = datetime.now(timezone.utc).isoformat()

            self.state["instances"][instance_id] = {
                "input": input,
                "start_time": start_time,
                "source": "user_input",
                "workflow_instance_id": instance_id,
                "triggering_workflow_instance_id": triggering_workflow_instance_id,
                "workflow_name": self._workflow_name,
                "session_id": self.memory.session_id,
                "messages": [],
                "tool_history": [],
                "status": DaprWorkflowStatus.RUNNING.value,
                "end_time": None,
                "trace_context": None,
            }

    def _process_user_message(
        self,
        instance_id: str,
        task: Optional[Union[str, Dict[str, Any]]],
        user_message_copy: Optional[Dict[str, Any]],
    ) -> None:
        """Process and save user message to memory and state."""
        if not (task and user_message_copy):
            return

        user_msg = UserMessage(content=user_message_copy.get("content", ""))
        self.memory.add_message(user_msg)

        msg_object = DurableAgentMessage(**user_message_copy)
        inst = self.state["instances"][instance_id]
        inst["messages"].append(msg_object.model_dump(mode="json"))
        inst["last_message"] = msg_object.model_dump(mode="json")
        self.save_state()

    def _call_llm(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate LLM response and return the assistant message."""
        response: LLMChatResponse = self.llm.generate(
            messages=messages,
            tools=self.get_llm_tools(),
            **(
                {"tool_choice": self.tool_choice}
                if self.tool_choice is not None
                else {}
            ),
        )
        response_message = response.get_message()
        if response_message is None:
            raise AgentError("LLM returned no assistant message")

        return response_message.model_dump()

    def _save_assistant_message(
        self, instance_id: str, assistant_message: Dict[str, Any]
    ) -> None:
        """Save assistant message to state with idempotency check."""
        assistant_message["name"] = self.name
        agent_msg = DurableAgentMessage(**assistant_message)

        inst = self.state["instances"][instance_id]
        messages_list = inst["messages"]

        # Check for duplicate by message ID (idempotent for workflow replay)
        message_exists = any(msg.get("id") == agent_msg.id for msg in messages_list)
        if not message_exists:
            messages_list.append(agent_msg.model_dump(mode="json"))
            inst["last_message"] = agent_msg.model_dump(mode="json")
            self.memory.add_message(AssistantMessage(**assistant_message))
            self.save_state()

    def _print_llm_interaction_messages(
        self,
        user_message_copy: Optional[Dict[str, Any]],
        assistant_message: Dict[str, Any],
    ) -> None:
        """Print user and assistant messages for context."""
        # Print user message
        if user_message_copy is not None:
            self.text_formatter.print_message(
                {str(k): v for k, v in user_message_copy.items()}
            )

        # Print assistant message
        self.text_formatter.print_message(assistant_message)

    @task
    async def call_llm(
        self,
        instance_id: str,
        time: datetime,
        task: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Ask the LLM for the assistant's next message.

        Args:
            instance_id (str): The workflow instance ID.
            time (str): The time of the message.
            task: The user's query for this turn (either a string or a dict),
                  or None if this is a follow-up iteration.

        Returns:
            A plain dict of the LLM's response (choices, finish_reason, etc).
            Pydantic models are `.model_dump()`-ed; any other object is coerced via `dict()`.
        """
        # Construct messages using instance-specific chat history instead of global memory
        # This ensures proper message sequence for tool calls and ensures formatting/structure
        messages: List[Dict[str, Any]] = self._construct_messages_with_instance_history(
            instance_id, task or {}
        )
        user_message = self.get_last_message_if_user(messages)

        # Always work with a copy of the user message for safety
        user_message_copy: Optional[Dict[str, Any]] = (
            dict(user_message) if user_message else None
        )

        self._ensure_instance_exists(
            instance_id, task or "No input provided", time=time
        )
        self._process_user_message(instance_id, task, user_message_copy)

        # Generate LLM response and atomically save assistant message
        try:
            assistant_message = self._call_llm(messages)
            self._save_assistant_message(instance_id, assistant_message)
            self._print_llm_interaction_messages(user_message_copy, assistant_message)

            return assistant_message
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            logger.exception(
                f"LLM generation failed in workflow {instance_id}: {error_type} - {error_msg}"
            )
            logger.exception(f"Task: {task}")
            logger.exception(f"Messages count: {len(messages)}")
            logger.exception(f"Tools available: {len(self.get_llm_tools())}")
            logger.exception("Full error details:", exc_info=True)

            raise AgentError(
                f"LLM generation failed in workflow {instance_id}: {error_type} - {error_msg}"
            ) from e

    @task
    def _create_tool_message_objects(self, tool_result: Dict[str, Any]) -> tuple:
        """
        Create ToolMessage and DurableAgentMessage objects from tool result.

        Args:
            tool_result: Dictionary containing tool execution details

        Returns:
            Tuple of (tool_msg, agent_msg, tool_history_entry)
        """
        tool_msg = ToolMessage(
            tool_call_id=tool_result["tool_call_id"],
            name=tool_result["tool_name"],
            content=tool_result["execution_result"],
            role="tool",
        )
        agent_msg = DurableAgentMessage(**tool_msg.model_dump())
        tool_history_entry = ToolExecutionRecord(**tool_result)

        return tool_msg, agent_msg, tool_history_entry

    def _append_tool_message_to_instance(
        self,
        instance_id: str,
        agent_msg: DurableAgentMessage,
        tool_history_entry: ToolExecutionRecord,
    ) -> None:
        """
        Append tool message and history to the instance state.

        Args:
            instance_id: The workflow instance ID
            agent_msg: The DurableAgentMessage object
            tool_history_entry: The ToolExecutionRecord object
        """
        wf_instance = self.state["instances"][instance_id]

        # Check if message already exists (idempotent operation for workflow replay)
        wf_messages = wf_instance["messages"]

        # Check for duplicate by message ID (idempotent for workflow replay)
        message_exists = any(msg.get("id") == agent_msg.id for msg in wf_messages)
        if not message_exists:
            wf_messages.append(agent_msg.model_dump(mode="json"))

        # Check for duplicate tool history entry by tool_call_id
        tool_history = wf_instance["tool_history"]

        tool_exists = any(
            th.get("tool_call_id") == tool_history_entry.tool_call_id
            for th in tool_history
        )
        if not tool_exists:
            tool_history.append(tool_history_entry.model_dump(mode="json"))

    def _update_agent_memory_and_history(
        self, tool_message: ToolMessage, tool_history_entry: ToolExecutionRecord
    ) -> None:
        """
        Update agent's memory and tool history.

        Args:
            tool_message: The ToolMessage object
            tool_history_entry: The ToolExecutionRecord object
        """
        # Update tool history and memory of agent (only if new)
        # Note: Memory updates are handled at workflow level to avoid replay issues
        self.tool_history.append(tool_history_entry)
        # Add the tool message to the agent's memory
        self.memory.add_message(tool_message)

    def _get_last_message_from_state(
        self, instance_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the last message from the instance state.

        Args:
            instance_id: The workflow instance ID

        Returns:
            The last message dict or None if not found
        """
        instance_data = self.state.get("instances", {}).get(instance_id)
        if instance_data is not None:
            return instance_data.get("last_message")
        return None

    @task
    async def run_tool(
        self,
        tool_call: Dict[str, Any],
        instance_id: str,
        time: datetime,
        execution_order: int = 0,
    ) -> Dict[str, Any]:
        """
        Executes a tool call atomically by invoking the specified function with the provided arguments
        and immediately persisting the result to the agent's state and memory.

        Args:
            tool_call (Dict[str, Any]): A dictionary containing tool execution details, including the function name and arguments.
            instance_id (str): The workflow instance ID for state persistence.
            time (str): The current time for state persistence.

        Returns:
            Dict[str, Any]: A dictionary containing the tool call ID, function name, function arguments

        Raises:
            AgentError: If the tool call is malformed or execution fails.
        """
        # Extract function name and raw args
        fn_name = tool_call["function"]["name"]
        raw_args = tool_call["function"].get("arguments", "")

        # Parse JSON arguments (or empty dict)
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as e:
            raise AgentError(f"Invalid JSON in tool args: {e}")

        # Run the tool
        logger.debug(f"Executing tool '{fn_name}' with args: {args}")
        try:
            result = await self.tool_executor.run_tool(fn_name, **args)
        except Exception as e:
            logger.exception(f"Error executing tool '{fn_name}': {e}", exc_info=True)
            raise AgentError(f"Error executing tool '{fn_name}': {e}") from e

        # Create the tool result payload
        tool_result = {
            "tool_call_id": tool_call["id"],
            "tool_name": fn_name,
            "tool_args": args,
            "execution_result": str(result) if result is not None else "",
        }

        # Atomically persist the tool execution result
        # Get existing input or use placeholder
        existing_input = (
            self.state["instances"][instance_id]["input"]
            if instance_id in self.state.get("instances", {})
            else "Tool execution"
        )
        self._ensure_instance_exists(instance_id, existing_input, time=time)
        tool_msg, agent_msg, tool_history_entry = self._create_tool_message_objects(
            tool_result
        )
        self._append_tool_message_to_instance(
            instance_id, agent_msg, tool_history_entry
        )
        self._update_agent_memory_and_history(tool_msg, tool_history_entry)
        self.save_state()
        self.text_formatter.print_message(tool_msg)

        return tool_result

    @task
    async def broadcast_message_to_agents(self, message: Dict[str, Any]):
        """
        Broadcasts it to all registered agents.

        Args:
            message (Dict[str, Any]): A message to append to the workflow state and broadcast to all agents.
        """
        # Format message for broadcasting
        message["role"] = "user"
        message["name"] = self.name
        response_message = BroadcastMessage(**message)

        # Broadcast message to all agents
        await self.broadcast_message(message=response_message)

    @task
    async def send_response_back(
        self, response: Dict[str, Any], target_agent: str, target_instance_id: str
    ):
        """
        Sends a task response back to a target agent within a workflow.

        Args:
            response (Dict[str, Any]): The response payload to be sent.
            target_agent (str): The name of the agent that should receive the response.
            target_instance_id (str): The workflow instance ID associated with the response.

        Raises:
            ValidationError: If the response does not match the expected structure for `AgentTaskResponse`.
        """
        response["role"] = "user"
        response["name"] = self.name
        response["workflow_instance_id"] = target_instance_id
        agent_response = AgentTaskResponse(**response)

        # Send the message to the target agent
        await self.send_message_to_agent(name=target_agent, message=agent_response)

    # TODO: add metrics on workflow run in future here?
    @task
    def finalize_workflow(
        self,
        instance_id: str,
        final_output: str,
        time: str,
        triggering_workflow_instance_id: Optional[str] = None,
    ) -> None:
        """
        Record the final output and end_time in the workflow state.
        """
        # Ensure the instance entry exists
        existing_input = (
            self.state["instances"][instance_id]["input"]
            if instance_id in self.state.get("instances", {})
            else "Workflow completion"
        )
        self._ensure_instance_exists(
            instance_id, existing_input, triggering_workflow_instance_id, time
        )
        instance = self.state["instances"][instance_id]
        instance["output"] = final_output
        # Convert time to string for JSON serialization
        if time:
            if isinstance(time, str):
                instance["end_time"] = time
            else:
                instance["end_time"] = time.isoformat()
        else:
            instance["end_time"] = datetime.now(timezone.utc).isoformat()
        instance["status"] = DaprWorkflowStatus.COMPLETED.value  # Mark as completed
        logger.info(f"Workflow {instance_id} completed successfully")
        self.save_state()

    @message_router(broadcast=True)
    async def process_broadcast_message(self, message: BroadcastMessage):
        """
        Processes a broadcast message by filtering out messages from the same agent,
        storing valid messages in memory, and triggering the agent's workflow if needed.

        Args:
            message (BroadcastMessage): The received broadcast message.

        Returns:
            None: The function updates the agent's memory and triggers a workflow.
        """
        try:
            # Extract metadata safely from message["_message_metadata"]
            metadata = getattr(message, "_message_metadata", {})

            if not isinstance(metadata, dict) or not metadata:
                logger.warning(
                    f"{self.name} received a broadcast message with missing or invalid metadata. Ignoring."
                )
                return

            source = metadata.get("source", "unknown_source")
            message_type = metadata.get("type", "unknown_type")
            message_content = getattr(message, "content", "No Data")
            logger.info(
                f"{self.name} received broadcast message of type '{message_type}' from '{source}'."
            )
            # Ignore messages sent by this agent
            if source == self.name:
                logger.debug(
                    f"{self.name} ignored its own broadcast message of type '{message_type}'."
                )
                return
            # Log and process the valid broadcast message
            logger.debug(
                f"{self.name} processing broadcast message from '{source}'. Content: {message_content}"
            )
            # Store the message in local memory
            self.memory.add_message(message)

            # Define DurableAgentMessage object for state persistence
            agent_msg = DurableAgentMessage(**message.model_dump())

            # Persist to global chat history
            if "chat_history" not in self.state:
                self.state["chat_history"] = []
            self.state["chat_history"].append(agent_msg.model_dump(mode="json"))
            # Save the state after processing the broadcast message
            self.save_state()

            # Trigger agent workflow to respond to the broadcast message
            workflow_instance_id = metadata.get("workflow_instance_id")
            if workflow_instance_id:
                # Create a TriggerAction to start the agent's workflow
                trigger_message = TriggerAction(
                    task=message.content, workflow_instance_id=workflow_instance_id
                )
                trigger_message._message_metadata = {
                    "source": metadata.get("source", "unknown"),
                    "type": "BroadcastMessage",
                    "workflow_instance_id": workflow_instance_id,
                }

                # Start the agent's workflow
                await self.run_and_monitor_workflow_async(
                    workflow="AgenticWorkflow", input=trigger_message
                )

        except Exception as e:
            logger.error(f"Error processing broadcast message: {e}", exc_info=True)

    # TODO: we need to better design context history management. Context engineering is important,
    # and too much context can derail the agent.
    def _construct_messages_with_instance_history(
        self, instance_id: str, input_data: Union[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Construct messages using instance-specific chat history and persistent memory.
        This ensures proper message sequence for tool calls and maintains conversation
        history across workflow executions using the session_id.

        Args:
            instance_id: The workflow instance ID
            input_data: User input, either as a string or dictionary

        Returns:
            List of formatted messages with proper sequence
        """
        additional_context_messages: List[Dict[str, Any]] = []
        if not self.prompt_template:
            raise ValueError(
                "Prompt template must be initialized before constructing messages."
            )

        # Get instance-specific chat history
        if self.state is None:
            logger.warning(
                f"Agent state is None for instance {instance_id}, initializing empty state"
            )
            self.state = {}

        instance_data = self.state.get("instances", {}).get(instance_id)
        if instance_data is not None:
            instance_messages = instance_data.get("messages", [])
        else:
            instance_messages = []

        # Get messages from persistent memory (session-based, cross-workflow)
        persistent_memory_messages = []
        try:
            persistent_memory_messages = self.memory.get_messages()
            logger.info(
                f"Retrieved {len(persistent_memory_messages)} messages for session {self.memory.session_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve persistent memory: {e}")

        # Get long-term memory from workflow state (for broadcast messages and persistent context)
        long_term_memory_data = self.state.get("chat_history", [])
        long_term_memory_messages = []
        for msg in long_term_memory_data:
            if isinstance(msg, dict):
                long_term_memory_messages.append(msg)
            elif hasattr(msg, "model_dump"):
                long_term_memory_messages.append(msg.model_dump())

        # Build chat history with proper context and order
        chat_history = []

        # First add persistent memory and long-term memory as user messages for context
        # This ensures we have cross-workflow context but doesn't interfere with tool state order
        for msg in persistent_memory_messages + long_term_memory_messages:
            msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
            if msg_dict in chat_history:
                continue
            # TODO: We need to properly design session-based memory.
            # Convert tool-related messages to user messages to avoid conversation order issues
            if msg_dict.get("role") in ["tool", "assistant"] and (
                msg_dict.get("tool_calls") or msg_dict.get("tool_call_id")
            ):
                msg_dict = {
                    "role": "user",
                    "content": f"[Previous {msg_dict['role']} message: {msg_dict.get('content', '')}]",
                }
            chat_history.append(msg_dict)

        # Then add instance messages in their original form to maintain tool state
        for msg in instance_messages:
            msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
            if msg_dict in chat_history:
                continue
            chat_history.append(msg_dict)

        # Add additional context memory last (for broadcast-triggered workflows)
        chat_history.extend(additional_context_messages)

        if isinstance(input_data, str):
            formatted_messages = self.prompt_template.format_prompt(
                chat_history=chat_history
            )
            if isinstance(formatted_messages, list):
                user_message = {"role": "user", "content": input_data}
                return formatted_messages + [user_message]
            else:
                return [
                    {"role": "system", "content": formatted_messages},
                    {"role": "user", "content": input_data},
                ]
        elif isinstance(input_data, dict):
            input_vars = dict(input_data)
            if "chat_history" not in input_vars:
                input_vars["chat_history"] = chat_history
            formatted_messages = self.prompt_template.format_prompt(**input_vars)
            if isinstance(formatted_messages, list):
                return formatted_messages
            else:
                return [{"role": "system", "content": formatted_messages}]
        else:
            raise ValueError("Input data must be either a string or dictionary.")
