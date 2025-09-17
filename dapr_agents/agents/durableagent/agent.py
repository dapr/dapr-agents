import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

from dapr.ext.workflow import DaprWorkflowContext  # type: ignore
from pydantic import Field, model_validator

from dapr_agents.agents.base import AgentBase
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

from .schemas import (
    AgentTaskResponse,
    BroadcastMessage,
    TriggerAction,
)
from .state import (
    DurableAgentMessage,
    DurableAgentWorkflowEntry,
    DurableAgentWorkflowState,
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
        self._workflow_name = "ToolCallingWorkflow"

        # Initialize state structure if it doesn't exist
        if not self.state:
            self.state = {}
        if "instances" not in self.state:
            self.state["instances"] = {}
        if "chat_history" not in self.state:
            self.state["chat_history"] = []

        # Load the current workflow instance ID from state if it exists
        logger.info(f"State after loading: {self.state}")
        if self.state and "instances" in self.state:
            logger.info(f"Found {len(self.state['instances'])} instances in state")
            for instance_id, instance_data in self.state["instances"].items():
                stored_workflow_name = instance_data.get("workflow_name")
                logger.info(
                    f"Instance {instance_id}: workflow_name={stored_workflow_name}, current_workflow_name={self._workflow_name}"
                )
                if stored_workflow_name == self._workflow_name:
                    self.workflow_instance_id = instance_id
                    logger.info(
                        f"Loaded current workflow instance ID from state: {instance_id}"
                    )
                    break
        else:
            logger.info("No instances found in state or state is empty")

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

        # Start the runtime if it's not already running
        logger.info("Starting workflow runtime...")
        self.start_runtime()

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
            logger.info("Workflow execution was cancelled")
            raise
        finally:
            # Clean up runtime
            if self.wf_runtime_is_running:
                self.stop_runtime()

    @message_router
    @workflow(name="ToolCallingWorkflow")
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
        # Step 1: pull out task + metadata + span context
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
        else:
            task = getattr(message, "task", None)
            metadata = getattr(message, "_message_metadata", {}) or {}
            # Extract OpenTelemetry span context if present
            otel_span_context = getattr(message, "_otel_span_context", None)
            # Extract workflow_instance_id from TriggerAction if present from orchestrator
            if hasattr(message, "workflow_instance_id"):
                metadata["triggering_workflow_instance_id"] = getattr(
                    message, "workflow_instance_id"
                )

        workflow_instance_id = ctx.instance_id
        triggering_workflow_instance_id = metadata.get(
            "triggering_workflow_instance_id"
        )
        source = metadata.get("source")

        # Set default source if not provided (for direct run() calls)
        if not source:
            source = "direct"

        # Store workflow instance ID for observability layer to use
        # The observability layer will handle AGENT span creation for resumed workflows
        if otel_span_context:
            # New workflow - store the provided span context (observability layer handles this)
            from dapr_agents.observability.context_storage import store_workflow_context

            instance_context_key = f"__workflow_context_{workflow_instance_id}__"
            store_workflow_context(instance_context_key, otel_span_context)

        # Load the latest state from database to ensure we have up-to-date instance data
        self.load_state()

        # Check if this instance already exists in state (from previous runs)
        is_resumed_workflow = workflow_instance_id in self.state.get("instances", {})
        if not is_resumed_workflow:
            # This is a new instance, create a minimal entry
            instance_entry = {
                "input": task,
                "output": "",
                "messages": [],
                # TODO(@Sicoyle): update to leverage deterministic time for replays.
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": None,  # Will be set when workflow completes
                "source": source,
                "workflow_instance_id": workflow_instance_id,
                "triggering_workflow_instance_id": triggering_workflow_instance_id,
                "workflow_name": self._workflow_name,
                "status": DaprWorkflowStatus.RUNNING.value,
                "trace_context": otel_span_context,  # Store original trace context for resumption
            }
            self.state.setdefault("instances", {})[
                workflow_instance_id
            ] = instance_entry
            logger.debug(f"Created new instance entry: {workflow_instance_id}")

            # Immediately save state so graceful shutdown can capture this instance
            self.save_state()
        else:
            logger.debug(
                f"Found existing instance entry for workflow {workflow_instance_id}"
            )
        final_message: Optional[Dict[str, Any]] = None

        if not ctx.is_replaying:
            logger.debug(f"Initial message from {source} -> {self.name}")

        try:
            # Loop up to max_iterations
            for turn in range(1, self.max_iterations + 1):
                if not ctx.is_replaying:
                    logger.debug(
                        f"Workflow turn {turn}/{self.max_iterations} (Instance ID: {workflow_instance_id})"
                    )

                # Step 2: On turn 1, record the initial entry
                if turn == 1:
                    yield ctx.call_activity(
                        self.record_initial_entry,
                        input={
                            "instance_id": workflow_instance_id,
                            "input": task or "Triggered without input.",
                            "source": source,
                            "triggering_workflow_instance_id": triggering_workflow_instance_id,
                            "output": "",
                        },
                    )
                # Step 4: Generate Response with LLM
                response_message: dict = yield ctx.call_activity(
                    self.generate_response,
                    input={"task": task, "instance_id": workflow_instance_id},
                )

                # Step 5: Add the assistant's response message to the chat history
                yield ctx.call_activity(
                    self.append_assistant_message,
                    input={
                        "instance_id": workflow_instance_id,
                        "message": response_message,
                    },
                )

                # Step 6: Handle tool calls response
                tool_calls = response_message.get("tool_calls") or []
                if tool_calls:
                    if not ctx.is_replaying:
                        logger.debug(
                            f"Turn {turn}: executing {len(tool_calls)} tool call(s)"
                        )
                    # fan‑out parallel tool executions
                    parallel = [
                        ctx.call_activity(self.run_tool, input={"tool_call": tc})
                        for tc in tool_calls
                    ]
                    tool_results: List[Dict[str, Any]] = yield self.when_all(parallel)
                    # Add tool results for the next iteration
                    for tr in tool_results:
                        yield ctx.call_activity(
                            self.append_tool_message,
                            input={
                                "instance_id": workflow_instance_id,
                                "tool_result": tr,
                            },
                        )
                    # 🔴 If this was the last turn, stop here—even though there were tool calls
                    if turn == self.max_iterations:
                        final_message = response_message
                        # Make sure content exists and is a string
                        final_message["content"] = final_message.get("content") or ""
                        final_message[
                            "content"
                        ] += "\n\n⚠️ Stopped: reached max iterations."
                        break

                    # Otherwise, prepare for next turn: clear task so that generate_response() uses memory/history
                    task = None
                    continue  # bump to next turn

                # No tool calls → this is your final answer
                final_message = response_message

                # 🔴 If it happened to be the last turn, banner it
                if turn == self.max_iterations:
                    # Again, ensure content is never None
                    final_message["content"] = final_message.get("content") or ""
                    final_message["content"] += "\n\n⚠️ Stopped: reached max iterations."

                break  # exit loop with final_message
            else:
                raise AgentError("Workflow ended without producing a final response")

        except Exception as e:
            logger.exception("Workflow error", exc_info=e)
            final_message = {
                "role": "assistant",
                "content": f"⚠️ Unexpected error: {e}",
            }

        # Step 7: Broadcast the final response if a broadcast topic is set
        if self.broadcast_topic_name:
            yield ctx.call_activity(
                self.broadcast_message_to_agents,
                input={"message": final_message},
            )

        # Respond to source agent if available
        if source and triggering_workflow_instance_id:
            yield ctx.call_activity(
                self.send_response_back,
                input={
                    "response": final_message,
                    "target_agent": source,
                    "target_instance_id": triggering_workflow_instance_id,
                },
            )

        yield ctx.call_activity(
            self.finalize_workflow,
            input={
                "instance_id": workflow_instance_id,
                "final_output": final_message["content"],
            },
        )

        # Set verdict for the workflow instance
        if not ctx.is_replaying:
            verdict = (
                "max_iterations_reached" if turn == self.max_iterations else "completed"
            )
            logger.info(f"Workflow {workflow_instance_id} finalized: {verdict}")

        # Return the final response message
        return final_message

    @task
    def record_initial_entry(
        self,
        instance_id: str,
        input: str,
        source: Optional[str],
        triggering_workflow_instance_id: Optional[str],
        output: str = "",
    ):
        """
        Records the initial workflow entry for a new workflow instance.
        Args:
            instance_id (str): The workflow instance ID.
            input (str): The input task for the workflow.
            source (Optional[str]): The source of the workflow trigger.
            triggering_workflow_instance_id (Optional[str]): The workflow instance ID of the triggering workflow.
            output (str): The output for the workflow entry (default: "").
        """
        entry = DurableAgentWorkflowEntry(
            input=input,
            source=source,
            workflow_instance_id=instance_id,
            triggering_workflow_instance_id=triggering_workflow_instance_id,
            output=output,
            workflow_name=self._workflow_name,
        )
        self.state.setdefault("instances", {})[instance_id] = entry.model_dump(
            mode="json"
        )

    @task
    def get_workflow_entry_info(self, instance_id: str) -> Dict[str, Any]:
        """
        Retrieves the 'source' and 'triggering_workflow_instance_id' for a given workflow instance.

        Args:
            instance_id (str): The workflow instance ID to look up.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'source': The source of the workflow trigger (str or None).
                - 'triggering_workflow_instance_id': The workflow instance ID of the triggering workflow (str or None).

        Raises:
            AgentError: If the entry is not found or invalid.
        """
        # Load state to ensure we have the latest data
        self.load_state()

        workflow_entry = self.state.get("instances", {}).get(instance_id)
        if workflow_entry is not None:
            return {
                "source": workflow_entry.get("source"),
                "triggering_workflow_instance_id": workflow_entry.get(
                    "triggering_workflow_instance_id"
                ),
            }
        raise AgentError(f"No workflow entry found for instance_id={instance_id}")

    @task
    async def generate_response(
        self, instance_id: str, task: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Ask the LLM for the assistant's next message.

        Args:
            instance_id (str): The workflow instance ID.
            task: The user's query for this turn (either a string or a dict),
                  or None if this is a follow-up iteration.

        Returns:
            A plain dict of the LLM's response (choices, finish_reason, etc).
            Pydantic models are `.model_dump()`-ed; any other object is coerced via `dict()`.
        """
        # Construct messages using instance-specific chat history instead of global memory
        # This ensures proper message sequence for tool calls
        messages: List[Dict[str, Any]] = self._construct_messages_with_instance_history(
            instance_id, task or {}
        )
        user_message = self.get_last_message_if_user(messages)

        # Always work with a copy of the user message for safety
        user_message_copy: Optional[Dict[str, Any]] = (
            dict(user_message) if user_message else None
        )

        if task and user_message_copy:
            # Add the new user message to memory only if input_data is provided and user message exists
            user_msg = UserMessage(content=user_message_copy.get("content", ""))
            self.memory.add_message(user_msg)
            # Define DurableAgentMessage object for state persistence
            msg_object = DurableAgentMessage(**user_message_copy)
            # Ensure the instance entry exists
            if instance_id not in self.state["instances"]:
                self.state["instances"][instance_id] = {
                    "messages": [],
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "source": "user_input",
                    "workflow_instance_id": instance_id,
                    "triggering_workflow_instance_id": None,
                    "workflow_name": self._workflow_name,
                }
            inst: dict = self.state["instances"][instance_id]
            inst.setdefault("messages", []).append(msg_object.model_dump(mode="json"))
            inst["last_message"] = msg_object.model_dump(mode="json")
            # Note: Do not use global chat_history update to prevent cross-instance contamination
            # Save the state after appending the user message
            self.save_state()

        # Always print the last user message for context, even if no input_data is provided
        if user_message_copy is not None:
            # Ensure keys are str for mypy
            self.text_formatter.print_message(
                {str(k): v for k, v in user_message_copy.items()}
            )

        # Generate response using the LLM
        try:
            response: LLMChatResponse = self.llm.generate(
                messages=messages,
                tools=self.get_llm_tools(),
                **(
                    {"tool_choice": self.tool_choice}
                    if self.tool_choice is not None
                    else {}
                ),
            )
            # Get the first candidate from the response
            response_message = response.get_message()
            # Check if the response contains an assistant message
            if response_message is None:
                raise AgentError("LLM returned no assistant message")
            # Convert the response message to a dict to work with JSON serialization
            assistant_message = response_message.model_dump()
            return assistant_message
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            logger.error(
                f"LLM generation failed in workflow {instance_id}: {error_type} - {error_msg}"
            )
            logger.error(f"Task: {task}")
            logger.error(f"Messages count: {len(messages)}")
            logger.error(f"Tools available: {len(self.get_llm_tools())}")
            logger.error("Full error details:", exc_info=True)

            raise AgentError(
                f"LLM generation failed in workflow {instance_id}: {error_type} - {error_msg}"
            ) from e

    @task
    async def run_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a tool call by invoking the specified function with the provided arguments.

        Args:
            tool_call (Dict[str, Any]): A dictionary containing tool execution details, including the function name and arguments.

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
            logger.error(f"Error executing tool '{fn_name}': {e}", exc_info=True)
            raise AgentError(f"Error executing tool '{fn_name}': {e}") from e

        # Return the plain payload for later persistence
        return {
            "tool_call_id": tool_call["id"],
            "tool_name": fn_name,
            "tool_args": args,
            "execution_result": str(result) if result is not None else "",
        }

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
        # Format Response
        response["role"] = "user"
        response["name"] = self.name
        response["workflow_instance_id"] = target_instance_id
        agent_response = AgentTaskResponse(**response)

        # Send the message to the target agent
        await self.send_message_to_agent(name=target_agent, message=agent_response)

    @task
    def append_assistant_message(
        self, instance_id: str, message: Dict[str, Any]
    ) -> None:
        """
        Append an assistant message into the workflow state.

        Args:
            instance_id (str): The workflow instance ID.
            message (Dict[str, Any]): The assistant message to append.
        """
        message["name"] = self.name
        # Convert the message to a DurableAgentMessage object
        msg_object = DurableAgentMessage(**message)
        # Ensure the instance entry exists
        if instance_id not in self.state["instances"]:
            self.state["instances"][instance_id] = {
                "messages": [],
                "start_time": datetime.now(timezone.utc).isoformat(),
                "source": "user_input",
                "workflow_instance_id": instance_id,
                "triggering_workflow_instance_id": None,
                "workflow_name": self._workflow_name,
            }
        inst: dict = self.state["instances"][instance_id]
        # Check if message already exists (idempotent operation for workflow replay)
        message_dict = msg_object.model_dump(mode="json")
        messages = inst.setdefault("messages", [])

        # Check for duplicate by message ID (idempotent for workflow replay)
        message_exists = any(
            msg.get("id") == message_dict.get("id") for msg in messages
        )
        if not message_exists:
            messages.append(message_dict)
            inst["last_message"] = message_dict
            # Add the assistant message to memory (only if new)
            # Note: Memory updates are handled at workflow level to avoid replay issues
            self.memory.add_message(AssistantMessage(**message))
            # Save the state after appending the assistant message
            self.save_state()
        # Print the assistant message
        self.text_formatter.print_message(message)

    @task
    def append_tool_message(
        self, instance_id: str, tool_result: Dict[str, Any]
    ) -> None:
        """
        Append a tool-execution record to both the per-instance history and the agent's tool_history.
        """
        # Define a ToolMessage object from the tool result
        tool_message = ToolMessage(
            tool_call_id=tool_result["tool_call_id"],
            name=tool_result["tool_name"],
            content=tool_result["execution_result"],
        )
        # Define DurableAgentMessage object for state persistence
        msg_object = DurableAgentMessage(**tool_message.model_dump())
        # Define a ToolExecutionRecord object
        # to store the tool execution details in the workflow state
        tool_history_entry = ToolExecutionRecord(**tool_result)
        # Ensure the instance entry exists
        if instance_id not in self.state["instances"]:
            self.state["instances"][instance_id] = {
                "messages": [],
                "start_time": datetime.now(timezone.utc).isoformat(),
                "source": "user_input",
                "workflow_instance_id": instance_id,
                "triggering_workflow_instance_id": None,
                "workflow_name": self._workflow_name,
            }
        inst: dict = self.state["instances"][instance_id]
        # Check if message already exists (idempotent operation for workflow replay)
        message_dict = msg_object.model_dump(mode="json")
        messages = inst.setdefault("messages", [])

        # Check for duplicate by message ID (idempotent for workflow replay)
        message_exists = any(
            msg.get("id") == message_dict.get("id") for msg in messages
        )
        if not message_exists:
            messages.append(message_dict)

        # Check for duplicate tool history entry by tool_call_id
        tool_history_dict = tool_history_entry.model_dump(mode="json")
        tool_history = inst.setdefault("tool_history", [])

        tool_exists = any(
            th.get("tool_call_id") == tool_history_dict.get("tool_call_id")
            for th in tool_history
        )
        if not tool_exists:
            tool_history.append(tool_history_dict)
            # Update tool history and memory of agent (only if new)
            # Note: Memory updates are handled at workflow level to avoid replay issues
            self.tool_history.append(tool_history_entry)
            # Add the tool message to the agent's memory
            self.memory.add_message(tool_message)
        # Save the state after appending the tool message
        self.save_state()
        # Print the tool message
        self.text_formatter.print_message(tool_message)

    @task
    def finalize_workflow(self, instance_id: str, final_output: str) -> None:
        """
        Record the final output and end_time in the workflow state.
        """
        end_time = datetime.now(timezone.utc)
        end_time_str = end_time.isoformat()
        # Ensure the instance entry exists
        if instance_id not in self.state["instances"]:
            self.state["instances"][instance_id] = {
                "messages": [],
                "start_time": datetime.now(timezone.utc).isoformat(),
                "source": "user_input",
                "workflow_instance_id": instance_id,
                "triggering_workflow_instance_id": None,
                "workflow_name": self._workflow_name,
            }
        inst: dict = self.state["instances"][instance_id]
        inst["output"] = final_output
        inst["end_time"] = end_time_str
        inst["status"] = DaprWorkflowStatus.COMPLETED.value  # Mark as completed
        logger.info(f"Workflow {instance_id} completed successfully")
        self.save_state()

    @message_router(broadcast=True)
    async def process_broadcast_message(self, message: BroadcastMessage):
        """
        Processes a broadcast message, filtering out messages sent by the same agent
        and updating local memory with valid messages.

        Args:
            message (BroadcastMessage): The received broadcast message.

        Returns:
            None: The function updates the agent's memory and ignores unwanted messages.
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
                logger.info(
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
            msg_object = DurableAgentMessage(**message.model_dump())

            # Persist to global chat history
            self.state.setdefault("chat_history", [])
            self.state["chat_history"].append(msg_object.model_dump(mode="json"))
            # Save the state after processing the broadcast message
            self.save_state()

        except Exception as e:
            logger.error(f"Error processing broadcast message: {e}", exc_info=True)

    def _construct_messages_with_instance_history(
        self, instance_id: str, input_data: Union[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Construct messages using instance-specific chat history instead of global memory.
        This ensures proper message sequence for tool calls and prevents OpenAI API errors
        in the event an app gets terminated or restarts while the workflow is running.

        Args:
            instance_id: The workflow instance ID
            input_data: User input, either as a string or dictionary

        Returns:
            List of formatted messages with proper sequence
        """
        if not self.prompt_template:
            raise ValueError(
                "Prompt template must be initialized before constructing messages."
            )

        # Get instance-specific chat history instead of global memory
        instance_data = self.state.get("instances", {}).get(instance_id, {})
        instance_messages = instance_data.get("messages", [])

        # Convert instance messages to the format expected by prompt template
        chat_history = []
        for msg in instance_messages:
            if isinstance(msg, dict):
                chat_history.append(msg)
            else:
                # Convert DurableAgentMessage to dict if needed
                chat_history.append(
                    msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
                )

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
