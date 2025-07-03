import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dapr_agents.agents.base import AgentBase
from dapr_agents.workflow.agentic import AgenticWorkflow
from pydantic import Field, model_validator


from dapr.ext.workflow import DaprWorkflowContext  # type: ignore

from dapr_agents.types import (
    AgentError,
    ToolMessage,
)
from .schemas import (
    AgentTaskResponse,
    BroadcastMessage,
    TriggerAction,
)
from .state import (
    AssistantWorkflowEntry,
    AssistantWorkflowMessage,
    AssistantWorkflowState,
    AssistantWorkflowToolMessage,
)
from dapr_agents.workflow.decorators import task, workflow
from dapr_agents.workflow.messaging.decorator import message_router

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

    tool_history: List[ToolMessage] = Field(
        default_factory=list, description="Executed tool calls during the conversation."
    )
    tool_choice: Optional[str] = Field(
        default=None,
        description="Strategy for selecting tools ('auto', 'required', 'none'). Defaults to 'auto' if tools are provided.",
    )
    agent_topic_name: Optional[str] = Field(
        None,
        description="The topic name dedicated to this specific agent, derived from the agent's name if not provided.",
    )

    _agent_metadata: Optional[Dict[str, Any]] = None

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
        self.state = AssistantWorkflowState()

        # Call AgenticWorkflow's model_post_init first to initialize state store and other dependencies
        super().model_post_init(__context)

        # Name of main Workflow
        # TODO: can this be configurable or dynamic? Would that make sense?
        self._workflow_name = "ToolCallingWorkflow"
        self.tool_choice = self.tool_choice or ("auto" if self.tools else None)

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

    async def run(self, input_data: Optional[Union[str, Dict[str, Any]]] = None) -> Any:
        """
        Run the durable agent with the given input.
        TODO: For DurableAgent, this method should trigger the workflow execution maybe..?

        Args:
            input_data: The input data for the agent to process.

        Returns:
            The result of the workflow execution.
        """
        # TODO: For DurableAgent, the run method should trigger the workflow
        logger.info(
            f"DurableAgent {self.name} run method called with input: {input_data}"
        )

        # Return a message indicating this is a durable agent and agent start via run for durable agent is yet to be determined.
        return f"DurableAgent {self.name} is designed to run as a workflow service asynchronously. Use .as_service() and/or .start() instead for now. The workflow endpoints can also be usedto interact with this agent."

    @message_router
    @workflow(name="ToolCallingWorkflow")
    def tool_calling_workflow(self, ctx: DaprWorkflowContext, message: TriggerAction):
        """
        Executes a tool-calling workflow, determining the task source (either an agent or an external user).
        This uses Dapr Workflows to run the agent in a ReAct-style loop until it generates a final answer or reaches max iterations,
        calling tools as needed.
        """
        # Step 0: Retrieve task and iteration input
        # Handle both TriggerAction objects and dictionaries
        if isinstance(message, dict):
            task = message.get("task")
            iteration = message.get("iteration", 0)
            workflow_instance_id = message.get("workflow_instance_id")
        else:
            task = message.task
            iteration = message.iteration or 0
            workflow_instance_id = message.workflow_instance_id

        instance_id = ctx.instance_id

        if not ctx.is_replaying:
            logger.info(
                f"Workflow iteration {iteration + 1} started (Instance ID: {instance_id})."
            )

        # Step 1: Initialize instance entry on first iteration
        if iteration == 0:
            # Handle metadata extraction for both TriggerAction objects and dictionaries
            if isinstance(message, dict):
                metadata = message.get("_message_metadata", {})
            else:
                metadata = getattr(message, "_message_metadata", {})

            # Ensure "instances" key exists
            if isinstance(self.state, dict) and "instances" not in self.state:
                self.state["instances"] = {}

            # Extract workflow metadata with proper defaults
            source = metadata.get("source") if isinstance(metadata, dict) else None
            source_workflow_instance_id = workflow_instance_id

            # Create a new workflow entry
            workflow_entry = AssistantWorkflowEntry(
                input=task or "Triggered without input.",
                source=source,
                source_workflow_instance_id=source_workflow_instance_id,
                output="",  # Required
                end_time=None,  # Required
            )

            # Store in state, converting to JSON only if necessary
            if isinstance(self.state, dict):
                self.state["instances"][instance_id] = workflow_entry.model_dump(
                    mode="json"
                )

            if not ctx.is_replaying:
                logger.info(f"Initial message from {source} -> {self.name}")

        # Step 2: Retrieve workflow entry for this instance
        if isinstance(self.state, dict):
            workflow_entry = self.state["instances"].get(instance_id, {})
            # Handle dictionary format
            if isinstance(workflow_entry, dict):
                source = workflow_entry.get("source")
                source_workflow_instance_id = workflow_entry.get(
                    "source_workflow_instance_id"
                )
            else:
                # Handle object format
                source = workflow_entry.source
                source_workflow_instance_id = workflow_entry.source_workflow_instance_id
        else:
            source = None
            source_workflow_instance_id = None

        # Step 3: Generate Response
        response = yield ctx.call_activity(
            self.generate_response, input={"instance_id": instance_id, "task": task}
        )
        response_message = yield ctx.call_activity(
            self.get_response_message, input={"response": response}
        )

        # Step 4: Extract Finish Reason
        finish_reason = yield ctx.call_activity(
            self.get_finish_reason, input={"response": response}
        )

        # Step 5: Choose execution path based on LLM response
        if finish_reason == "tool_calls":
            if not ctx.is_replaying:
                logger.info(
                    "Tool calls detected in LLM response, extracting and preparing for execution.."
                )

            # Retrieve the list of tool calls extracted from the LLM response
            tool_calls = yield ctx.call_activity(
                self.get_tool_calls, input={"response": response}
            )

            # Execute tool calls in parallel
            if not ctx.is_replaying:
                logger.info(f"Executing {len(tool_calls)} tool call(s)..")

            parallel_tasks = [
                ctx.call_activity(
                    self.execute_tool,
                    input={"instance_id": instance_id, "tool_call": tool_call},
                )
                for tool_call in tool_calls
            ]
            yield self.when_all(parallel_tasks)
        else:
            if not ctx.is_replaying:
                logger.info("Agent generating response without tool execution..")

            # No Tool Calls → Clear tools
            self.tool_history.clear()

        # Step 6: Determine if Workflow Should Continue
        next_iteration_count = iteration + 1
        max_iterations_reached = next_iteration_count > self.max_iterations

        if finish_reason == "stop" or max_iterations_reached:
            # Determine the reason for stopping
            if max_iterations_reached:
                verdict = "max_iterations_reached"
                if not ctx.is_replaying:
                    logger.warning(
                        f"Workflow {instance_id} reached the max iteration limit ({self.max_iterations}) before finishing naturally."
                    )

                # Modify the response message to indicate forced stop
                response_message[
                    "content"
                ] += "\n\nThe workflow was terminated because it reached the maximum iteration limit. The task may not be fully complete."

            else:
                # TODO: make this one word how we have max_iterations_reached for ex.
                verdict = "model hit a natural stop point."

            # Step 8: Broadcasting Response to all agents if available
            yield ctx.call_activity(
                self.broadcast_message_to_agents, input={"message": response_message}
            )

            # Step 9: Respond to source agent if available
            if source and source_workflow_instance_id:
                yield ctx.call_activity(
                    self.send_response_back,
                    input={
                        "response": response_message,
                        "target_agent": source,
                        "target_instance_id": source_workflow_instance_id,
                    },
                )

            # Step 10: Share Final Message
            yield ctx.call_activity(
                self.finish_workflow,
                input={"instance_id": instance_id, "message": response_message},
            )

            if not ctx.is_replaying:
                logger.info(
                    f"Workflow {instance_id} has been finalized with verdict: {verdict}"
                )

            return response_message

        # Step 7: Continue Workflow Execution
        if isinstance(message, dict):
            message.update({"task": None, "iteration": next_iteration_count})

        ctx.continue_as_new(message)

    @task
    async def generate_response(
        self, instance_id: str, task: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generates a response using the LLM based on the current conversation context.

        Args:
            instance_id (str): The unique identifier of the workflow instance.
            task (Optional[Union[str, Dict[str, Any]]]): The task or query provided by the user.

        Returns:
            Dict[str, Any]: The LLM response as a dictionary.
        """
        # Construct prompt messages
        messages = self.construct_messages(task or {})

        # Store message in workflow state and local memory
        if task:
            task_message = {"role": "user", "content": task}
            await self.update_workflow_state(
                instance_id=instance_id, message=task_message
            )

        # Convert ToolMessage objects to dictionaries for LLM compatibility
        tool_messages = []
        for tool_msg in self.tool_history:
            if isinstance(tool_msg, ToolMessage):
                tool_messages.append(
                    {
                        "role": tool_msg.role,
                        "content": tool_msg.content,
                        "tool_call_id": tool_msg.tool_call_id,
                    }
                )
            else:
                # Handle case where tool_msg is already a dict
                tool_messages.append(tool_msg)

        messages.extend(tool_messages)

        try:
            response = self.llm.generate(
                messages=messages,
                tools=self.get_llm_tools(),
                tool_choice=self.tool_choice,
            )

            # Convert ChatCompletion object to dictionary for workflow serialization
            if hasattr(response, "model_dump"):
                return response.model_dump()
            elif isinstance(response, dict):
                return response
            else:
                # Fallback: convert to string and wrap in dict
                return {"content": str(response)}

        except Exception as e:
            raise AgentError(f"Failed during chat generation: {e}") from e

    @task
    def get_response_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts the response message from the first choice in the LLM response.

        Args:
            response (Dict[str, Any]): The response dictionary from the LLM, expected to contain a "choices" key.

        Returns:
            Dict[str, Any]: The extracted response message with the agent's name added.
        """
        choices = response.get("choices", [])
        response_message = choices[0].get("message", {})

        return response_message

    @task
    def get_finish_reason(self, response: Dict[str, Any]) -> str:
        """
        Extracts the finish reason from the LLM response, indicating why generation stopped.

        Args:
            response (Dict[str, Any]): The response dictionary from the LLM, expected to contain a "choices" key.

        Returns:
            str: The reason the model stopped generating tokens. Possible values include:
                - "stop": Natural stop point or stop sequence encountered.
                - "length": Maximum token limit reached.
                - "content_filter": Content flagged by filters.
                - "tool_calls": The model called a tool.
                - "function_call" (deprecated): The model called a function.
                - None: If no valid choice exists in the response.
        """
        try:
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices and len(choices) > 0:
                    return choices[0].get("finish_reason", "unknown")
            return "unknown"
        except Exception as e:
            logger.error(f"Error extracting finish reason: {e}")
            return "unknown"

    @task
    def get_tool_calls(
        self, response: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extracts tool calls from the first choice in the LLM response, if available.

        Args:
            response (Dict[str, Any]): The response dictionary from the LLM, expected to contain "choices"
                                    and potentially tool call information.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of tool calls if present, otherwise None.
        """
        choices = response.get("choices", [])

        if not choices:
            logger.warning("No choices found in LLM response.")
            return None

        # Save Tool Call Response Message
        response_message = choices[0].get("message", {})
        self.tool_history.append(response_message)

        # Extract tool calls safely
        tool_calls = choices[0].get("message", {}).get("tool_calls")

        if not tool_calls:
            logger.info("No tool calls found in LLM response.")
            return None

        return tool_calls

    @task
    async def execute_tool(self, instance_id: str, tool_call: Dict[str, Any]):
        """
        Executes a tool call by invoking the specified function with the provided arguments.

        Args:
            instance_id (str): The unique identifier of the workflow instance.
            tool_call (Dict[str, Any]): A dictionary containing tool execution details, including the function name and arguments.

        Raises:
            AgentError: If the tool call is malformed or execution fails.
        """
        function_details = tool_call.get("function", {})
        function_name = function_details.get("name")

        if not function_name:
            raise AgentError("Missing function name in tool execution request.")

        try:
            function_args = function_details.get("arguments", "")
            logger.info(
                f"Executing tool '{function_name}' with raw arguments: {function_args}"
            )

            function_args_as_dict = json.loads(function_args) if function_args else {}
            logger.info(
                f"Parsed arguments for '{function_name}': {function_args_as_dict}"
            )

            # Execute tool function
            result = await self.tool_executor.run_tool(
                function_name, **function_args_as_dict
            )

            logger.info(
                f"Tool '{function_name}' executed successfully with result: {result}"
            )

            # Construct tool execution message payload
            workflow_tool_message = {
                "tool_call_id": tool_call.get("id"),
                "function_name": function_name,
                "function_args": function_args,
                "content": str(result),
            }

            # Update workflow state and agent tool history
            await self.update_workflow_state(
                instance_id=instance_id, tool_message=workflow_tool_message
            )

        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON in tool arguments for function '{function_name}': {function_args}"
            )
            raise AgentError(
                f"Invalid JSON format in arguments for tool '{function_name}': {e}"
            )

        except Exception as e:
            logger.error(f"Error executing tool '{function_name}': {e}", exc_info=True)
            raise AgentError(f"Error executing tool '{function_name}': {e}") from e

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
    async def finish_workflow(self, instance_id: str, message: Dict[str, Any]):
        """
        Finalizes the workflow by storing the provided message as the final output.

        Args:
            instance_id (str): The unique identifier of the workflow instance.
            summary (Dict[str, Any]): The final summary to be stored in the workflow state.
        """
        # Store message in workflow state
        await self.update_workflow_state(instance_id=instance_id, message=message)

        # Store final output
        await self.update_workflow_state(
            instance_id=instance_id, final_output=message["content"]
        )

    async def update_workflow_state(
        self,
        instance_id: str,
        message: Optional[Dict[str, Any]] = None,
        tool_message: Optional[Dict[str, Any]] = None,
        final_output: Optional[str] = None,
    ):
        """
        Updates the workflow state by appending a new message or setting the final output.
        Accepts both dict and AssistantWorkflowState as valid state types.
        """
        # Accept both dict and AssistantWorkflowState
        if isinstance(self.state, dict):
            if "instances" not in self.state:
                self.state["instances"] = {}
            workflow_entry = self.state["instances"].get(instance_id)
            if not workflow_entry:
                raise ValueError(
                    f"No workflow entry found for instance_id {instance_id} in local state."
                )
        elif isinstance(self.state, AssistantWorkflowState):
            if instance_id not in self.state.instances:
                raise ValueError(
                    f"No workflow entry found for instance_id {instance_id} in AssistantWorkflowState."
                )
            workflow_entry = self.state.instances[instance_id]
        else:
            raise ValueError(f"Invalid state type: {type(self.state)}")

        # Store user/assistant messages separately
        if message is not None:
            serialized_message = AssistantWorkflowMessage(**message).model_dump(
                mode="json"
            )
            if isinstance(workflow_entry, dict):
                workflow_entry.setdefault("messages", []).append(serialized_message)
                workflow_entry["last_message"] = serialized_message
            else:
                workflow_entry.messages.append(AssistantWorkflowMessage(**message))
                workflow_entry.last_message = AssistantWorkflowMessage(**message)

            # Add to memory only if it's a user/assistant message
            from dapr_agents.types.message import UserMessage

            if message.get("role") == "user":
                user_msg = UserMessage(content=message.get("content", ""))
                self.memory.add_message(user_msg)

        # Store tool execution messages separately in tool_history
        if tool_message is not None:
            serialized_tool_message = AssistantWorkflowToolMessage(
                **tool_message
            ).model_dump(mode="json")
            if isinstance(workflow_entry, dict):
                workflow_entry.setdefault("tool_history", []).append(
                    serialized_tool_message
                )
            else:
                workflow_entry.tool_history.append(
                    AssistantWorkflowToolMessage(**tool_message)
                )

            # Also update agent-level tool history (execution tracking)
            agent_tool_message = ToolMessage(
                tool_call_id=tool_message["tool_call_id"],
                name=tool_message["function_name"],
                content=tool_message["content"],
            )
            self.tool_history.append(agent_tool_message)

        # Store final output
        if final_output is not None:
            if isinstance(workflow_entry, dict):
                workflow_entry["output"] = final_output
                workflow_entry["end_time"] = datetime.now().isoformat()
            else:
                workflow_entry.output = final_output
                workflow_entry.end_time = datetime.now()

        # Persist updated state
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
            # Extract metadata safely from message attributes
            metadata = getattr(message, "_message_metadata", {})

            if not isinstance(metadata, dict):
                logger.warning(
                    f"{self.name} received a broadcast message with invalid metadata format. Ignoring."
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

        except Exception as e:
            logger.error(f"Error processing broadcast message: {e}", exc_info=True)

    @property
    def agent_metadata(self) -> Optional[Dict[str, Any]]:
        """Get the agent metadata."""
        return self._agent_metadata
