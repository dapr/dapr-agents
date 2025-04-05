import logging
from typing import List, Optional, Union
from dapr.actor import Actor
from dapr.actor.id import ActorId
from dapr.actor.runtime.context import ActorRuntimeContext
from dapr_agents.agent.actor.interface import AgentActorInterface
from dapr_agents.agent.base import AgentBase
from dapr_agents.types.agent import (
    AgentActorMessage,
    AgentActorState,
    AgentStatus,
    AgentTaskEntry,
    AgentTaskStatus,
)
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class AgentActorBase(Actor, AgentActorInterface):
    """Base class for all agent actors, including task execution and agent state management."""

    def __init__(self, ctx: ActorRuntimeContext, actor_id: ActorId):
        super().__init__(ctx, actor_id)
        self.actor_id = actor_id
        self.agent: AgentBase
        self.agent_state_key = "agent_state"
    
    async def _on_activate(self) -> None:
        """
        Called when the actor is activated. Initializes the agent's state if not present.
        """
        logger.info(f"Activating actor with ID: {self.actor_id}")
        has_state, state_data = await self._state_manager.try_get_state(self.agent_state_key)

        if not has_state:
            # Initialize state with default values if it doesn't exist
            logger.info(f"Initializing state for {self.actor_id}")
            self.state = AgentActorState(overall_status=AgentStatus.IDLE)
            await self._state_manager.set_state(self.agent_state_key, self.state.model_dump())
            await self._state_manager.save_state()
        else:
            # Load existing state
            logger.info(f"Loading existing state for {self.actor_id}")
            logger.debug(f"Existing state for {self.actor_id}: {state_data}")
            self.state = AgentActorState(**state_data)

    async def _on_deactivate(self) -> None:
        """
        Called when the actor is deactivated.
        """
        logger.info(f"Deactivate {self.__class__.__name__} actor with ID: {self.actor_id}.")
    
    async def set_status(self, status: AgentStatus) -> None:
        """
        Sets the current operational status of the agent and saves the state.
        """
        self.state.overall_status = status
        await self._state_manager.set_state(self.agent_state_key, self.state.model_dump())
        await self._state_manager.save_state()
    
    async def invoke_task(self, task: Optional[str] = None) -> str:
        """
        Execute the agent's main task, log the input/output in the task history,
        and update state with observations, plans, and feedback.

        If no task is provided, use the most recent message content as the task entry input,
        but still execute `run()` directly if no task is passed.
        """
        logger.info(f"Actor {self.actor_id} invoking a task")

        # Determine the input for the task entry
        messages = await self.get_messages()  # Fetch messages from state
        default_task = None

        if messages:
            # Look for the last message in the conversation history
            last_message = messages[-1]
            default_task = last_message.get("content")
            logger.debug(f"Default task entry input derived from last message: {default_task}")

        # Prepare the input for task entry
        task_entry_input = task or default_task or "Triggered without a specific task"
        logger.debug(f"Task entry input: {task_entry_input}")

        # Set the agent's status to active
        await self.set_status(AgentStatus.ACTIVE)

        # Create a new task entry with the determined input
        task_entry = AgentTaskEntry(
            input=task_entry_input,
            status=AgentTaskStatus.IN_PROGRESS,
        )
        self.state.task_history.append(task_entry)

        # Save initial task state with IN_PROGRESS status
        await self._state_manager.set_state(self.agent_state_key, self.state.model_dump())
        await self._state_manager.save_state()

        try:
            # Run the task if provided, or fallback to agent.run() if no task
            task_input = task or None
            result = await self.agent.run(task_input)

            # Update the task entry with the result and mark as COMPLETE
            task_entry.output = result
            task_entry.status = AgentTaskStatus.COMPLETE

            # Add the result as a new message in conversation history
            assistant_message = AgentActorMessage(role="assistant", content=result)
            await self.add_message(assistant_message)

            return result

        except Exception as e:
            # Handle task failure
            logger.error(f"Error running task for actor {self.actor_id}: {str(e)}")
            task_entry.status = AgentTaskStatus.FAILED
            task_entry.output = str(e)
            raise e

        finally:
            # Ensure the final state of the task is saved
            await self._state_manager.set_state(self.agent_state_key, self.state.model_dump())
            await self._state_manager.save_state()
            # Revert the agent's status to idle
            await self.set_status(AgentStatus.IDLE)
    
    async def add_message(self, message: Union[AgentActorMessage, dict]) -> None:
        """
        Adds a message to the conversation history in the actor's state.

        Args:
            message (Union[AgentActorMessage, dict]): The message to add, either as a dictionary or an AgentActorMessage instance.
        """
        # Convert dictionary to AgentActorMessage if necessary
        if isinstance(message, dict):
            message = AgentActorMessage(**message)
        
        # Add the new message to the state
        self.state.messages.append(message)
        self.state.message_count += 1

        # Save state back to Dapr
        await self._state_manager.set_state(self.agent_state_key, self.state.model_dump())
        await self._state_manager.save_state()

    async def get_messages(self) -> List[dict]:
        """
        Retrieves the messages from the actor's state, validates it using Pydantic, 
        and returns a list of dictionaries if valid.
        """
        has_state, state_data = await self._state_manager.try_get_state(self.agent_state_key)

        if has_state:
            try:
                # Validate the state data using Pydantic
                state: AgentActorState = AgentActorState.model_validate(state_data)

                # Return the list of messages as dictionaries (timestamp will be automatically serialized to ISO format)
                return [message.model_dump() for message in state.messages]
            except ValidationError as e:
                # Handle validation errors
                print(f"Validation error: {e}")
                return []
        return []