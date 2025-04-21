from abc import abstractmethod
from typing import List, Optional, Union
from dapr.actor import ActorInterface, actormethod
from dapr_agents.types.agent import AgentActorMessage, AgentStatus

class AgentActorInterface(ActorInterface):
    @abstractmethod
    @actormethod(name='InvokeTask')
    async def invoke_task(self, task: Optional[str] = None) -> str:
        """
        Invoke a task and returns the result as a string.
        """
        pass

    @abstractmethod
    @actormethod(name='AddMessage')
    async def add_message(self, message: Union[AgentActorMessage, dict]) -> None:
        """
        Adds a message to the conversation history in the actor's state.
        """
        pass

    @abstractmethod
    @actormethod(name='GetMessages')
    async def get_messages(self) -> List[dict]:
        """
        Retrieves the conversation history from the actor's state.
        """
        pass

    @abstractmethod
    @actormethod(name='SetStatus')
    async def set_status(self, status: AgentStatus) -> None:
        """
        Sets the current operational status of the agent.
        """
        pass