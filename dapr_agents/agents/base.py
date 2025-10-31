from dapr_agents.agents.memory_store import MemoryStore
from dapr_agents.agents.utils.text_printer import ColorTextFormatter
from dapr_agents.types import BaseMessage, ToolExecutionRecord
from dapr_agents.tool.executor import AgentToolExecutor
from dapr_agents.prompt.agent_prompt import Prompt
from dapr_agents.prompt.agent_prompt_context import Context
from dapr_agents.tool.base import AgentTool
import re
import json
from datetime import datetime
import logging
import asyncio
import signal
from abc import ABC, abstractmethod
from typing import (
    List,
    Optional,
    Dict,
    Any,
    Union,
    Callable,
)
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ConfigDict
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.utils.defaults import get_default_llm

logger = logging.getLogger(__name__)


class AgentBase(BaseModel, ABC):
    """
    Base class for agents that interact with language models and manage tools for task execution.

     Args:
        name: Agent name
        role: Agent role
        goal: Agent goal
        instructions: List of instructions
        tools: List of tools
        llm: LLM client
        memory: Memory instance
    """

    name: str = Field(
        default="Dapr Agent",
        description="The agent's name, defaulting to the role if not provided.",
    )
    role: Optional[str] = Field(
        default="Assistant",
        description="The agent's role in the interaction (e.g., 'Weather Expert').",
    )
    goal: Optional[str] = Field(
        default="Help humans",
        description="The agent's main objective (e.g., 'Provide Weather information').",
    )
    # TODO: add a background/backstory field that would be useful for the agent to know about it's context/background for it's role.
    instructions: Optional[List[str]] = Field(
        default=None, description="Instructions guiding the agent's tasks."
    )
    llm: Optional[ChatClientBase] = Field(
        default=None,
        description="Language model client for generating responses.",
    )
    # TODO: we need to add RBAC to tools to define what users and/or agents can use what tool(s).
    tools: List[Union[AgentTool, Callable]] = Field(
        default_factory=list,
        description="Tools available for the agent to assist with tasks.",
    )
    tool_choice: Optional[str] = Field(
        default=None,
        description="Strategy for selecting tools ('auto', 'required', 'none'). Defaults to 'auto' if tools are provided.",
    )
    tool_history: List[ToolExecutionRecord] = Field(
        default_factory=list, description="Executed tool calls during the conversation."
    )
    # TODO: add a forceFinalAnswer field in case maxIterations is near/reached. Or do we have a conclusion baked in by default? Do we want this to derive a conclusion by default?
    max_iterations: int = Field(
        default=10, description="Max iterations for conversation cycles."
    )
    memory_store: Optional["MemoryStore"] = Field(
        default=None,
        description=(
            "Storage for conversation history. "
            "If None, a default in-memory Storage will be created. "
            "For persistent storage, specify the name of the Dapr State Store to use. "
        ),
    )
    prompt: Optional[Prompt] = Field(
        default_factory=Prompt,
        description="Prompt handles how agent prompts (system messages and full chat context) are built and formatted before being sent to the LLM.",
    )

    _tool_executor: AgentToolExecutor = PrivateAttr()
    _text_formatter: ColorTextFormatter = PrivateAttr(
        default_factory=ColorTextFormatter
    )
    _dapr_client: Optional[Any] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    def set_name_from_role(cls, values: dict):
        # Set name to role if name is not provided
        if not values.get("name") and values.get("role"):
            values["name"] = values["role"]
        return values

    @model_validator(mode="after")
    def validate_llm(self):
        """Validate that LLM is properly configured."""
        if hasattr(self, "llm"):
            if self.llm is None:
                logger.warning("LLM client is None, some functionality may be limited.")
            else:
                try:
                    # Validate LLM is properly configured by accessing it as this is required to be set.
                    _ = self.llm
                except Exception as e:
                    logger.error(f"Failed to initialize LLM: {e}")
                    self.llm = None

        return self

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook for AgentBase.
        Sets up the prompt template using a centralized helper, ensuring agent and LLM client reference the same template.
        Also validates and pre-fills the template, and sets up graceful shutdown.

        Args:
            __context (Any): Context passed from Pydantic's model initialization.
        """
        self._tool_executor = AgentToolExecutor(tools=self.tools)

        # Set tool_choice to 'auto' if tools are provided, otherwise None
        if self.tool_choice is None:
            self.tool_choice = "auto" if self.tools else None

        # Initialize LLM if not provided
        if self.llm is None:
            self.llm = get_default_llm()
        elif getattr(self.llm, "prompt_template", None):
            # Agent owns the prompt; warn if LLM has its own template set
            logger.warning(
                "LLM prompt template is set, but agent will use its own prompt template."
            )

        # Initialize storage if not provided (in-memory by default)
        if self.memory_store is None:
            self.memory_store = MemoryStore()
            logger.debug("Initialized default in-memory Storage")

        # Initialize Dapr client if storage is persistent
        # This is needed for state store access and agent registration
        if self.memory_store and self.memory_store.name and self._dapr_client is None:
            from dapr.clients import DaprClient

            self._dapr_client = DaprClient()
            logger.debug(
                f"Initialized Dapr client for agent '{self.name}' with persistent storage"
            )

        # Register agent if it has persistent storage
        # This applies to both Agent and DurableAgent with persistent storage
        if self.memory_store and self.memory_store.name:
            if self.registry_store is None:
                self.registry_store = self.memory_store.name
            agent_metadata = {
                "name": self.name,
                "role": self.role,
                "goal": self.goal,
                "tool_choice": self.tool_choice,
                "instructions": self.instructions,
                "topic_name": getattr(self, "agent_topic_name", None),
                "pubsub_name": getattr(self, "message_bus_name", None),
                "orchestrator": False,
                "statestore_name": self.memory_store.name,
                "registry_name": self.registry_store,
            }

            self.register_agent(
                store_name=self.registry_store,
                store_key="agent_registry",
                agent_name=self.name,
                agent_metadata=self._serialize_metadata(agent_metadata),
            )

        if self.prompt is None:
            self.prompt = Prompt()
        if getattr(self.prompt, "context", None) is None:
            self.prompt.context = Context()
        self.prompt.context.name = self.name
        self.prompt.context.role = self.role
        self.prompt.context.goal = self.goal
        self.prompt.context.instructions = self.instructions

        # Set up graceful shutdown
        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()

        super().model_post_init(__context)

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (OSError, ValueError):
            # TODO: test this bc signal handlers may not work in all environments (e.g., Windows)
            pass

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self._shutdown_event.set()

    @property
    def tool_executor(self) -> AgentToolExecutor:
        """Returns the client to execute and manage tools, ensuring it's accessible but read-only."""
        return self._tool_executor

    @property
    def text_formatter(self) -> ColorTextFormatter:
        """Returns the text formatter for the agent."""
        return self._text_formatter

    @abstractmethod
    def get_chat_history(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the chat history as a list of dictionaries.

        Args:
            task (Optional[str]): The task or query provided by the user.

        Returns:
            List[Dict[str, Any]]: The chat history as dictionaries.
        """

    @property
    def chat_history(self) -> List[Dict[str, Any]]:
        """
        Returns the full chat history as a list of dictionaries.

        Returns:
            List[Dict[str, Any]]: The chat history.
        """
        return self.get_chat_history()

    @abstractmethod
    def run(self, input_data: Union[str, Dict[str, Any]]) -> Any:
        """
        Executes the agent's main logic based on provided inputs.

        Args:
            inputs (Dict[str, Any]): A dictionary with dynamic input values for task execution.
        """
        pass

    def construct_messages(
        self, input_data: Union[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Constructs and formats initial messages based on input type, passing chat_history as a list, without mutating the prompt template.

        Args:
            input_data (Union[str, Dict[str, Any]]): User input, either as a string or dictionary.

        Returns:
            List[Dict[str, Any]]: List of formatted messages, including the user message if input_data is a string.
        """
        has_prompt = bool(self.prompt and getattr(self.prompt, "template", None))
        if not has_prompt:
            raise ValueError(
                "Prompt template must be initialized before constructing messages."
            )

        chat_history = self.get_chat_history()  # List[Dict[str, Any]]

        if isinstance(input_data, str):
            formatted_messages = self.prompt.template.format(chat_history=chat_history)
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
            formatted_messages = self.prompt.template.format(**input_vars)
            if isinstance(formatted_messages, list):
                return formatted_messages
            else:
                return [{"role": "system", "content": formatted_messages}]

        else:
            raise ValueError("Input data must be either a string or dictionary.")

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the last message from the chat history.

        Returns:
            Optional[Dict[str, Any]]: The last message in the history as a dictionary, or None if none exist.
        """
        chat_history = self.get_chat_history()
        if chat_history:
            last_msg = chat_history[-1]
            if isinstance(last_msg, BaseMessage):
                return last_msg.model_dump()
            return last_msg
        return None

    def get_last_user_message(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves the last user message in a list of messages, returning a copy with trimmed content.

        Args:
            messages (List[Dict[str, Any]]): List of formatted messages to search.

        Returns:
            Optional[Dict[str, Any]]: The last user message (copy) with trimmed content, or None if no user message exists.
        """
        # Iterate in reverse to find the most recent 'user' role message
        for message in reversed(messages):
            if message.get("role") == "user":
                # Return a copy with trimmed content
                msg_copy = dict(message)
                msg_copy["content"] = msg_copy["content"].strip()
                return msg_copy
        return None

    def get_last_message_if_user(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Returns the last message only if it is a user message; otherwise, returns None.

        Args:
            messages (List[Dict[str, Any]]): List of formatted messages to check.

        Returns:
            Optional[Dict[str, Any]]: The last message (copy) with trimmed content if it is a user message, else None.
        """
        if messages and messages[-1].get("role") == "user":
            msg_copy = dict(messages[-1])
            msg_copy["content"] = msg_copy["content"].strip()
            return msg_copy
        return None

    def get_llm_tools(self) -> List[Union[AgentTool, Dict[str, Any]]]:
        """
        Converts tools to the format expected by LLM clients.

        Returns:
            List[Union[AgentTool, Dict[str, Any]]]: Tools in LLM-compatible format.
        """
        llm_tools: List[Union[AgentTool, Dict[str, Any]]] = []
        for tool in self.tools:
            if isinstance(tool, AgentTool):
                llm_tools.append(tool)
            elif callable(tool):
                try:
                    agent_tool = AgentTool.from_func(tool)
                    llm_tools.append(agent_tool)
                except Exception as e:
                    logger.warning(f"Failed to convert callable to AgentTool: {e}")
                    continue
        return llm_tools

    def register_agent(
        self, store_name: str, store_key: str, agent_name: str, agent_metadata: dict
    ) -> None:
        """
        Merges the existing data with the new data and updates the store.
        Only works for agents with Dapr client access (AgenticWorkflow subclasses).

        Args:
            store_name (str): The name of the Dapr state store component.
            store_key (str): The key to update.
            agent_name (str): The name of the agent to register.
            agent_metadata (dict): The metadata to register for the agent.
        """
        import json
        import time
        from dapr.clients.grpc._response import StateResponse
        from dapr.clients.grpc._state import StateOptions, Concurrency, Consistency
        from dapr.clients.grpc._request import (
            TransactionalStateOperation,
            TransactionOperationType,
        )

        # Only proceed if agent has Dapr client
        if not hasattr(self, "_dapr_client"):
            logger.debug(
                f"Agent '{self.name}' does not have Dapr client, skipping registration"
            )
            return

        # retry the entire operation up to twenty times sleeping 1-2 seconds between each
        # TODO: rm the custom retry logic here and use the DaprClient retry_policy instead.
        for attempt in range(1, 21):
            try:
                # Get current registry and etag
                response: StateResponse = self._dapr_client.get_state(
                    store_name=store_name, key=store_key
                )
                if not response.etag:
                    # if there is no etag the following transaction won't work as expected
                    # so we need to save an empty object with a strong consistency to force the etag to be created
                    self._dapr_client.save_state(
                        store_name=store_name,
                        key=store_key,
                        value=json.dumps({}),
                        state_metadata={
                            "contentType": "application/json",
                            "partitionKey": store_key,
                        },
                        options=StateOptions(
                            concurrency=Concurrency.first_write,
                            consistency=Consistency.strong,
                        ),
                    )

                    # reread to obtain the freshly minted ETag
                    response = self._dapr_client.get_state(
                        store_name=store_name, key=store_key
                    )
                    if not response.etag:
                        raise RuntimeError("ETag still missing after init")

                existing = (
                    self._deserialize_state(response.data) if response.data else {}
                )

                if existing.get(agent_name) == agent_metadata:
                    logger.debug(f"Agent '{agent_name}' already registered")
                    return

                safe_metadata = self._serialize_metadata(agent_metadata)

                merged = {**existing, agent_name: safe_metadata}
                merged_json = json.dumps(merged)

                logger.debug(f"merged data: {merged_json} etag: {response.etag}")
                try:
                    # using the transactional API to be able to later support the Dapr outbox pattern
                    self._dapr_client.execute_state_transaction(
                        store_name=store_name,
                        operations=[
                            TransactionalStateOperation(
                                key=store_key,
                                data=merged_json,
                                etag=response.etag,
                                operation_type=TransactionOperationType.upsert,
                            )
                        ],
                        transactional_metadata={
                            "contentType": "application/json",
                            "partitionKey": store_key,
                        },
                    )
                except Exception as e:
                    raise e
                return None
            except Exception as e:
                logger.error(f"Error on transaction attempt: {attempt}: {e}")
                # Add random jitter
                import random

                delay = 1 + random.uniform(0, 1)  # 1-2 seconds
                logger.info(
                    f"Sleeping for {delay:.2f} seconds before retrying transaction..."
                )
                time.sleep(delay)
        raise Exception(
            f"Failed to update state store key: {store_key} after 20 attempts."
        )

    def pre_fill_prompt_template(self, **kwargs: Union[str, Callable[[], str]]) -> None:
        """
        Pre-fills the prompt template with specified variables, updating input variables if applicable.

        Args:
            **kwargs: Variables to pre-fill in the prompt template. These can be strings or callables
                    that return strings.

        Notes:
            - Existing pre-filled variables will be overwritten by matching keys in `kwargs`.
            - This method does not affect the `chat_history` which is dynamically updated.
        """
        if not self.prompt_template:
            raise ValueError(
                "Prompt template must be initialized before pre-filling variables."
            )

        self.prompt_template = self.prompt_template.pre_fill_variables(**kwargs)
        logger.debug(f"Pre-filled prompt template with variables: {kwargs.keys()}")

    def _deserialize_state(self, raw: Union[bytes, str, dict]) -> dict:
        """
        Convert Dapr's raw payload (bytes, JSON string, or already a dict) into a dict.
        Raises helpful errors on failure.
        """
        if isinstance(raw, dict):
            return raw

        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("State bytes are not valid UTF-8") from exc

        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"State is not valid JSON: {exc}") from exc

        raise TypeError(f"Unsupported state type {type(raw)!r}")

    def _serialize_metadata(self, metadata: Any) -> Any:
        """
        Recursively convert Pydantic models (e.g., AgentTool), lists, dicts to JSON-serializable format.
        Handles mixed tools: [AgentTool(...), "string", ...] → [{"name": "..."}, "string", ...]
        """

        def convert(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"):
                return obj.dict()
            if isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        return convert(metadata)
