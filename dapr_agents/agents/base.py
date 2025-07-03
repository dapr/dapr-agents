from dapr_agents.memory import (
    MemoryBase,
    ConversationListMemory,
    ConversationVectorMemory,
)
from dapr_agents.storage import VectorStoreBase
from dapr_agents.agents.utils.text_printer import ColorTextFormatter
from dapr_agents.types import (
    MessageContent,
    MessagePlaceHolder,
    BaseMessage,
)
from dapr_agents.tool.executor import AgentToolExecutor
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt import ChatPromptTemplate
from dapr_agents.tool.base import AgentTool
from typing import (
    List,
    Optional,
    Dict,
    Any,
    Union,
    Callable,
    Literal,
)
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ConfigDict
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import asyncio
import signal
from dapr_agents.llm.openai import OpenAIChatClient
from dapr_agents.llm.huggingface import HFHubChatClient
from dapr_agents.llm.nvidia import NVIDIAChatClient
from dapr_agents.llm.dapr import DaprChatClient

logger = logging.getLogger(__name__)

# Type alias for all concrete chat client implementations
ChatClientType = Union[
    OpenAIChatClient, HFHubChatClient, NVIDIAChatClient, DaprChatClient
]


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
    system_prompt: Optional[str] = Field(
        default=None,
        description="A custom system prompt, overriding name, role, goal, and instructions.",
    )
    llm: ChatClientType = Field(
        default_factory=OpenAIChatClient,
        description="Language model client for generating responses.",
    )
    prompt_template: Optional[PromptTemplateBase] = Field(
        default=None, description="The prompt template for the agent."
    )
    # TODO: we need to add RBAC to tools to define what users and/or agents can use what tool(s).
    tools: List[Union[AgentTool, Callable]] = Field(
        default_factory=list,
        description="Tools available for the agent to assist with tasks.",
    )
    # TODO: add a forceFinalAnswer field in case maxIterations is near/reached. Or do we have a conclusion baked in by default? Do we want this to derive a conclusion by default?
    max_iterations: int = Field(
        default=10, description="Max iterations for conversation cycles."
    )
    # NOTE for reviewer: am I missing anything else here for vector stores?
    vector_store: Optional[VectorStoreBase] = Field(
        default=None,
        description="Vector store to enable semantic search and retrieval.",
    )
    memory: MemoryBase = Field(
        default_factory=ConversationListMemory,
        description="Handles conversation history and context storage.",
    )
    # TODO: we should have a system_template, prompt_template, and response_template, or better separation here.
    # If we have something like a customer service agent, we want diff templates for different types of interactions.
    # In future, we could also have a way to dynamically change the template based on the context of the interaction.
    template_format: Literal["f-string", "jinja2"] = Field(
        default="jinja2",
        description="The format used for rendering the prompt template.",
    )

    _tool_executor: AgentToolExecutor = PrivateAttr()
    _text_formatter: ColorTextFormatter = PrivateAttr(
        default_factory=ColorTextFormatter
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    def set_name_from_role(cls, values: dict):
        # Set name to role if name is not provided
        if not values.get("name") and values.get("role"):
            values["name"] = values["role"]
        return values

    @model_validator(mode="after")
    def validate_llm(cls, values):
        """Validate that LLM is properly configured."""
        if hasattr(values, "llm") and values.llm:
            try:
                # Validate LLM is properly configured by accessing it as this is required to be set.
                _ = values.llm
            except Exception as e:
                raise ValueError(f"Failed to initialize LLM: {e}") from e

        return values

    def model_post_init(self, __context: Any) -> None:
        """
        Sets up the prompt template based on system_prompt or attributes like name, role, goal, and instructions.
        Confirms the source of prompt_template post-initialization.
        """
        self._tool_executor = AgentToolExecutor(tools=self.tools)

        if self.prompt_template and self.llm.prompt_template:
            raise ValueError(
                "Conflicting prompt templates: both an agent prompt_template and an LLM prompt_template are provided. "
                "Please set only one or ensure synchronization between the two."
            )

        if self.prompt_template:
            logger.info(
                "Using the provided agent prompt_template. Skipping system prompt construction."
            )
            self.llm.prompt_template = self.prompt_template

        # If the LLM client already has a prompt template, sync it and prefill/validate as needed
        elif self.llm.prompt_template:
            logger.info("Using existing LLM prompt_template. Synchronizing with agent.")
            self.prompt_template = self.llm.prompt_template

        else:
            if not self.system_prompt:
                logger.info("Constructing system_prompt from agent attributes.")
                self.system_prompt = self.construct_system_prompt()

            logger.info("Using system_prompt to create the prompt template.")
            self.prompt_template = self.construct_prompt_template()

        if not self.llm.prompt_template:
            self.llm.prompt_template = self.prompt_template

        self._validate_prompt_template()
        self.prefill_agent_attributes()

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

    def _validate_prompt_template(self) -> None:
        """
        Validates that the prompt template is properly constructed and attributes are handled correctly.
        This runs after prompt template setup to ensure all attributes are properly handled.
        """
        if not self.prompt_template:
            return

        input_variables = ["chat_history"]  # Always include chat_history
        if self.name:
            input_variables.append("name")
        if self.role:
            input_variables.append("role")
        if self.goal:
            input_variables.append("goal")
        if self.instructions:
            input_variables.append("instructions")

        self.prompt_template.input_variables = list(
            set(self.prompt_template.input_variables + input_variables)
        )

        # Collect attributes set by user
        set_attributes = {
            "name": self.name,
            "role": self.role,
            "goal": self.goal,
            "instructions": self.instructions,
        }

        # Use Pydantic's model_fields_set to detect if attributes were user-set
        user_set_attributes = {
            attr for attr in set_attributes if attr in self.model_fields_set
        }

        # Check if attributes are in input_variables
        ignored_attributes = [
            attr
            for attr in set_attributes
            if attr not in self.prompt_template.input_variables
            and set_attributes[attr] is not None
            and attr in user_set_attributes
        ]

        if ignored_attributes:
            logger.warning(
                f"The following agent attributes were explicitly set but are not in the prompt template: {', '.join(ignored_attributes)}. "
                "These will be handled during initialization."
            )

    @property
    def tool_executor(self) -> AgentToolExecutor:
        """Returns the client to execute and manage tools, ensuring it's accessible but read-only."""
        return self._tool_executor

    @property
    def text_formatter(self) -> ColorTextFormatter:
        """Returns the text formatter for the agent."""
        return self._text_formatter

    @property
    def chat_history(self, task: Optional[str] = None) -> List[MessageContent]:
        """
        Retrieves the chat history from memory based on the memory type.

        Args:
            task (Optional[str]): The task or query provided by the user.

        Returns:
            List[MessageContent]: The chat history.
        """
        if (
            isinstance(self.memory, ConversationVectorMemory)
            and task
            and self.vector_store
        ):
            if (
                hasattr(self.vector_store, "embedding_function")
                and self.vector_store.embedding_function
                and hasattr(self.vector_store.embedding_function, "embed_documents")
            ):
                query_embeddings = self.vector_store.embedding_function.embed_documents(
                    [task]
                )
                return self.memory.get_messages(
                    query_embeddings=query_embeddings
                )  # returns List[MessageContent]
            else:
                return self.memory.get_messages()  # returns List[MessageContent]
        else:
            messages = (
                self.memory.get_messages()
            )  # returns List[BaseMessage] or List[Dict]
            converted_messages: List[MessageContent] = []
            for msg in messages:
                if isinstance(msg, MessageContent):
                    converted_messages.append(msg)
                elif isinstance(msg, BaseMessage):
                    converted_messages.append(MessageContent(**msg.model_dump()))
                elif isinstance(msg, dict):
                    converted_messages.append(MessageContent(**msg))
                else:
                    # Fallback: try to convert to dict and then to MessageContent
                    converted_messages.append(MessageContent(**dict(msg)))
            return converted_messages

    @abstractmethod
    def run(self, input_data: Union[str, Dict[str, Any]]) -> Any:
        """
        Executes the agent's main logic based on provided inputs.

        Args:
            inputs (Dict[str, Any]): A dictionary with dynamic input values for task execution.
        """
        pass

    def prefill_agent_attributes(self) -> None:
        """
        Pre-fill prompt template with agent attributes if specified in `input_variables`.
        Logs any agent attributes set but not used by the template.
        """
        if not self.prompt_template:
            return

        prefill_data = {}
        if "name" in self.prompt_template.input_variables and self.name:
            prefill_data["name"] = self.name

        if "role" in self.prompt_template.input_variables:
            prefill_data["role"] = self.role or ""

        if "goal" in self.prompt_template.input_variables:
            prefill_data["goal"] = self.goal or ""

        if "instructions" in self.prompt_template.input_variables and self.instructions:
            prefill_data["instructions"] = "\n".join(self.instructions)

        # Collect attributes set but not in input_variables for informational logging
        set_attributes = {
            "name": self.name,
            "role": self.role,
            "goal": self.goal,
            "instructions": self.instructions,
        }

        # Use Pydantic's model_fields_set to detect if attributes were user-set
        user_set_attributes = {
            attr for attr in set_attributes if attr in self.model_fields_set
        }

        ignored_attributes = [
            attr
            for attr in set_attributes
            if attr not in self.prompt_template.input_variables
            and set_attributes[attr] is not None
            and attr in user_set_attributes
        ]

        # Apply pre-filled data only for attributes that are in input_variables
        if prefill_data:
            self.prompt_template = self.prompt_template.pre_fill_variables(
                **prefill_data
            )
            logger.info(
                f"Pre-filled prompt template with attributes: {list(prefill_data.keys())}"
            )
        elif ignored_attributes:
            raise ValueError(
                f"The following agent attributes were explicitly set by the user but are not considered by the prompt template: {', '.join(ignored_attributes)}. "
                "Please ensure that these attributes are included in the prompt template's input variables if they are needed."
            )
        else:
            logger.info(
                "No agent attributes were pre-filled, as the template did not require any."
            )

    def construct_system_prompt(self) -> str:
        """
        Constructs a system prompt with agent attributes like `name`, `role`, `goal`, and `instructions`.
        Sets default values for `role` and `goal` if not provided.

        Returns:
            str: A system prompt template string.
        """
        # Initialize prompt parts with the current date as the first entry
        prompt_parts = [f"# Today's date is: {datetime.now().strftime('%B %d, %Y')}"]

        # Append name if provided
        if self.name:
            prompt_parts.append("## Name\nYour name is {{name}}.")

        # Append role and goal with default values if not set
        prompt_parts.append("## Role\nYour role is {{role}}.")
        prompt_parts.append("## Goal\n{{goal}}.")

        # Append instructions if provided
        if self.instructions:
            prompt_parts.append("## Instructions\n{{instructions}}")

        return "\n\n".join(prompt_parts)

    def construct_prompt_template(self) -> ChatPromptTemplate:
        """
        Constructs a ChatPromptTemplate that includes the system prompt and a placeholder for chat history.
        Ensures that the template is flexible and adaptable to dynamically handle pre-filled variables.

        Returns:
            ChatPromptTemplate: A formatted prompt template for the agent.
        """
        # Construct the system prompt if not provided
        system_prompt = self.system_prompt or self.construct_system_prompt()

        # Create the template with placeholders for system message and chat history
        return ChatPromptTemplate.from_messages(
            messages=[
                ("system", system_prompt),
                MessagePlaceHolder(variable_name="chat_history"),
            ],
            template_format=self.template_format,
        )

    def construct_messages(
        self, input_data: Union[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Constructs and formats initial messages based on input type, pre-filling chat history as needed.

        Args:
            input_data (Union[str, Dict[str, Any]]): User input, either as a string or dictionary.

        Returns:
            List[Dict[str, Any]]: List of formatted messages, including the user message if input_data is a string.
        """
        if not self.prompt_template:
            raise ValueError(
                "Prompt template must be initialized before constructing messages."
            )

        # Pre-fill chat history in the prompt template
        chat_history = self.memory.get_messages()
        # Convert List[BaseMessage] to string for the prompt template
        chat_history_str = "\n".join([str(msg) for msg in chat_history])
        self.pre_fill_prompt_template(chat_history=chat_history_str)

        # Handle string input by adding a user message
        if isinstance(input_data, str):
            formatted_messages = self.prompt_template.format_prompt()
            if isinstance(formatted_messages, list):
                user_message = {"role": "user", "content": input_data}
                return formatted_messages + [user_message]
            else:
                return [
                    {"role": "system", "content": formatted_messages},
                    {"role": "user", "content": input_data},
                ]

        # Handle dictionary input as dynamic variables for the template
        elif isinstance(input_data, dict):
            # Pass the dictionary directly, assuming it contains keys expected by the prompt template
            formatted_messages = self.prompt_template.format_prompt(**input_data)
            if isinstance(formatted_messages, list):
                return formatted_messages
            else:
                return [{"role": "system", "content": formatted_messages}]

        else:
            raise ValueError("Input data must be either a string or dictionary.")

    def reset_memory(self):
        """Clears all messages stored in the agent's memory."""
        self.memory.reset_memory()

    def get_last_message(self) -> Optional[MessageContent]:
        """
        Retrieves the last message from the chat history.

        Returns:
            Optional[MessageContent]: The last message in the history, or None if none exist.
        """
        chat_history = self.chat_history
        if chat_history:
            last_msg = chat_history[-1]
            # Ensure we return MessageContent type
            if isinstance(last_msg, BaseMessage) and not isinstance(
                last_msg, MessageContent
            ):
                return MessageContent(**last_msg.model_dump())
            return last_msg
        return None

    def get_last_user_message(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves the last user message in a list of messages.

        Args:
            messages (List[Dict[str, Any]]): List of formatted messages to search.

        Returns:
            Optional[Dict[str, Any]]: The last user message with trimmed content, or None if no user message exists.
        """
        # Iterate in reverse to find the most recent 'user' role message
        for message in reversed(messages):
            if message.get("role") == "user":
                # Trim the content of the user message
                message["content"] = message["content"].strip()
                return message
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
