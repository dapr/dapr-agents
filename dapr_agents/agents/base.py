from dapr_agents.memory import (
    MemoryBase,
    ConversationListMemory,
    ConversationVectorMemory,
)
from dapr_agents.agents.utils.text_printer import ColorTextFormatter
from dapr_agents.types import MessagePlaceHolder, BaseMessage, ToolExecutionRecord
from dapr_agents.tool.executor import AgentToolExecutor
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt import ChatPromptTemplate
from dapr_agents.tool.base import AgentTool
import re
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
    Literal,
    ClassVar,
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
    system_prompt: Optional[str] = Field(
        default=None,
        description="A custom system prompt, overriding name, role, goal, and instructions.",
    )
    llm: Optional[ChatClientBase] = Field(
        default=None,
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
    # TODO(@Sicoyle): Rename this to make clearer
    memory: MemoryBase = Field(
        default_factory=ConversationListMemory,
        description="Handles long-term conversation history (for all workflow instance-ids within the same session) and context storage.",
    )
    # TODO: we should have a system_template, prompt_template, and response_template, or better separation here.
    # If we have something like a customer service agent, we want diff templates for different types of interactions.
    # In future, we could also have a way to dynamically change the template based on the context of the interaction.
    template_format: Literal["f-string", "jinja2"] = Field(
        default="jinja2",
        description="The format used for rendering the prompt template.",
    )

    DEFAULT_SYSTEM_PROMPT: ClassVar[str]
    """Default f-string template; placeholders will be swapped to Jinja if needed."""
    DEFAULT_SYSTEM_PROMPT = """
# Today's date is: {date}

## Name
Your name is {name}.

## Role
Your role is {role}.

## Goal
{goal}.

## Instructions
{instructions}.
""".strip()

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

        # Centralize prompt template selection logic
        self.prompt_template = self._initialize_prompt_template()
        # Ensure LLM client and agent both reference the same template
        if self.llm is not None:
            self.llm.prompt_template = self.prompt_template

        self._validate_prompt_template()
        self.prefill_agent_attributes()

        # Set up graceful shutdown
        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()

        super().model_post_init(__context)

    def _initialize_prompt_template(self) -> PromptTemplateBase:
        """
        Determines which prompt template to use for the agent:
        1. If the user supplied one, use it.
        2. Else if the LLM client already has one, adopt that.
        3. Else generate a system_prompt and ChatPromptTemplate from agent attributes.

        Returns:
            PromptTemplateBase: The selected or constructed prompt template.
        """
        # 1) User provided one?
        if self.prompt_template:
            logger.debug("🛠️ Using provided agent.prompt_template")
            return self.prompt_template

        # 2) LLM client has one?
        if (
            self.llm
            and hasattr(self.llm, "prompt_template")
            and self.llm.prompt_template
        ):
            logger.debug("🔄 Syncing from llm.prompt_template")
            return self.llm.prompt_template

        # 3) Build from system_prompt or attributes
        if not self.system_prompt:
            logger.debug("⚙️ Constructing system_prompt from attributes")
            self.system_prompt = self.construct_system_prompt()

        logger.debug("⚙️ Building ChatPromptTemplate from system_prompt")
        return self.construct_prompt_template()

    def _collect_template_attrs(self) -> tuple[Dict[str, str], List[str]]:
        """
        Collect agent attributes for prompt template pre-filling and warn about unused ones.
        - valid: attributes set on self and declared in prompt_template.input_variables.
        - unused: attributes set on self but not present in the template.
        Returns:
            (valid, unused): Tuple of dict of valid attrs and list of unused attr names.
        """
        attrs = ["name", "role", "goal", "instructions"]
        valid: Dict[str, str] = {}
        unused: List[str] = []
        if not self.prompt_template or not hasattr(
            self.prompt_template, "input_variables"
        ):
            return valid, attrs  # No template, all attrs are unused
        original = set(self.prompt_template.input_variables)

        for attr in attrs:
            val = getattr(self, attr, None)
            if val is None:
                continue
            if attr in original:
                # Only join instructions if it's a list and the template expects it
                if attr == "instructions" and isinstance(val, list):
                    valid[attr] = "\n".join(val)
                else:
                    valid[attr] = str(val)
            else:
                unused.append(attr)
        return valid, unused

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
        Ensures chat_history is always available, injects any declared attributes,
        and warns if the user set attributes that aren't in the template.
        """
        if not self.prompt_template:
            return

        # Always make chat_history available
        vars_set = set(self.prompt_template.input_variables) | {"chat_history"}

        # Inject any attributes the template declares
        valid_attrs, unused_attrs = self._collect_template_attrs()
        vars_set |= set(valid_attrs.keys())
        self.prompt_template.input_variables = list(vars_set)

        if unused_attrs:
            logger.warning(
                "Agent attributes set but not referenced in prompt_template: "
                f"{', '.join(unused_attrs)}. Consider adding them to input_variables."
            )

    @property
    def tool_executor(self) -> AgentToolExecutor:
        """Returns the client to execute and manage tools, ensuring it's accessible but read-only."""
        return self._tool_executor

    @property
    def text_formatter(self) -> ColorTextFormatter:
        """Returns the text formatter for the agent."""
        return self._text_formatter

    def get_chat_history(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the chat history from memory as a list of dictionaries.

        Args:
            task (Optional[str]): The task or query provided by the user (used for vector search).

        Returns:
            List[Dict[str, Any]]: The chat history as dictionaries.
        """
        if isinstance(self.memory, ConversationVectorMemory) and task:
            if (
                hasattr(self.memory.vector_store, "embedding_function")
                and self.memory.vector_store.embedding_function
                and hasattr(
                    self.memory.vector_store.embedding_function, "embed_documents"
                )
            ):
                query_embeddings = self.memory.vector_store.embedding_function.embed(
                    task
                )
                messages = self.memory.get_messages(query_embeddings=query_embeddings)
            else:
                messages = self.memory.get_messages()
        else:
            messages = self.memory.get_messages()
        return messages

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

    def prefill_agent_attributes(self) -> None:
        """
        Pre-fill prompt_template with agent attributes if specified in `input_variables`.
        Uses _collect_template_attrs to avoid duplicate logic and ensure consistency.
        """
        if not self.prompt_template:
            return

        # Re-use our helper to split valid vs. unused
        valid_attrs, unused_attrs = self._collect_template_attrs()

        if unused_attrs:
            logger.warning(
                "Agent attributes set but not used in prompt_template: "
                f"{', '.join(unused_attrs)}. Consider adding them to input_variables."
            )

        if valid_attrs:
            self.prompt_template = self.prompt_template.pre_fill_variables(
                **valid_attrs
            )
            logger.debug(f"Pre-filled template with: {list(valid_attrs.keys())}")
        else:
            logger.debug("No prompt_template variables needed pre-filling.")

    def construct_system_prompt(self) -> str:
        """
        Build the system prompt for the agent using a single template string.
        - Fills in the current date.
        - Leaves placeholders for name, role, goal, and instructions as variables (instructions only if set).
        - Converts placeholders to Jinja2 syntax if requested.

        Returns:
            str: The formatted system prompt string.
        """
        # Only fill in the date; leave all other placeholders as variables
        instructions_placeholder = "{instructions}" if self.instructions else ""
        filled = self.DEFAULT_SYSTEM_PROMPT.format(
            date=datetime.now().strftime("%B %d, %Y"),
            name="{name}",
            role="{role}",
            goal="{goal}",
            instructions=instructions_placeholder,
        )

        # If using Jinja2, swap braces for all placeholders
        if self.template_format == "jinja2":
            # Replace every {foo} with {{foo}}
            return re.sub(r"\{(\w+)\}", r"{{\1}}", filled)
        else:
            return filled

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
        Constructs and formats initial messages based on input type, passing chat_history as a list, without mutating self.prompt_template.

        Args:
            input_data (Union[str, Dict[str, Any]]): User input, either as a string or dictionary.

        Returns:
            List[Dict[str, Any]]: List of formatted messages, including the user message if input_data is a string.
        """
        if not self.prompt_template:
            raise ValueError(
                "Prompt template must be initialized before constructing messages."
            )

        chat_history = self.get_chat_history()  # List[Dict[str, Any]]

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

    def reset_memory(self):
        """Clears all messages stored in the agent's memory."""
        self.memory.reset_memory()

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
