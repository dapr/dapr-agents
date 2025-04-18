from dapr_agents.memory import MemoryBase, ConversationListMemory, ConversationVectorMemory
from dapr_agents.agent.utils.text_printer import ColorTextFormatter
from dapr_agents.types import MessageContent, MessagePlaceHolder
from dapr_agents.tool.executor import AgentToolExecutor
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.llm import LLMClientBase, OpenAIChatClient
from dapr_agents.prompt import ChatPromptTemplate
from dapr_agents.tool.base import AgentTool
from typing import List, Optional, Dict, Any, Union, Callable, Literal
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ConfigDict
from abc import ABC, abstractmethod
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AgentBase(BaseModel, ABC):
    """
    Base class for agents that interact with language models and manage tools for task execution.
    """

    name: Optional[str] = Field(default=None, description="The agent's name, defaulting to the role if not provided.")
    role: Optional[str] = Field(default="Assistant", description="The agent's role in the interaction (e.g., 'Weather Expert').")
    goal: Optional[str] = Field(default="Help humans", description="The agent's main objective (e.g., 'Provide Weather information').")
    instructions: Optional[List[str]] = Field(default=None, description="Instructions guiding the agent's tasks.")
    system_prompt: Optional[str] = Field(default=None, description="A custom system prompt, overriding name, role, goal, and instructions.")
    llm: LLMClientBase = Field(default_factory=OpenAIChatClient, description="Language model client for generating responses.")
    prompt_template: Optional[PromptTemplateBase] = Field(default=None, description="The prompt template for the agent.")
    tools: List[Union[AgentTool, Callable]] = Field(default_factory=list, description="Tools available for the agent to assist with tasks.")
    max_iterations: int = Field(default=10, description="Max iterations for conversation cycles.")
    memory: MemoryBase = Field(default_factory=ConversationListMemory, description="Handles conversation history and context storage.")
    template_format: Literal["f-string", "jinja2"] = Field(default="jinja2", description="The format used for rendering the prompt template.")

    # Private attributes
    _tool_executor: AgentToolExecutor = PrivateAttr()
    _text_formatter: ColorTextFormatter = PrivateAttr(default_factory=ColorTextFormatter)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @model_validator(mode="before")
    def set_name_from_role(cls, values: dict):
        # Set name to role if name is not provided
        if not values.get("name") and values.get("role"):
            values["name"] = values["role"]
        return values
    
    @property
    def tool_executor(self) -> AgentToolExecutor:
        """Returns the tool executor, ensuring it's accessible but read-only."""
        return self._tool_executor
    
    @property
    def text_formatter(self) -> ColorTextFormatter:
        """Returns the text formatter for the agent."""
        return self._text_formatter
    
    @property
    def chat_history(self, task: str = None) -> List[MessageContent]:
        """
        Retrieves the chat history from memory based on the memory type.

        Args:
            task (str): The task or query provided by the user.

        Returns:
            List[MessageContent]: The chat history.
        """
        if isinstance(self.memory, ConversationVectorMemory) and task:
            query_embeddings = self.memory.vector_store.embed_documents([task])
            return self.memory.get_messages(query_embeddings=query_embeddings)
        return self.memory.get_messages()
    
    @abstractmethod
    def run(self, input_data: Union[str, Dict[str, Any]]) -> Any:
        """
        Executes the agent's main logic based on provided inputs.

        Args:
            inputs (Dict[str, Any]): A dictionary with dynamic input values for task execution.
        """
        pass

    def model_post_init(self, __context: Any) -> None:
        """
        Sets up the prompt template based on system_prompt or attributes like name, role, goal, and instructions.
        Confirms the source of prompt_template post-initialization.
        """
        # Initialize tool executor with provided tools
        self._tool_executor = AgentToolExecutor(tools=self.tools)

        # Check if both agent and LLM have a prompt template specified and raise an error if both exist
        if self.prompt_template and self.llm.prompt_template:
            raise ValueError(
                "Conflicting prompt templates: both an agent prompt_template and an LLM prompt_template are provided. "
                "Please set only one or ensure synchronization between the two."
            )

        # If the agent's prompt_template is provided, use it and skip further configuration
        if self.prompt_template:
            logger.info("Using the provided agent prompt_template. Skipping system prompt construction.")
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
        
        # Pre-fill Agent Attributes if needed
        self.prefill_agent_attributes()

        if not self.llm.prompt_template:
            # Assign the prompt template to the LLM client
            self.llm.prompt_template = self.prompt_template

        # Complete post-initialization
        super().model_post_init(__context)

    def prefill_agent_attributes(self) -> None:
        """
        Pre-fill prompt template with agent attributes if specified in `input_variables`.
        Logs any agent attributes set but not used by the template.
        """
        # Start with a dictionary for attributes
        prefill_data = {}

        # Check if each attribute is defined in input_variables before adding
        if "name" in self.prompt_template.input_variables and self.name:
            prefill_data["name"] = self.name

        if "role" in self.prompt_template.input_variables:
            prefill_data["role"] = self.role

        if "goal" in self.prompt_template.input_variables:
            prefill_data["goal"] = self.goal

        if "instructions" in self.prompt_template.input_variables and self.instructions:
            prefill_data["instructions"] = "\n".join(self.instructions)

        # Collect attributes set but not in input_variables for informational logging
        set_attributes = {"name": self.name, "role": self.role, "goal": self.goal, "instructions": self.instructions}
        
        # Use Pydantic's model_fields_set to detect if attributes were user-set
        user_set_attributes = {attr for attr in set_attributes if attr in self.model_fields_set}
        
        ignored_attributes = [
            attr for attr in set_attributes
            if attr not in self.prompt_template.input_variables and set_attributes[attr] is not None and attr in user_set_attributes
        ]

        # Apply pre-filled data only for attributes that are in input_variables
        if prefill_data:
            self.prompt_template = self.prompt_template.pre_fill_variables(**prefill_data)
            logger.info(f"Pre-filled prompt template with attributes: {list(prefill_data.keys())}")
        elif ignored_attributes:
            raise ValueError(
                f"The following agent attributes were explicitly set by the user but are not considered by the prompt template: {', '.join(ignored_attributes)}. "
                "Please ensure that these attributes are included in the prompt template's input variables if they are needed."
            )
        else:
            logger.info("No agent attributes were pre-filled, as the template did not require any.")
    
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
                ('system', system_prompt),
                MessagePlaceHolder(variable_name="chat_history")
            ],
            template_format=self.template_format
        )
    
    def construct_messages(self, input_data: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Constructs and formats initial messages based on input type, pre-filling chat history as needed.

        Args:
            input_data (Union[str, Dict[str, Any]]): User input, either as a string or dictionary.
        
        Returns:
            List[Dict[str, Any]]: List of formatted messages, including the user message if input_data is a string.
        """
        # Pre-fill chat history in the prompt template
        chat_history = self.memory.get_messages()
        self.pre_fill_prompt_template(**{"chat_history": chat_history})

        # Handle string input by adding a user message
        if isinstance(input_data, str):
            formatted_messages = self.prompt_template.format_prompt()
            user_message = {"role": "user", "content": input_data}
            return formatted_messages + [user_message]

        # Handle dictionary input as dynamic variables for the template
        elif isinstance(input_data, dict):
            # Pass the dictionary directly, assuming it contains keys expected by the prompt template
            formatted_messages = self.prompt_template.format_prompt(**input_data)
            return formatted_messages

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
        return chat_history[-1] if chat_history else None
    
    def get_last_user_message(self, messages: List[Dict[str, Any]]) -> Optional[MessageContent]:
        """
        Retrieves the last user message in a list of messages.

        Args:
            messages (List[Dict[str, Any]]): List of formatted messages to search.

        Returns:
            Optional[MessageContent]: The last user message with trimmed content, or None if no user message exists.
        """
        # Iterate in reverse to find the most recent 'user' role message
        for message in reversed(messages):
            if message.get("role") == "user":
                # Trim the content of the user message
                message["content"] = message["content"].strip()
                return message
        return None
    
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
            raise ValueError("Prompt template must be initialized before pre-filling variables.")
        
        self.prompt_template = self.prompt_template.pre_fill_variables(**kwargs)
        logger.debug(f"Pre-filled prompt template with variables: {kwargs.keys()}")