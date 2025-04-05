import asyncio
import inspect
import logging
from dataclasses import is_dataclass
from functools import update_wrapper
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from dapr.ext.workflow import WorkflowActivityContext

from dapr_agents.agent.base import AgentBase
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.openai import OpenAIChatClient
from dapr_agents.llm.utils import StructureHandler
from dapr_agents.types import BaseMessage, ChatCompletion, UserMessage

logger = logging.getLogger(__name__)

class WorkflowTask(BaseModel):
    """
    Encapsulates task logic for execution by an LLM, agent, or Python function.

    Supports both synchronous and asynchronous tasks, with optional output validation
    using Pydantic models or specified return types.
    """

    func: Optional[Callable] = Field(None, description="The original function to be executed, if provided.")
    description: Optional[str] = Field(None, description="A description template for the task, used with LLM or agent.")
    agent: Optional[AgentBase] = Field(None, description="The agent used for task execution, if applicable.")
    llm: Optional[ChatClientBase] = Field(None, description="The LLM client for executing the task, if applicable.")
    include_chat_history: Optional[bool] = Field(False, description="Whether to include past conversation history in the LLM call.")
    workflow_app: Optional[Any] = Field(None, description="Reference to the WorkflowApp instance.")
    structured_mode: Literal["json", "function_call"] = Field(default="json", description="Structured response mode for LLM output. Valid values: 'json', 'function_call'.")
    task_kwargs: Dict[str, Any] = Field(default_factory=dict, exclude=True, description="Additional keyword arguments passed via the @task decorator.")

    # Initialized during setup
    signature: Optional[inspect.Signature] = Field(None, init=False, description="The signature of the provided function.")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to set up function signatures and default LLM clients.
        """
        if self.description and not self.llm:
            self.llm = OpenAIChatClient()
        
        if self.func:
            update_wrapper(self, self.func)

        self.signature = inspect.signature(self.func) if self.func else None

        if not self.structured_mode and "structured_mode" in self.task_kwargs:
            self.structured_mode = self.task_kwargs["structured_mode"]

        # Proceed with base model setup
        super().model_post_init(__context)
    
    async def __call__(self, ctx: WorkflowActivityContext, input: Any = None) -> Any:
        """
        Executes the task and validates its output.
        Ensures all coroutines are awaited before returning.

        Args:
            ctx (WorkflowActivityContext): The workflow execution context.
            input (Any): The task input.

        Returns:
            Any: The result of the task.
        """
        input = self._normalize_input(input) if input is not None else {}

        try:
            if self.agent or self.llm:
                if not self.description:
                    raise ValueError(f"Task {self.func.__name__} is LLM-based but has no description!")

                result = await self._run_task(self.format_description(self.description, input))
                result = await self._validate_output(result)
            elif self.func:
                # Task is a Python function
                logger.info(f"Invoking Regular Task")
                if asyncio.iscoroutinefunction(self.func):
                    # Await async function
                    result = await self.func(**input)
                else:
                    # Call sync function
                    result = self._execute_function(input)
                result = await self._validate_output(result)
            else:
                raise ValueError("Task must have an LLM, agent, or regular function for execution.")
            
            return result
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            raise

    def _normalize_input(self, input: Any) -> dict:
        """
        Converts input into a normalized dictionary.

        Args:
            input (Any): Input to normalize (e.g., dictionary, dataclass, or object).

        Returns:
            dict: Normalized dictionary representation of the input.
        """
        if is_dataclass(input):
            return input.__dict__
        elif isinstance(input, SimpleNamespace):
            return vars(input)
        elif not isinstance(input, dict):
            return self._single_value_to_dict(input)
        return input

    def _single_value_to_dict(self, value: Any) -> dict:
        """
        Wraps a single input value in a dictionary.

        Args:
            value (Any): Single input value.

        Returns:
            dict: Dictionary with parameter name as the key.

        Raises:
            ValueError: If no function signature is available.
        """
        if not self.signature:
            raise ValueError("Cannot convert single input to dict: function signature is missing.")

        param_name = list(self.signature.parameters.keys())[0]
        return {param_name: value}
    
    def format_description(self, description: str, input: dict) -> str:
        """
        Formats a description string with input parameters.

        Args:
            description (str): Description template.
            input (dict): Input parameters for formatting.

        Returns:
            str: Formatted description string.
        """
        if self.signature:
            bound_args = self.signature.bind(**input)
            bound_args.apply_defaults()
            return description.format(**bound_args.arguments)
        return description.format(**input)
    
    async def _run_task(self, formatted_description: str) -> Any:
        """
        Determine whether to run the task using an agent or an LLM.

        Args:
            formatted_description (str): The formatted description to pass to the agent or LLM.

        Returns:
            Any: The result of the agent or LLM execution.

        Raises:
            ValueError: If neither an agent nor an LLM is provided.
        """
        logger.debug(f"Task Description: {formatted_description}")

        if self.agent:
            return await self._run_agent(formatted_description)
        elif self.llm:
            return await self._run_llm(formatted_description)
        else:
            raise ValueError("No agent or LLM provided.")

    async def _run_agent(self, description: str) -> Any:
        """
        Execute the task using the provided agent.

        Args:
            description (str): The formatted description to pass to the agent.

        Returns:
            Any: The result of the agent execution.
        """
        logger.info("Invoking Task with AI Agent...")

        result = await self.agent.run(description)

        logger.debug(f"Agent result type: {type(result)}, value: {result}")
        return self._convert_result(result)
    
    async def _run_llm(self, description: Union[str, List[BaseMessage]]) -> Any:
        logger.info("Invoking Task with LLM...")

        # 1. Get chat history if enabled
        conversation_history = []
        if self.include_chat_history and self.workflow_app:
            logger.info("Retrieving conversation history...")
            conversation_history = self.workflow_app.get_chat_history()
            logger.debug(f"Conversation history retrieved: {conversation_history}")

        # 2. Convert string input to structured messages
        if isinstance(description, str):
            description = [UserMessage(description)]
        llm_messages = conversation_history + description

        # 3. Base LLM parameters
        llm_params = {"messages": llm_messages}

        # 4. Add structured response config if a valid Pydantic model is the return type
        if self.signature and self.signature.return_annotation is not inspect.Signature.empty:
            return_type = self.signature.return_annotation
            model_cls = StructureHandler.resolve_response_model(return_type)

            # Only proceed if we resolved a Pydantic model
            if model_cls:
                if not hasattr(self.llm, "provider"):
                    raise AttributeError(
                        f"{type(self.llm).__name__} is missing the `.provider` attribute — required for structured response generation."
                    )

                logger.debug(f"Using LLM provider: {self.llm.provider}")

                llm_params["response_format"] = return_type
                llm_params["structured_mode"] = self.structured_mode or "json"

        # 5. Call the LLM client
        result = self.llm.generate(**llm_params)

        logger.debug(f"LLM result type: {type(result)}, value: {result}")
        return self._convert_result(result)

    def _convert_result(self, result: Any) -> Any:
        """
        Convert the task result to a dictionary if necessary.

        Args:
            result (Any): The raw task result.

        Returns:
            Any: The converted result.
        """
        if isinstance(result, ChatCompletion):
            logger.debug("Extracted message content from ChatCompletion.")
            return result.get_content()

        if isinstance(result, BaseModel):
            logger.debug("Converting Pydantic model to dictionary.")
            return result.model_dump()

        if isinstance(result, list) and all(isinstance(item, BaseModel) for item in result):
            logger.debug("Converting list of Pydantic models to list of dictionaries.")
            return [item.model_dump() for item in result]

        # If no specific conversion is necessary, return as-is
        logger.info("Returning final task result.")
        return result
    
    def _execute_function(self, input: dict) -> Any:
        """
        Execute the wrapped function with the provided input.

        Args:
            input (dict): The input data to pass to the function.

        Returns:
            Any: The result of the function execution.
        """
        return self.func(**input)
    
    async def _validate_output(self, result: Any) -> Any:
        """
        Validates the output of the task against the expected return type.

        Supports coroutine outputs and structured type validation.

        Returns:
            Any: The validated and potentially transformed result.
        """
        if asyncio.iscoroutine(result):
            logger.warning("Result is a coroutine; awaiting.")
            result = await result

        if not self.signature or self.signature.return_annotation is inspect.Signature.empty:
            return result

        expected_type = self.signature.return_annotation
        return StructureHandler.validate_against_signature(result, expected_type)

class TaskWrapper:
    """
    A wrapper for WorkflowTask that preserves callable behavior and attributes like __name__.
    """

    def __init__(self, task_instance: WorkflowTask, name: str):
        """
        Initialize the TaskWrapper.

        Args:
            task_instance (WorkflowTask): The task instance to wrap.
            name (str): The task name.
        """
        self.task_instance = task_instance
        self.__name__ = name
        self.__doc__ = getattr(task_instance.func, "__doc__", None)
        self.__module__ = getattr(task_instance.func, "__module__", None)

    def __call__(self, *args, **kwargs):
        """
        Delegate the call to the wrapped WorkflowTask instance.
        """
        return self.task_instance(*args, **kwargs)

    def __getattr__(self, item):
        """
        Delegate attribute access to the wrapped task.
        """
        return getattr(self.task_instance, item)

    def __repr__(self):
        return f"<TaskWrapper name={self.__name__}>"