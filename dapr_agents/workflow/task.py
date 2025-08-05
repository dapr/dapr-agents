import asyncio
import inspect
import logging
from dataclasses import is_dataclass
from functools import update_wrapper
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Optional

from dapr.ext.workflow import WorkflowActivityContext
from pydantic import BaseModel, ConfigDict, Field

from dapr_agents.agents.base import AgentBase
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.openai import OpenAIChatClient
from dapr_agents.llm.utils import StructureHandler
from dapr_agents.prompt.utils.chat import ChatPromptHelper
from dapr_agents.types import BaseMessage, UserMessage, LLMChatResponse

logger = logging.getLogger(__name__)


class WorkflowTask(BaseModel):
    """
    Encapsulates task logic for execution by an LLM, agent, or Python function.

    Supports both synchronous and asynchronous tasks, with optional output validation
    using Pydantic models or specified return types.
    """

    func: Optional[Callable] = Field(
        None, description="The original function to be executed, if provided."
    )
    description: Optional[str] = Field(
        None, description="A description template for the task, used with LLM or agent."
    )
    agent: Optional[AgentBase] = Field(
        None, description="The agent used for task execution, if applicable."
    )
    llm: Optional[ChatClientBase] = Field(
        None, description="The LLM client for executing the task, if applicable."
    )
    include_chat_history: Optional[bool] = Field(
        False,
        description="Whether to include past conversation history in the LLM call.",
    )
    workflow_app: Optional[Any] = Field(
        None, description="Reference to the WorkflowApp instance."
    )
    structured_mode: Literal["json", "function_call"] = Field(
        default="json",
        description="Structured response mode for LLM output. Valid values: 'json', 'function_call'.",
    )
    task_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description="Additional keyword arguments passed via the @task decorator.",
    )

    # Initialized during setup
    signature: Optional[inspect.Signature] = Field(
        None, init=False, description="The signature of the provided function."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to set up function signatures and default LLM clients.
        """
        # Default to OpenAIChatClient if prompt‐based but no llm provided
        if self.description and not self.llm:
            try:
                self.llm = OpenAIChatClient()
            except Exception as e:
                logger.warning(
                    f"Could not create default OpenAI client: {e}. Task will require explicit LLM."
                )
                self.llm = None

        if self.func:
            # Preserve name / docs for stack traces
            try:
                update_wrapper(self, self.func)
            except AttributeError:
                # If the function doesn't have the expected attributes, skip update_wrapper
                logger.debug(
                    f"Could not update wrapper for function {self.func}, skipping"
                )
                pass

        # Capture signature for input / output handling
        self.signature = inspect.signature(self.func) if self.func else None

        # Honor any structured_mode override
        if not self.structured_mode and "structured_mode" in self.task_kwargs:
            self.structured_mode = self.task_kwargs["structured_mode"]

        # Proceed with base model setup
        super().model_post_init(__context)

    async def __call__(self, ctx: WorkflowActivityContext, payload: Any = None) -> Any:
        """
        Executes the task, routing to agent, LLM, or pure-Python logic.

        Dispatches to Python, Agent, or LLM paths and validates output.

        Args:
            ctx (WorkflowActivityContext): The workflow execution context.
            payload (Any): The task input.

        Returns:
            Any: The result of the task.
        """
        # Prepare input dict
        data = self._normalize_input(payload) if payload is not None else {}
        func_name = getattr(self.func, "__name__", "unknown_function")
        logger.info(f"Executing task '{func_name}'")
        logger.debug(f"Executing task '{func_name}' with input {data!r}")

        try:
            executor = self._choose_executor()
            if executor in ("agent", "llm"):
                if executor == "llm" and not self.description:
                    raise ValueError("LLM tasks require a description template")
                elif executor == "agent":
                    # For agents, prefer string input for natural conversation
                    if self.description:
                        # Use description template with parameter substitution
                        prompt = self.format_description(self.description, data)
                    else:
                        # Pass string input naturally for direct agent conversation
                        prompt = self._format_natural_agent_input(payload, data)
                else:
                    # LLM with description
                    prompt = self.format_description(self.description, data)
                raw = await self._run_via_ai(prompt, executor)
            else:
                raw = await self._run_python(data)

            validated = await self._validate_output(raw)
            return validated

        except Exception:
            func_name = getattr(self.func, "__name__", "unknown_function")
            logger.exception(f"Error in task '{func_name}'")
            raise

    def _choose_executor(self) -> Literal["agent", "llm", "python"]:
        """
        Pick execution path.

        Returns:
            One of "agent", "llm", or "python".

        Raises:
            ValueError: If no valid executor is configured.
        """
        if self.agent:
            return "agent"
        if self.llm:
            return "llm"
        if self.func:
            return "python"
        raise ValueError("No execution path found for this task")

    async def _run_python(self, data: dict) -> Any:
        """
        Invoke the Python function directly.

        Args:
            data: Keyword arguments for the function.

        Returns:
            The function's return value.
        """
        logger.debug("Invoking regular Python function")
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**data)
        else:
            return self.func(**data)

    async def _run_via_ai(self, prompt: Any, executor: Literal["agent", "llm"]) -> Any:
        """
        Run the prompt through an Agent or LLM.

        Args:
            prompt: The prompt data - string for LLM, string/dict/Any for agent.
            executor: "agent" or "llm".

        Returns:
            Raw result from the AI path.
        """
        logger.debug(f"Invoking task via {executor.upper()}")
        logger.debug(f"Invoking task with prompt: {prompt!r}")
        if executor == "agent":
            # Agents can handle string, dict, or other input types
            result = await self.agent.run(prompt)
        else:
            # LLM expects a string prompt
            if not isinstance(prompt, str):
                raise ValueError(
                    f"LLM executor requires string prompt, got {type(prompt)}"
                )
            result = await self._invoke_llm(prompt)
        return self._convert_result(result)

    async def _invoke_llm(self, prompt: str) -> Any:
        """
        Build messages and call the LLM client.

        Args:
            prompt: The formatted prompt string.

        Returns:
            LLM-generated result.
        """
        # Gather history if needed
        history: List[BaseMessage] = []
        if self.include_chat_history and self.workflow_app:
            logger.debug("Retrieving chat history")
            history_dicts = self.workflow_app.get_chat_history()
            history = ChatPromptHelper.normalize_chat_messages(history_dicts)

        messages: List[BaseMessage] = history + [UserMessage(prompt)]
        params: Dict[str, Any] = {"messages": messages}

        # Add structured formatting if return type is a Pydantic model
        if (
            self.signature
            and self.signature.return_annotation is not inspect.Signature.empty
        ):
            model_cls = StructureHandler.resolve_response_model(
                self.signature.return_annotation
            )
            if model_cls:
                params["response_format"] = self.signature.return_annotation
                params["structured_mode"] = self.structured_mode

        logger.debug(f"LLM call params: {params}")
        return self.llm.generate(**params)

    def _normalize_input(self, raw_input: Any) -> dict:
        """
        Normalize various input types into a dict.

        Args:
            raw_input: Dataclass, SimpleNamespace, single value, or dict.

        Returns:
            A dict suitable for function invocation.

        Raises:
            ValueError: If signature is missing when wrapping a single value.
        """
        if is_dataclass(raw_input):
            return raw_input.__dict__
        if isinstance(raw_input, SimpleNamespace):
            return vars(raw_input)
        if not isinstance(raw_input, dict):
            # wrap single argument
            if not self.signature or len(self.signature.parameters) == 0:
                # No signature or no parameters - return empty dict for consistency
                return {}
            name = next(iter(self.signature.parameters))
            return {name: raw_input}
        return raw_input

    async def _validate_output(self, result: Any) -> Any:
        """
        Await and validate the result against return-type model.

        Args:
            result: Raw result from executor.

        Returns:
            Validated/transformed result.
        """
        if asyncio.iscoroutine(result):
            result = await result

        if (
            not self.signature
            or self.signature.return_annotation is inspect.Signature.empty
        ):
            return result

        return StructureHandler.validate_against_signature(
            result, self.signature.return_annotation
        )

    def _convert_result(self, result: Any) -> Any:
        """
        Unwrap AI return types into plain Python.

        Args:
            result: One of:
                - LLMChatResponse
                - BaseModel (Pydantic)
                - List[BaseModel]
                - primitive (str/int/etc) or dict

        Returns:
            • str (assistant content) when `LLMChatResponse`
            • dict when a single BaseModel
            • List[dict] when a list of BaseModels
            • otherwise, the raw `result`
        """
        # 1) Unwrap our unified LLMChatResponse → return the assistant's text
        if isinstance(result, LLMChatResponse):
            logger.debug("Extracted message content from LLMChatResponse.")
            msg = result.get_message()
            return getattr(msg, "content", None)

        # 2) Single Pydantic model → dict
        if isinstance(result, BaseModel):
            logger.debug("Converting Pydantic model to dictionary.")
            return result.model_dump()

        # 3) List of Pydantic models → list of dicts
        if isinstance(result, list) and all(isinstance(x, BaseModel) for x in result):
            logger.debug("Converting list of Pydantic models to list of dictionaries.")
            return [x.model_dump() for x in result]

        # 4) Fallback: primitive, dict, etc.
        logger.info("Returning final task result.")
        return result

    def format_description(self, template: str, data: dict) -> str:
        """
        Interpolate inputs into the prompt template.

        Args:
            template: The `{}`-style template string.
            data: Mapping of variable names to values.

        Returns:
            The fully formatted prompt.
        """
        if self.signature:
            bound = self.signature.bind(**data)
            bound.apply_defaults()
            return template.format(**bound.arguments)
        return template.format(**data)

    def _format_natural_agent_input(self, payload: Any, data: dict) -> str:
        """
        Format input for natural agent conversation.
        Favors string input over dictionary for better agent interaction.

        Args:
            payload: The original raw payload from the workflow
            data: The normalized dictionary version

        Returns:
            String input for natural agent conversation
        """
        if payload is None:
            return ""

        # If payload is already a simple string/number, use it directly
        if isinstance(payload, (str, int, float, bool)):
            return str(payload)

        # If we have function parameters, format them naturally
        if data and len(data) == 1:
            # Single parameter: extract the value
            value = next(iter(data.values()))
            return str(value) if value is not None else ""
        elif data:
            # Multiple parameters: format as natural text
            parts = []
            for key, value in data.items():
                if value is not None:
                    parts.append(f"{key}: {value}")
            return "\n".join(parts)
        else:
            # Fallback to string representation of payload
            return str(payload)


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
