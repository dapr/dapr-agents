import asyncio
import inspect
import logging
from dataclasses import is_dataclass
from functools import update_wrapper
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from dapr.ext.workflow import WorkflowActivityContext

from dapr_agents.agents.base import AgentBase
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
            self.llm = OpenAIChatClient()

        if self.func:
            # Preserve name / docs for stack traces
            update_wrapper(self, self.func)

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
        logger.info(f"Executing task '{self.func.__name__}'")
        logger.debug(f"Executing task '{self.func.__name__}' with input {data!r}")

        try:
            executor = self._choose_executor()
            if executor in ("agent", "llm"):
                if not self.description:
                    raise ValueError("LLM/agent tasks require a description template")
                prompt = self.format_description(self.description, data)
                raw = await self._run_via_ai(prompt, executor)
            else:
                raw = await self._run_python(data)

            validated = await self._validate_output(raw)
            return validated

        except Exception:
            logger.exception(f"Error in task '{self.func.__name__}'")
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
        logger.info("Invoking regular Python function")
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**data)
        else:
            return self.func(**data)

    async def _run_via_ai(self, prompt: str, executor: Literal["agent", "llm"]) -> Any:
        """
        Run the prompt through an Agent or LLM.

        Args:
            prompt: The fully formatted prompt string.
            kind: "agent" or "llm".

        Returns:
            Raw result from the AI path.
        """
        logger.info(f"Invoking task via {executor.upper()}")
        logger.debug(f"Invoking task with prompt: {prompt!r}")
        if executor == "agent":
            result = await self.agent.run(prompt)
        else:
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
            history = self.workflow_app.get_chat_history()

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
            if not self.signature:
                raise ValueError("Cannot infer param name without signature")
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
            result: ChatCompletion, BaseModel, or list of BaseModel.

        Returns:
            A primitive, dict, or list of dicts.
        """
        # Unwrap ChatCompletion
        if isinstance(result, ChatCompletion):
            logger.debug("Extracted message content from ChatCompletion.")
            return result.get_content()
        # Pydantic → dict
        if isinstance(result, BaseModel):
            logger.debug("Converting Pydantic model to dictionary.")
            return result.model_dump()
        if isinstance(result, list) and all(isinstance(x, BaseModel) for x in result):
            logger.debug("Converting list of Pydantic models to list of dictionaries.")
            return [x.model_dump() for x in result]
        # If no specific conversion is necessary, return as-is
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
