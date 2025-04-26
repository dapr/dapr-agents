import inspect
import logging
from typing import Callable, Type, Optional, Any, Dict
from inspect import signature, Parameter
from pydantic import BaseModel, Field, ValidationError, model_validator, PrivateAttr

from dapr_agents.tool.utils.tool import ToolHelper
from dapr_agents.tool.utils.function_calling import to_function_call_definition
from dapr_agents.types import ToolError

logger = logging.getLogger(__name__)


class AgentTool(BaseModel):
    """
    Base class for agent tools, supporting both synchronous and asynchronous execution.

    Attributes:
        name (str): The tool's name.
        description (str): A brief description of the tool's purpose.
        args_model (Optional[Type[BaseModel]]): Model for validating tool arguments.
        func (Optional[Callable]): Function defining tool behavior.
    """

    name: str = Field(
        ...,
        description="The name of the tool, formatted with capitalization and no spaces.",
    )
    description: str = Field(
        ..., description="A brief description of the tool's functionality."
    )
    args_model: Optional[Type[BaseModel]] = Field(
        None, description="Pydantic model for validating tool arguments."
    )
    func: Optional[Callable] = Field(
        None, description="Optional function implementing the tool's behavior."
    )

    _is_async: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def set_name_and_description(cls, values: dict) -> dict:
        """
        Validator to dynamically set `name` and `description` before validation.
        """
        func = values.get("func")
        if func:
            values.setdefault("name", func.__name__)
            values.setdefault("description", func.__doc__ or "")
        return values

    @classmethod
    def from_func(cls, func: Callable) -> "AgentTool":
        """
        Creates an instance of `AgentTool` from a raw Python function.

        Args:
            func (Callable): The function to wrap in the tool.

        Returns:
            AgentTool: An instance of `AgentTool`.
        """
        ToolHelper.check_docstring(func)
        return cls(func=func)

    def model_post_init(self, __context: Any) -> None:
        """
        Handles post-initialization logic for both class-based and function-based tools.
        Ensures `name` formatting and infers `args_model` if necessary.
        """
        self.name = self.name.replace(" ", "_").title().replace("_", "")

        if self.func:
            self._is_async = inspect.iscoroutinefunction(self.func)
            self._initialize_from_func(self.func)
        else:
            self._initialize_from_run()
        return super().model_post_init(__context)

    def _initialize_from_func(self, func: Callable) -> None:
        """Initialize Tool fields from a provided function."""
        if self.args_model is None:
            self.args_model = ToolHelper.infer_func_schema(func)

    def _initialize_from_run(self) -> None:
        """Initialize Tool fields based on the abstract `_run` method."""
        if self.args_model is None:
            self.args_model = ToolHelper.infer_func_schema(self._run)

    def _validate_and_prepare_args(
        self, func: Callable, *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Normalize and validate arguments for the given function.

        Args:
            func (Callable): The function whose signature is used.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dict[str, Any]: Validated and prepared arguments.

        Raises:
            ToolError: If argument validation fails.
        """
        sig = signature(func)
        if args:
            arg_names = list(sig.parameters.keys())
            kwargs.update(dict(zip(arg_names, args)))

        if self.args_model:
            try:
                validated_args = self.args_model(**kwargs)
                return validated_args.model_dump()
            except ValidationError as ve:
                logger.debug(f"Validation failed for tool '{self.name}': {ve}")
                raise ToolError(f"Validation error in tool '{self.name}': {ve}") from ve

        return kwargs

    def run(self, *args, **kwargs) -> Any:
        """
        Execute the tool synchronously.

        Raises:
            ToolError if the tool is async or execution fails.
        """
        if self._is_async:
            raise ToolError(
                f"Tool '{self.name}' is async and must be awaited. Use `await tool.arun(...)` instead."
            )
        try:
            func = self.func or self._run
            kwargs = self._validate_and_prepare_args(func, *args, **kwargs)
            return func(**kwargs)
        except Exception as e:
            self._log_and_raise_error(e)

    async def arun(self, *args, **kwargs) -> Any:
        """
        Execute the tool asynchronously (whether it's sync or async under the hood).
        """
        try:
            func = self.func or self._run
            kwargs = self._validate_and_prepare_args(func, *args, **kwargs)
            return await func(**kwargs) if self._is_async else func(**kwargs)
        except Exception as e:
            self._log_and_raise_error(e)

    def _run(self, *args, **kwargs) -> Any:
        """Fallback default run logic if no `func` is set."""
        if self.func:
            return self.func(*args, **kwargs)
        raise NotImplementedError("No function or _run method defined for this tool.")

    def _log_and_raise_error(self, error: Exception) -> None:
        """Log the error and raise a ToolError."""
        logger.error(f"Error executing tool '{self.name}': {str(error)}")
        raise ToolError(
            f"An error occurred during the execution of tool '{self.name}': {str(error)}"
        )

    def __call__(self, *args, **kwargs) -> Any:
        """
        Enables `tool(...)` syntax.

        Raises:
            ToolError: if async tool is called without `await`.
        """
        if self._is_async:
            raise ToolError(
                f"Tool '{self.name}' is async and must be awaited. Use `await tool.arun(...)`."
            )
        return self.run(*args, **kwargs)

    def to_function_call(
        self, format_type: str = "openai", use_deprecated: bool = False
    ) -> Dict:
        """
        Converts the tool to a specified function call format.

        Args:
            format_type (str): The format type (e.g., 'openai').
            use_deprecated (bool): Whether to use deprecated format.

        Returns:
            Dict: The function call representation.
        """
        return to_function_call_definition(
            self.name, self.description, self.args_model, format_type, use_deprecated
        )

    def __repr__(self) -> str:
        """Returns a string representation of the AgentTool."""
        return f"AgentTool(name={self.name}, description={self.description})"

    @property
    def args_schema(self) -> dict:
        """Returns a JSON-serializable dictionary of the tool's function args_model."""
        if self.args_model:
            schema = self.args_model.model_json_schema()
            for prop in schema.get("properties", {}).values():
                prop.pop("title", None)
            return schema.get("properties", {})
        return {}

    @property
    def signature(self) -> str:
        """Provides a dynamic and detailed string representation of the tool's function signature."""
        func_to_inspect = self.func if self.func else self._run
        params = signature(func_to_inspect).parameters
        args = [
            f"{name}: {param.annotation.__name__ if param.annotation != Parameter.empty else 'Any'}"
            f"{' = ' + repr(param.default) if param.default != Parameter.empty else ''}"
            for name, param in params.items()
        ]
        return f"{self.name}({', '.join(args)})"


def tool(
    func: Optional[Callable] = None, *, args_model: Optional[Type[BaseModel]] = None
) -> AgentTool:
    """
    A decorator to wrap a function with an `AgentTool` for validation and metadata.

    Args:
        func (Optional[Callable]): The function to wrap.
        args_model (Optional[Type[BaseModel]]): Optional Pydantic model for argument validation.

    Returns:
        AgentTool: The wrapped function as an `AgentTool`.
    """

    def decorator(f: Callable) -> AgentTool:
        ToolHelper.check_docstring(f)
        return AgentTool(func=f, args_model=args_model)

    return decorator(func) if func else decorator
