import inspect
import logging
from typing import Callable, Type, Optional, Any, Dict, TYPE_CHECKING
from inspect import signature, Parameter
from pydantic import BaseModel, Field, ValidationError, model_validator, PrivateAttr

from dapr_agents.tool.utils.tool import ToolHelper
from dapr_agents.tool.utils.function_calling import to_function_call_definition
from dapr_agents.types import ToolError

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool
    from mcp import ClientSession

logger = logging.getLogger(__name__)


class AgentTool(BaseModel):
    """
    Base class for agent tools, supporting both synchronous and asynchronous execution.
    This class can be used by the Agent and DurableAgent types to define tools that can be executed

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

    @classmethod
    def from_mcp(
        cls,
        mcp_tool: "MCPTool",
        session: "ClientSession" = None,
        connection: Any = None,
        process_result_fn=None,
    ) -> "AgentTool":
        """
        Create an AgentTool from an MCPTool and a session or connection.

        Args:
            mcp_tool: The MCPTool object to wrap.
            session: An active MCP ClientSession (preferred).
            connection: Optional connection config (if no session provided).
            process_result_fn: Optional function to process the tool result.

        Returns:
            AgentTool: A ready-to-use AgentTool instance.
        """
        if session is None and connection is None:
            raise ValueError("Either a session or a connection config must be provided")

        tool_name = mcp_tool.name
        tool_docs = mcp_tool.description or ""

        async def executor(**kwargs: Any) -> Any:
            try:
                logger.debug(f"Calling MCP tool '{tool_name}' with args: {kwargs}")
                if session is not None:
                    result = await session.call_tool(tool_name, kwargs)
                else:
                    logger.debug(f"Starting transport session for tool '{tool_name}'")
                    from dapr_agents.tool.mcp.transport import start_transport_session

                    async with start_transport_session(connection) as tool_session:
                        await tool_session.initialize()
                        result = await tool_session.call_tool(tool_name, kwargs)
                if process_result_fn:
                    return process_result_fn(result)
                # Default: extract text content
                if hasattr(result, "isError") and result.isError:
                    error_message = "Unknown error"
                    if hasattr(result, "content") and result.content:
                        for content in result.content:
                            if hasattr(content, "text"):
                                error_message = content.text
                                break
                    raise ToolError(f"MCP tool error: {error_message}")
                if hasattr(result, "content") and result.content:
                    text_contents = [
                        c.text for c in result.content if hasattr(c, "text")
                    ]
                    if len(text_contents) == 1:
                        return text_contents[0]
                    elif text_contents:
                        return text_contents
                return str(result)
            except Exception as e:
                raise ToolError(f"Error executing tool '{tool_name}': {str(e)}") from e

        # Optionally generate args model from input schema
        tool_args_model = None
        if getattr(mcp_tool, "inputSchema", None):
            try:
                from dapr_agents.tool.mcp.schema import (
                    create_pydantic_model_from_schema,
                )

                tool_args_model = create_pydantic_model_from_schema(
                    mcp_tool.inputSchema, f"{tool_name}Args"
                )
            except Exception as e:
                logger.warning(f"Failed to create schema for tool '{tool_name}': {e}")
                pass

        return cls(
            name=tool_name,
            description=tool_docs,
            func=executor,
            args_model=tool_args_model,
        )

    @classmethod
    def from_mcp_many(
        cls,
        mcp_tools: list,
        session: "ClientSession" = None,
        connection: Any = None,
        process_result_fn=None,
    ) -> list:
        """
        Batch-create AgentTool objects from a list of MCPTool objects.

        Args:
            mcp_tools (List[MCPTool]): List of MCP tool objects to convert.
            session: An active MCP ClientSession (preferred).
            connection: Optional connection config (if no session provided).
            process_result_fn: Optional function to process the tool result.

        Returns:
            List[AgentTool]: List of ready-to-use AgentTool objects.
        """
        return [
            cls.from_mcp(
                tool,
                session=session,
                connection=connection,
                process_result_fn=process_result_fn,
            )
            for tool in mcp_tools
        ]

    @classmethod
    async def from_mcp_session(
        cls, session: "ClientSession", process_result_fn=None
    ) -> list:
        """
        Fetch all tools and wrap them as AgentTool objects.

        Args:
            session: An active MCP ClientSession.
            process_result_fn: Optional function to process the tool result.

        Returns:
            List[AgentTool]: List of ready-to-use AgentTool objects.
        """
        mcp_tools_response = await session.list_tools()
        return cls.from_mcp_many(
            mcp_tools_response.tools,
            session=session,
            process_result_fn=process_result_fn,
        )

    def model_post_init(self, __context: Any) -> None:
        """
        Handles post-initialization logic for both class-based and function-based tools.
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
                return validated_args.model_dump(exclude_none=True)
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
