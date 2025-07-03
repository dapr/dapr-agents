import logging
from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field, PrivateAttr
from rich.table import Table
from rich.console import Console

from dapr_agents.tool.base import AgentTool
from dapr_agents.types import AgentToolExecutorError, ToolError

logger = logging.getLogger(__name__)


# The existing AgentBase class allows tools to be both Callable and AgentTool instances.
# Therefore, the AgentToolExecutor must support both types of tools as well.
class AgentToolExecutor(BaseModel):
    """
    Manages the registration and execution of tools, providing both sync and async interfaces.

    Attributes:
        tools (List[Union[AgentTool, Callable]]): List of tools to register and manage.
    """

    tools: List[Union[AgentTool, Callable]] = Field(
        default_factory=list, description="List of tools to register and manage."
    )
    _tools_map: Dict[str, AgentTool] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Initializes the internal tools map after model creation."""
        for tool in self.tools:
            self.register_tool(tool)
        logger.info(f"Tool Executor initialized with {len(self._tools_map)} tool(s).")
        super().model_post_init(__context)

    def register_tool(self, tool: Union[AgentTool, Callable]) -> None:
        """
        Registers a tool instance, ensuring no duplicate names.

        Args:
            tool (Union[AgentTool, Callable]): The tool to register.

        Raises:
            AgentToolExecutorError: If the tool name is already registered.
        """
        # Convert callable to AgentTool if needed since we support both Callable and AgentTool instances.
        if callable(tool) and not isinstance(tool, AgentTool):
            try:
                tool = AgentTool.from_func(tool)
                logger.info(f"Converted callable to AgentTool: {tool.name}")
            except Exception as e:
                logger.error(f"Failed to convert callable to AgentTool: {e}")
                raise AgentToolExecutorError(
                    f"Failed to convert callable to AgentTool: {e}"
                ) from e

        if isinstance(tool, AgentTool):
            if tool.name in self._tools_map:
                logger.error(f"Attempted to register duplicate tool: {tool.name}")
                raise AgentToolExecutorError(
                    f"Tool '{tool.name}' is already registered."
                )
            self._tools_map[tool.name] = tool
            logger.info(f"Tool registered: {tool.name}")
        else:
            raise TypeError(f"Unsupported tool type: {type(tool).__name__}")

    def get_tool(self, tool_name: str) -> Optional[AgentTool]:
        """
        Retrieves a tool by name.

        Args:
            tool_name (str): Name of the tool to retrieve.

        Returns:
            AgentTool or None if not found.
        """
        return self._tools_map.get(tool_name)

    def get_tool_names(self) -> List[str]:
        """
        Lists all registered tool names.

        Returns:
            List[str]: Names of all registered tools.
        """
        return list(self._tools_map.keys())

    def get_tool_signatures(self) -> str:
        """
        Retrieves the signatures of all registered tools.

        Returns:
            str: Tool signatures, each on a new line.
        """
        return "\n".join(tool.signature for tool in self._tools_map.values())

    def get_tool_details(self) -> str:
        """
        Retrieves names, descriptions, and argument schemas of all tools.

        Returns:
            str: Detailed tool information, each on a new line.
        """
        return "\n".join(
            f"{tool.name}: {tool.description}. Args schema: {tool.args_schema}"
            for tool in self._tools_map.values()
        )

    async def run_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Executes a tool by name, automatically handling both sync and async tools.

        Args:
            tool_name (str): Tool name to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Any: Result of tool execution.

        Raises:
            AgentToolExecutorError: If the tool is not found or execution fails.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            logger.error(f"Tool not found: {tool_name}")
            raise AgentToolExecutorError(f"Tool '{tool_name}' not found.")
        try:
            logger.info(f"Running tool (auto): {tool_name}")
            if tool._is_async:
                return await tool.arun(*args, **kwargs)
            return tool(*args, **kwargs)
        except ToolError as e:
            logger.error(f"Tool execution error in '{tool_name}': {e}")
            raise AgentToolExecutorError(str(e)) from e
        except Exception as e:
            logger.error(f"Unexpected error in '{tool_name}': {e}")
            raise AgentToolExecutorError(
                f"Unexpected error in tool '{tool_name}': {e}"
            ) from e

    @property
    def help(self) -> None:
        """Displays a rich-formatted table of registered tools."""
        table = Table(title="Available Tools")
        table.add_column("Name", style="bold cyan")
        table.add_column("Description")
        table.add_column("Signature")

        for name, tool in self._tools_map.items():
            table.add_row(name, tool.description, tool.signature)

        console = Console()
        console.print(table)
