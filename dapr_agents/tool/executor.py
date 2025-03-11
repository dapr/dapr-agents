from dapr_agents.types import AgentToolExecutorError, ToolError
from pydantic import BaseModel, Field, PrivateAttr
from dapr_agents.tool import AgentTool
from typing import Any, Dict, List
from rich.table import Table
from rich.console import Console
import logging

logger = logging.getLogger(__name__)

class AgentToolExecutor(BaseModel):
    """
    Manages the registration and execution of tools, providing efficient access and validation
    for tool instances by name.
    """

    tools: List[AgentTool] = Field(default_factory=list, description="List of tools to register and manage.")
    _tools_map: Dict[str, AgentTool] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """
        Registers each tool after model initialization, populating `_tools_map`.
        """
        if self.tools:  # Only register tools if the list is not empty
            for tool in self.tools:
                self.register_tool(tool)
            logger.info(f"Tool Executor initialized with {len(self.tools)} registered tools.")
        else:
            logger.info("Tool Executor initialized with no tools to register.")

        # Complete post-initialization
        super().model_post_init(__context)

    def register_tool(self, tool: AgentTool) -> None:
        """
        Registers a tool, ensuring no duplicate names.

        Args:
            tool (AgentTool): The tool instance to register.

        Raises:
            AgentToolExecutorError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools_map:
            logger.error(f"Attempt to register duplicate tool: {tool.name}")
            raise AgentToolExecutorError(f"Tool '{tool.name}' is already registered.")
        self._tools_map[tool.name] = tool
        logger.info(f"Tool registered: {tool.name}")
    
    def execute(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Executes a specified tool by name, passing any arguments.

        Args:
            tool_name (str): Name of the tool to execute.
            *args: Positional arguments for tool execution.
            **kwargs: Keyword arguments for tool execution.

        Returns:
            Any: Result from tool execution.

        Raises:
            AgentToolExecutorError: If tool not found or if an execution error occurs.
        """
        tool = self._tools_map.get(tool_name)
        if not tool:
            logger.error(f"Tool not found: {tool_name}")
            raise AgentToolExecutorError(f"Tool '{tool_name}' not found.")
        try:
            logger.info(f"Executing tool: {tool_name}")
            logger.debug(f"Tool Arguments: {kwargs}")
            result = tool(*args, **kwargs)
            logger.info(f"Tool '{tool_name}' executed successfully.")
            return result
        except ToolError as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            raise AgentToolExecutorError(f"Execution error in tool '{tool_name}': {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error executing tool '{tool_name}': {e}")
            raise AgentToolExecutorError(f"Unexpected execution error in tool '{tool_name}': {e}") from e

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
        return '\n'.join(tool.signature for tool in self._tools_map.values())

    def get_tool_details(self) -> str:
        """
        Retrieves names, descriptions, and argument schemas of all tools.

        Returns:
            str: Detailed tool information, each on a new line.
        """
        return '\n'.join(
            f"{tool.name}: {tool.description}. Args schema: {tool.args_schema}"
            for tool in self._tools_map.values()
        )

    @property
    def help(self) -> None:
        """
        Displays a tabular view of all registered tools with descriptions and signatures.
        """
        table = Table(title="Available Tools")
        table.add_column("Name", style="bold cyan")
        table.add_column("Description")
        table.add_column("Signature")

        for name, tool in self._tools_map.items():
            table.add_row(name, tool.description, tool.signature)

        console = Console()
        console.print(table)