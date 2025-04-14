from contextlib import AsyncExitStack, asynccontextmanager
from typing import Dict, List, Optional, Set, Any, Type, AsyncGenerator
from types import TracebackType
import asyncio
import logging

from pydantic import BaseModel, Field, PrivateAttr
from mcp import ClientSession
from mcp.types import Tool as MCPTool, Prompt

from dapr_agents.tool import AgentTool
from dapr_agents.types import ToolError


logger = logging.getLogger(__name__)


class MCPClient(BaseModel):
    """
    Client for connecting to MCP servers and integrating their tools with the Dapr agent framework.

    This client manages connections to one or more MCP servers, retrieves their tools,
    and converts them to native AgentTool objects that can be used in the agent framework.

    Attributes:
        allowed_tools: Optional set of tool names to include (when None, all tools are included)
        server_timeout: Timeout in seconds for server connections
        sse_read_timeout: Read timeout for SSE connections in seconds
    """

    allowed_tools: Optional[Set[str]] = Field(
        default=None,
        description="Optional set of tool names to include (when None, all tools are included)",
    )
    server_timeout: float = Field(
        default=5.0, description="Timeout in seconds for server connections"
    )
    sse_read_timeout: float = Field(
        default=300.0, description="Read timeout for SSE connections in seconds"
    )

    # Private attributes not exposed in model schema
    _exit_stack: AsyncExitStack = PrivateAttr(default_factory=AsyncExitStack)
    _sessions: Dict[str, ClientSession] = PrivateAttr(default_factory=dict)
    _server_tools: Dict[str, List[AgentTool]] = PrivateAttr(default_factory=dict)
    _server_prompts: Dict[str, Dict[str, Prompt]] = PrivateAttr(default_factory=dict)
    _task_locals: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _server_configs: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the client after the model is created."""
        logger.debug("Initializing MCP client")
        super().model_post_init(__context)

    @asynccontextmanager
    async def create_session(
        self, server_name: str
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Create an ephemeral session for the given server and yield it.
        Used during tool execution to avoid reuse issues.

        Args:
            server_name: The server to create a session for.

        Yields:
            A short-lived, initialized MCP session.
        """
        logger.debug(f"[MCP] Creating ephemeral session for server '{server_name}'")
        session = await self._create_ephemeral_session(server_name)
        try:
            yield session
        finally:
            # Session cleanup is managed by AsyncExitStack (via transport module)
            pass

    async def connect_stdio(
        self,
        server_name: str,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Connect to an MCP server using stdio transport and store the connection
        metadata for future dynamic reconnection if needed.

        Args:
            server_name (str): Unique identifier for this server connection.
            command (str): Executable to run.
            args (List[str]): Command-line arguments.
            env (Optional[Dict[str, str]]): Environment variables for the process.

        Raises:
            RuntimeError: If a server with the same name is already connected.
            Exception: If connection setup or initialization fails.
        """
        logger.info(
            f"Connecting to MCP server '{server_name}' via stdio: {command} {args}"
        )

        if server_name in self._sessions:
            raise RuntimeError(f"Server '{server_name}' is already connected")

        try:
            self._task_locals[server_name] = asyncio.current_task()

            from dapr_agents.tool.mcp.transport import (
                connect_stdio as transport_connect_stdio,
            )

            session = await transport_connect_stdio(
                command=command, args=args, env=env, stack=self._exit_stack
            )

            await session.initialize()
            self._sessions[server_name] = session

            # Store how to reconnect this server later
            self._server_configs[server_name] = {
                "type": "stdio",
                "params": {"command": command, "args": args, "env": env},
            }

            logger.debug(
                f"Initialized session for server '{server_name}', loading tools and prompts"
            )
            await self._load_tools_from_session(server_name, session)
            await self._load_prompts_from_session(server_name, session)
            logger.info(f"Successfully connected to MCP server '{server_name}'")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{server_name}': {str(e)}")
            self._sessions.pop(server_name, None)
            self._task_locals.pop(server_name, None)
            self._server_configs.pop(server_name, None)
            raise

    async def connect_sse(
        self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Connect to an MCP server using Server-Sent Events (SSE) transport and store
        the connection metadata for future dynamic reconnection if needed.

        Args:
            server_name (str): Unique identifier for this server connection.
            url (str): The SSE endpoint URL.
            headers (Optional[Dict[str, str]]): HTTP headers to include with the request.

        Raises:
            RuntimeError: If a server with the same name is already connected.
            Exception: If connection setup or initialization fails.
        """
        logger.info(f"Connecting to MCP server '{server_name}' via SSE: {url}")

        if server_name in self._sessions:
            raise RuntimeError(f"Server '{server_name}' is already connected")

        try:
            self._task_locals[server_name] = asyncio.current_task()

            from dapr_agents.tool.mcp.transport import (
                connect_sse as transport_connect_sse,
            )

            session = await transport_connect_sse(
                url=url,
                headers=headers,
                timeout=self.server_timeout,
                read_timeout=self.sse_read_timeout,
                stack=self._exit_stack,
            )

            await session.initialize()
            self._sessions[server_name] = session

            # Store how to reconnect this server later
            self._server_configs[server_name] = {
                "type": "sse",
                "params": {
                    "url": url,
                    "headers": headers,
                    "timeout": self.server_timeout,
                    "read_timeout": self.sse_read_timeout,
                },
            }

            logger.debug(
                f"Initialized session for server '{server_name}', loading tools and prompts"
            )
            await self._load_tools_from_session(server_name, session)
            await self._load_prompts_from_session(server_name, session)
            logger.info(f"Successfully connected to MCP server '{server_name}'")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{server_name}': {str(e)}")
            self._sessions.pop(server_name, None)
            self._task_locals.pop(server_name, None)
            self._server_configs.pop(server_name, None)
            raise

    async def _load_tools_from_session(
        self, server_name: str, session: ClientSession
    ) -> None:
        """
        Load tools from a given MCP session and convert them to AgentTools.

        Args:
            server_name: Unique identifier for this server
            session: The MCP client session
        """
        logger.debug(f"Loading tools from server '{server_name}'")

        try:
            # Get tools from the server
            tools_response = await session.list_tools()

            # Convert MCP tools to agent tools
            converted_tools = []
            for mcp_tool in tools_response.tools:
                # Skip tools not in allowed_tools if filtering is enabled
                if self.allowed_tools and mcp_tool.name not in self.allowed_tools:
                    logger.debug(
                        f"Skipping tool '{mcp_tool.name}' (not in allowed_tools)"
                    )
                    continue

                try:
                    agent_tool = await self.wrap_mcp_tool(server_name, mcp_tool)
                    converted_tools.append(agent_tool)
                except Exception as e:
                    logger.warning(
                        f"Failed to convert tool '{mcp_tool.name}': {str(e)}"
                    )

            self._server_tools[server_name] = converted_tools
            logger.info(
                f"Loaded {len(converted_tools)} tools from server '{server_name}'"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load tools from server '{server_name}': {str(e)}"
            )
            self._server_tools[server_name] = []

    async def _load_prompts_from_session(
        self, server_name: str, session: ClientSession
    ) -> None:
        """
        Load prompts from a given MCP session.

        Args:
            server_name: Unique identifier for this server
            session: The MCP client session
        """
        logger.debug(f"Loading prompts from server '{server_name}'")
        try:
            response = await session.list_prompts()
            prompt_dict = {prompt.name: prompt for prompt in response.prompts}
            self._server_prompts[server_name] = prompt_dict

            loaded = [
                f"{p.name} ({len(p.arguments or [])} args)" for p in response.prompts
            ]
            logger.info(
                f"Loaded {len(loaded)} prompts from server '{server_name}': "
                + ", ".join(loaded)
            )
        except Exception as e:
            logger.warning(
                f"Failed to load prompts from server '{server_name}': {str(e)}"
            )
            self._server_prompts[server_name] = []

    async def _create_ephemeral_session(self, server_name: str) -> ClientSession:
        """
        Create a fresh session for a single tool call.

        Args:
            server_name: The MCP server to connect to.

        Returns:
            A fully initialized ephemeral ClientSession.
        """
        config = self._server_configs.get(server_name)
        if not config:
            raise ToolError(f"No stored config found for server '{server_name}'")

        try:
            if config["type"] == "stdio":
                from dapr_agents.tool.mcp.transport import connect_stdio

                session = await connect_stdio(
                    **config["params"], stack=self._exit_stack
                )
            elif config["type"] == "sse":
                from dapr_agents.tool.mcp.transport import connect_sse

                session = await connect_sse(**config["params"], stack=self._exit_stack)
            else:
                raise ToolError(f"Unknown transport type: {config['type']}")

            await session.initialize()
            return session
        except Exception as e:
            logger.error(f"Failed to create ephemeral session: {e}")
            raise ToolError(f"Could not create session for '{server_name}': {e}") from e

    async def wrap_mcp_tool(self, server_name: str, mcp_tool: MCPTool) -> AgentTool:
        """
        Wrap an MCPTool as an AgentTool with dynamic session creation at runtime,
        based on stored server configuration.

        Args:
            server_name: The MCP server that registered the tool.
            mcp_tool: The MCPTool object describing the tool.

        Returns:
            An AgentTool instance that can be executed by the agent.

        Raises:
            ToolError: If the tool cannot be executed or configuration is missing.
        """
        tool_name = f"{server_name}_{mcp_tool.name}"
        tool_docs = f"{mcp_tool.description or ''} (from MCP server: {server_name})"

        logger.debug(f"Wrapping MCP tool: {tool_name}")

        def build_executor(client: MCPClient, server_name: str, tool_name: str):
            async def executor(**kwargs: Any) -> Any:
                """
                Execute the tool using a dynamically created session context.

                Args:
                    kwargs: Input arguments to the tool.

                Returns:
                    Result from the tool execution.

                Raises:
                    ToolError: If execution fails or response is malformed.
                """
                logger.info(f"[MCP] Executing tool '{tool_name}' with args: {kwargs}")
                try:
                    async with client.create_session(server_name) as session:
                        result = await session.call_tool(tool_name, kwargs)
                        logger.debug(f"[MCP] Received result from tool '{tool_name}'")
                        return client._process_tool_result(result)
                except Exception as e:
                    logger.exception(f"Execution failed for '{tool_name}'")
                    raise ToolError(
                        f"Error executing tool '{tool_name}': {str(e)}"
                    ) from e

            return executor

        # Build executor using dynamic context-managed session resolution
        tool_func = build_executor(self, server_name, mcp_tool.name)

        # Optionally generate args model from input schema
        args_model = None
        if getattr(mcp_tool, "inputSchema", None):
            try:
                from dapr_agents.tool.mcp.schema import (
                    create_pydantic_model_from_schema,
                )

                args_model = create_pydantic_model_from_schema(
                    mcp_tool.inputSchema, f"{tool_name}Args"
                )
                logger.debug(f"Generated argument model for tool '{tool_name}'")
            except Exception as e:
                logger.warning(
                    f"Failed to create schema for tool '{tool_name}': {str(e)}"
                )

        return AgentTool(
            name=tool_name,
            description=tool_docs,
            func=tool_func,
            args_model=args_model,
        )

    def _process_tool_result(self, result: Any) -> Any:
        """
        Process the result from an MCP tool call.

        Args:
            result: The result from calling an MCP tool

        Returns:
            Processed result in a format expected by AgentTool

        Raises:
            ToolError: If the result indicates an error
        """
        # Handle error result
        if hasattr(result, "isError") and result.isError:
            error_message = "Unknown error"
            if hasattr(result, "content") and result.content:
                for content in result.content:
                    if hasattr(content, "text"):
                        error_message = content.text
                        break
            raise ToolError(f"MCP tool error: {error_message}")

        # Extract text content from result
        if hasattr(result, "content") and result.content:
            text_contents = []
            for content in result.content:
                if hasattr(content, "text"):
                    text_contents.append(content.text)

            # Return single string if only one content item
            if len(text_contents) == 1:
                return text_contents[0]
            elif text_contents:
                return text_contents
        # Fallback for unexpected formats
        return str(result)

    def get_all_tools(self) -> List[AgentTool]:
        """
        Get all tools from all connected MCP servers.

        Returns:
            A list of all available AgentTools from MCP servers
        """
        all_tools = []
        for server_tools in self._server_tools.values():
            all_tools.extend(server_tools)
        return all_tools

    def get_server_tools(self, server_name: str) -> List[AgentTool]:
        """
        Get tools from a specific MCP server.

        Args:
            server_name: The name of the server to get tools from

        Returns:
            A list of AgentTools from the specified server
        """
        return self._server_tools.get(server_name, [])

    def get_server_prompts(self, server_name: str) -> List[Prompt]:
        """
        Get all prompt definitions from a specific MCP server.

        Args:
            server_name: The name of the server to retrieve prompts from

        Returns:
            A list of Prompt objects available on the specified server.
            Returns an empty list if no prompts are available.
        """
        return list(self._server_prompts.get(server_name, {}).values())

    def get_all_prompts(self) -> Dict[str, List[Prompt]]:
        """
        Get all prompt definitions from all connected MCP servers.

        Returns:
            A dictionary mapping each server name to a list of Prompt objects.
            Returns an empty dictionary if no servers are connected.
        """
        return {
            server: list(prompts.values())
            for server, prompts in self._server_prompts.items()
        }

    def get_prompt_names(self, server_name: str) -> List[str]:
        """
        Get the names of all prompts from a specific MCP server.

        Args:
            server_name: The name of the server

        Returns:
            A list of prompt names registered on the server.
        """
        return list(self._server_prompts.get(server_name, {}).keys())

    def get_all_prompt_names(self) -> Dict[str, List[str]]:
        """
        Get prompt names from all connected servers.

        Returns:
            A dictionary mapping server names to lists of prompt names.
        """
        return {
            server: list(prompts.keys())
            for server, prompts in self._server_prompts.items()
        }

    def get_prompt_metadata(
        self, server_name: str, prompt_name: str
    ) -> Optional[Prompt]:
        """
        Retrieve the full metadata for a given prompt from a connected MCP server.

        Args:
            server_name: The server that registered the prompt
            prompt_name: The name of the prompt to retrieve

        Returns:
            The full Prompt object if available, otherwise None.
        """
        return self._server_prompts.get(server_name, {}).get(prompt_name)

    def get_prompt_arguments(
        self, server_name: str, prompt_name: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get the list of arguments defined for a prompt, if available.

        Useful for generating forms or validating prompt input.

        Args:
            server_name: The server where the prompt is registered
            prompt_name: The name of the prompt to inspect

        Returns:
            A list of argument definitions, or None if the prompt is not found.
        """
        prompt = self.get_prompt_metadata(server_name, prompt_name)
        return prompt.arguments if prompt else None

    def describe_prompt(self, server_name: str, prompt_name: str) -> Optional[str]:
        """
        Retrieve a human-readable description of a specific prompt.

        Args:
            server_name: The name of the server where the prompt is registered
            prompt_name: The name of the prompt to describe

        Returns:
            The description string if available, otherwise None.
        """
        prompt = self.get_prompt_metadata(server_name, prompt_name)
        return prompt.description if prompt else None

    def get_connected_servers(self) -> List[str]:
        """
        Get a list of all connected server names.

        Returns:
            List of server names that are currently connected
        """
        return list(self._sessions.keys())

    async def close(self) -> None:
        """
        Close all connections to MCP servers and clean up resources.

        This method should be called when the client is no longer needed to
        ensure proper cleanup of all resources and connections.
        """
        logger.info("Closing MCP client and all server connections")

        # Verify we're in the same task as the one that created the connections
        current_task = asyncio.current_task()
        for server_name, server_task in self._task_locals.items():
            if server_task != current_task:
                logger.warning(
                    f"Attempting to close server '{server_name}' in a different task "
                    f"than it was created in. This may cause errors."
                )

        # Close all connections
        try:
            await self._exit_stack.aclose()
            self._sessions.clear()
            self._server_tools.clear()
            self._task_locals.clear()
            logger.info("MCP client successfully closed")
        except Exception as e:
            logger.error(f"Error closing MCP client: {str(e)}")
            raise

    async def __aenter__(self) -> "MCPClient":
        """Context manager entry point."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit - close all connections."""
        await self.close()


def create_sync_mcp_client(*args, **kwargs) -> MCPClient:
    """
    Create an MCPClient with synchronous wrapper methods for each async method.

    This allows the client to be used in synchronous code.

    Args:
        *args: Positional arguments for MCPClient constructor
        **kwargs: Keyword arguments for MCPClient constructor

    Returns:
        An MCPClient with additional sync_* methods
    """
    client = MCPClient(*args, **kwargs)

    # Add sync versions of async methods
    def create_sync_wrapper(async_func):
        def sync_wrapper(*args, **kwargs):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    raise RuntimeError(
                        f"Cannot call {async_func.__name__} synchronously in an async context. "
                        f"Use {async_func.__name__} directly instead."
                    )
            except RuntimeError:
                pass  # No event loop, which is fine for sync operation

            return asyncio.run(async_func(*args, **kwargs))

        # Copy metadata
        sync_wrapper.__name__ = f"sync_{async_func.__name__}"
        sync_wrapper.__doc__ = (
            f"Synchronous version of {async_func.__name__}.\n\n{async_func.__doc__}"
        )

        return sync_wrapper

    # Add sync wrappers for all async methods
    client.sync_connect_stdio = create_sync_wrapper(client.connect_stdio)
    client.sync_connect_sse = create_sync_wrapper(client.connect_sse)
    client.sync_close = create_sync_wrapper(client.close)

    return client
