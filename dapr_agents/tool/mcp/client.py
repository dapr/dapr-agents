from contextlib import AsyncExitStack, asynccontextmanager
from typing import Dict, List, Optional, Set, Any, Type, AsyncIterator
from types import TracebackType
import asyncio
import logging

from pydantic import BaseModel, Field, PrivateAttr
from mcp import ClientSession
from mcp.types import Tool as MCPTool, Prompt

from dapr_agents.tool import AgentTool
from dapr_agents.types import ToolError
from dapr_agents.tool.mcp.transport import start_transport_session


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
        persistent_connections: If True, keep persistent connections to all MCP servers for both tool calls and metadata. This is the recommended and robust mode for all transports except stateless HTTP. If False, ephemeral (per-call) sessions are used. Ephemeral is only safe for stateless HTTP.
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
    persistent_connections: bool = Field(
        default=False,
        description="If True, keep persistent connections to all MCP servers for both tool calls and metadata. This is the recommended and robust mode for all transports except stateless HTTP. If False, ephemeral (per-call) sessions are used. Ephemeral is only safe for stateless HTTP.",
    )

    # Private attributes
    _exit_stack: AsyncExitStack = PrivateAttr(default_factory=AsyncExitStack)
    _sessions: Dict[str, ClientSession] = PrivateAttr(default_factory=dict)
    _server_tools: Dict[str, List[AgentTool]] = PrivateAttr(default_factory=dict)
    _server_prompts: Dict[str, Dict[str, Prompt]] = PrivateAttr(default_factory=dict)
    _task_locals: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _server_configs: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    @asynccontextmanager
    async def create_ephemeral_session(
        self, server_name: str
    ) -> AsyncIterator[ClientSession]:
        """
        Context manager: yield a fresh, initialized session for a single tool call.
        Ensures proper cleanup after use.

        Args:
            server_name: The server to create a session for.

        Yields:
            A short-lived, initialized MCP session.
        """
        config = self._server_configs.get(server_name)
        if not config:
            raise ToolError(f"No stored config found for server '{server_name}'")
        async with AsyncExitStack() as stack:
            transport = config["transport"]
            params = config["params"]
            try:
                session = await start_transport_session(transport, params, stack)
                await session.initialize()
                yield session
            except Exception as e:
                logger.error(f"Failed to create ephemeral session: {e}")
                raise ToolError(
                    f"Could not create session for '{server_name}': {e}"
                ) from e

    async def connect(self, config: dict) -> None:
        """
        Connect to an MCP server using the modular connection layer.

        Args:
            config: dict
        """
        # Make a copy so we don't mutate the caller's config
        config = dict(config)
        server_name = config.pop("server_name", None)
        transport = config.pop("transport", None)
        if server_name in self._sessions:
            raise RuntimeError(f"Server '{server_name}' is already connected")
        try:
            self._task_locals[server_name] = asyncio.current_task()
            stack = self._exit_stack
            if self.persistent_connections:
                # Persistent: session is managed by the main exit stack
                session = await start_transport_session(transport, config, stack)
                await session.initialize()
                self._server_configs[server_name] = {
                    "transport": transport,
                    "params": config,
                }
                logger.debug(
                    f"Initialized session for server '{server_name}', loading tools and prompts"
                )
                await self._load_tools_from_session(server_name, session)
                await self._load_prompts_from_session(server_name, session)
                self._sessions[server_name] = session
                logger.info(
                    f"Successfully connected to MCP server '{server_name}' (persistent mode)"
                )
            else:
                # Ephemeral: use a temporary AsyncExitStack for initial tool/prompt loading
                async with AsyncExitStack() as ephemeral_stack:
                    session = await start_transport_session(
                        transport, config, ephemeral_stack
                    )
                    await session.initialize()
                    self._server_configs[server_name] = {
                        "transport": transport,
                        "params": config,
                    }
                    logger.debug(
                        f"Initialized ephemeral session for server '{server_name}', loading tools and prompts"
                    )
                    await self._load_tools_from_session(server_name, session)
                    await self._load_prompts_from_session(server_name, session)
                logger.info(
                    f"Successfully connected to MCP server '{server_name}' (ephemeral mode)"
                )
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{server_name}': {str(e)}")
            self._sessions.pop(server_name, None)
            self._task_locals.pop(server_name, None)
            self._server_configs.pop(server_name, None)
            raise

    async def connect_many(self, server_configs: list) -> None:
        """
        Connect to multiple MCP servers of various types using the modular connection layer.

        Args:
            server_configs: List of dicts, each with keys:
                - type: "stdio", "sse", "http", "streamable_http", "websocket"
                - server_name: str
                - command/args/env (for stdio)
                - url/headers (for sse/http/streamable_http/websocket)
                - timeout, sse_read_timeout, terminate_on_close, session_kwargs, etc.
        """
        for config in server_configs:
            await self.connect(config)

    async def connect_from_config(self, config_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Connect to multiple MCP servers using a config dict mapping server names to config dicts.

        Args:
            config_dict: Dict mapping server names to config dicts.
        """
        for server_name, config in config_dict.items():
            config = dict(config)  # Copy to avoid mutating input
            config["server_name"] = server_name
            await self.connect(config)

    async def connect_sse(
        self,
        server_name: str,
        url: str,
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
        sse_read_timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Convenience method to connect to an MCP server using SSE transport.

        Args:
            server_name: Name to register the server under.
            url: URL of the SSE server endpoint.
            headers: Optional HTTP headers to send.
            timeout: Optional connection timeout in seconds.
            sse_read_timeout: Optional SSE read timeout in seconds.
            **kwargs: Additional transport-specific parameters.
        """
        config = {
            "server_name": server_name,
            "transport": "sse",
            "url": url,
        }
        if headers is not None:
            config["headers"] = headers
        if timeout is not None:
            config["timeout"] = timeout
        if sse_read_timeout is not None:
            config["sse_read_timeout"] = sse_read_timeout
        config.update(kwargs)
        await self.connect(config)

    async def connect_stdio(
        self,
        server_name: str,
        command: str,
        args: Optional[list] = None,
        env: Optional[dict] = None,
        cwd: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Convenience method to connect to an MCP server using stdio transport.

        Args:
            server_name: Name to register the server under.
            command: Command to launch the stdio server (e.g., 'python').
            args: Optional list of command-line arguments.
            env: Optional environment variables dict.
            cwd: Optional working directory.
            **kwargs: Additional transport-specific parameters.
        """
        config = {
            "server_name": server_name,
            "transport": "stdio",
            "command": command,
        }
        if args is not None:
            config["args"] = args
        if env is not None:
            config["env"] = env
        if cwd is not None:
            config["cwd"] = cwd
        config.update(kwargs)
        await self.connect(config)

    async def connect_streamable_http(
        self,
        server_name: str,
        url: str,
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
        sse_read_timeout: Optional[float] = None,
        terminate_on_close: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """
        Convenience method to connect to an MCP server using streamable HTTP transport.

        Args:
            server_name: Name to register the server under.
            url: URL of the streamable HTTP server endpoint.
            headers: Optional HTTP headers to send.
            timeout: Optional connection timeout in seconds.
            sse_read_timeout: Optional SSE read timeout in seconds.
            terminate_on_close: Optional flag to terminate session on close.
            **kwargs: Additional transport-specific parameters.
        """
        config = {
            "server_name": server_name,
            "transport": "streamable_http",
            "url": url,
        }
        if headers is not None:
            config["headers"] = headers
        if timeout is not None:
            config["timeout"] = timeout
        if sse_read_timeout is not None:
            config["sse_read_timeout"] = sse_read_timeout
        if terminate_on_close is not None:
            config["terminate_on_close"] = terminate_on_close
        config.update(kwargs)
        await self.connect(config)

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
                Execute the tool using either a persistent or ephemeral session context.

                Args:
                    kwargs: Input arguments to the tool.

                Returns:
                    Result from the tool execution.

                Raises:
                    ToolError: If execution fails or response is malformed.
                """
                logger.debug(f"Executing tool '{tool_name}' with args: {kwargs}")
                try:
                    if (
                        client.persistent_connections
                        and server_name in client._sessions
                    ):
                        session = client._sessions[server_name]
                        result = await session.call_tool(tool_name, kwargs)
                        logger.debug(f"Used persistent session for tool '{tool_name}'")
                    else:
                        async with client.create_ephemeral_session(
                            server_name
                        ) as session:
                            result = await session.call_tool(tool_name, kwargs)
                            logger.debug(
                                f"Used ephemeral session for tool '{tool_name}'"
                            )
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
        if not self.persistent_connections:
            return list(self._server_configs.keys())
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
