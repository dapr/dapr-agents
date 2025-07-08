from contextlib import AsyncExitStack
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.websocket import websocket_client

from dapr_agents.types.tools import (
    SseServerParameters,
    StreamableHTTPServerParameters,
    WebSocketServerParameters,
)

logger = logging.getLogger(__name__)


async def start_stdio_session(
    params: dict, stack: AsyncExitStack = None
) -> ClientSession:
    """
    Start a session with MCP server using stdio transport.

    Args:
        params: Dict with keys 'command', 'args', 'env', 'cwd'
        stack: AsyncExitStack for resource management

    Returns:
        An initialized MCP client session

    Raises:
        Exception: If connection fails
    """
    server_params = StdioServerParameters(**params)
    try:
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        logger.debug("Stdio connection established successfully")
        return session
    except Exception as e:
        logger.error(f"Failed to establish stdio connection: {str(e)}")
        raise


async def start_sse_session(
    params: dict, stack: AsyncExitStack = None
) -> ClientSession:
    """
    Start a session with MCP server using Server-Sent Events (SSE) transport.

    Args:
        params: Dict with keys 'url', 'headers', 'timeout', 'sse_read_timeout'
        stack: AsyncExitStack for resource management

    Returns:
        An initialized MCP client session

    Raises:
        Exception: If connection fails
    """
    server_params = SseServerParameters(**params)
    try:
        read_stream, write_stream = await stack.enter_async_context(
            sse_client(
                url=server_params.url,
                headers=server_params.headers,
                timeout=server_params.timeout,
                sse_read_timeout=server_params.sse_read_timeout,
            )
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        logger.debug("SSE connection established successfully")
        return session
    except Exception as e:
        logger.error(f"Failed to establish SSE connection: {str(e)}")
        raise


async def start_streamable_http_session(
    params: dict, stack: AsyncExitStack = None
) -> ClientSession:
    """
    Start a session with MCP server using streamable HTTP transport.

    Args:
        params: Dict with keys 'url', 'headers', 'timeout', 'sse_read_timeout', 'terminate_on_close'
        stack: AsyncExitStack for resource management

    Returns:
        An initialized MCP client session

    Raises:
        Exception: If connection fails
    """
    server_params = StreamableHTTPServerParameters(**params)
    try:
        read_stream, write_stream, _ = await stack.enter_async_context(
            streamablehttp_client(
                url=server_params.url,
                headers=server_params.headers,
                timeout=server_params.timeout,
                sse_read_timeout=server_params.sse_read_timeout,
                terminate_on_close=server_params.terminate_on_close,
            )
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        logger.debug("streamable_http connection established successfully")
        return session
    except Exception as e:
        logger.error(f"Failed to establish streamable_http connection: {str(e)}")
        raise


async def start_websocket_session(
    params: dict, stack: AsyncExitStack = None
) -> ClientSession:
    """
    Start a session with MCP server using websocket transport.

    Args:
        params: Dict with key 'url'
        stack: AsyncExitStack for resource management

    Returns:
        An initialized MCP client session

    Raises:
        Exception: If connection fails
    """
    server_params = WebSocketServerParameters(**params)
    try:
        read_stream, write_stream = await stack.enter_async_context(
            websocket_client(server_params.url)
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        logger.debug("Websocket connection established successfully")
        return session
    except Exception as e:
        logger.error(f"Failed to establish websocket connection: {str(e)}")
        raise


async def start_transport_session(
    transport: str, params: dict, stack: AsyncExitStack = None
) -> ClientSession:
    """
    Unified entry point to start a session with an MCP server using any supported transport.
    Dispatches to the correct transport session helper based on the transport type and params.

    Args:
        transport: The transport type (e.g., 'stdio', 'sse', 'streamable_http', 'websocket').
        params: Dict of transport-specific parameters.
        stack: AsyncExitStack for resource management.

    Returns:
        An initialized MCP client session.

    Raises:
        ValueError: If the transport type is unsupported or params are invalid.
    """
    if not transport:
        raise ValueError("Missing 'transport' argument")
    if transport == "stdio":
        return await start_stdio_session(params, stack=stack)
    elif transport == "sse":
        return await start_sse_session(params, stack=stack)
    elif transport == "streamable_http":
        return await start_streamable_http_session(params, stack=stack)
    elif transport == "websocket":
        return await start_websocket_session(params, stack=stack)
    else:
        raise ValueError(f"Unsupported transport: {transport}")
