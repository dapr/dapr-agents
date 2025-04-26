from contextlib import AsyncExitStack
from typing import Optional, Dict
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


async def connect_stdio(
    command: str, args: list[str], env: Optional[Dict[str, str]], stack: AsyncExitStack
) -> ClientSession:
    """
    Connect to an MCP server using stdio transport.

    Args:
        command: The executable to run
        args: Command line arguments
        env: Optional environment variables
        stack: AsyncExitStack for resource management

    Returns:
        An initialized MCP client session

    Raises:
        Exception: If connection fails
    """
    logger.debug(f"Establishing stdio connection: {command} {args}")

    # Create server parameters
    params = StdioServerParameters(command=command, args=args, env=env)

    # Establish connection via stdio
    try:
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(params)
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        logger.debug("Stdio connection established successfully")
        return session
    except Exception as e:
        logger.error(f"Failed to establish stdio connection: {str(e)}")
        raise


async def connect_sse(
    url: str,
    headers: Optional[Dict[str, str]],
    timeout: float,
    read_timeout: float,
    stack: AsyncExitStack,
) -> ClientSession:
    """
    Connect to an MCP server using Server-Sent Events (SSE) transport.

    Args:
        url: The SSE endpoint URL
        headers: Optional HTTP headers
        timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds
        stack: AsyncExitStack for resource management

    Returns:
        An initialized MCP client session

    Raises:
        Exception: If connection fails
    """
    logger.debug(f"Establishing SSE connection to: {url}")

    # Establish connection via SSE
    try:
        read_stream, write_stream = await stack.enter_async_context(
            sse_client(url, headers, timeout, read_timeout)
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        logger.debug("SSE connection established successfully")
        return session
    except Exception as e:
        logger.error(f"Failed to establish SSE connection: {str(e)}")
        raise
