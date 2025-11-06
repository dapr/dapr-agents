import argparse
import logging
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route

from mcp.server.sse import SseServerTransport
from test_mcp_math_tools import mcp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp-server")


class MCPServer:
    """
    Unified MCPServer supporting multiple transports (stdio, SSE, streamable HTTP).
    Add new transports by implementing new methods.
    """

    def __init__(self):
        self.mcp = mcp

    def run_stdio(self):
        """Run the MCP server using stdio transport."""
        logger.info("Starting MCP server in STDIO mode")
        self.mcp.run("stdio")

    def run_sse(self, host="127.0.0.1", port=8000):
        """Run the MCP server using SSE transport."""
        logger.info("Starting MCP server in SSE mode")
        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request):
            logger.info("🔌 SSE connection established")
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                await self.mcp._mcp_server.run(
                    read_stream,
                    write_stream,
                    self.mcp._mcp_server.create_initialization_options(),
                )
            return Response(status_code=200)

        app = Starlette(
            debug=False,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )
        uvicorn.run(app, host=host, port=port)

    def run_streamable_http(
        self,
        host="127.0.0.1",
        port=8000,
        stateless=False,
        resumable=False,
        json_response=False,
    ):
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

        event_store = None
        if resumable:
            try:
                from event_store import InMemoryEventStore

                event_store = InMemoryEventStore()
                logger.warning("⚠️ Using in-memory event store. Not for production use!")
            except ImportError:
                logger.warning(
                    "Resumability requested but event_store module not found. Proceeding without resumability."
                )

        session_manager = StreamableHTTPSessionManager(
            self.mcp._mcp_server,
            event_store=event_store,
            json_response=json_response,
            stateless=stateless,
        )
        logger.info(
            f"Starting MCP server in streamable HTTP mode (stateless={stateless}, json_response={json_response}, resumable={resumable})"
        )

        async def handle_streamable_http(scope, receive, send):
            logger.info("🔌 Streamable HTTP connection established")
            await session_manager.handle_request(scope, receive, send)

        app = Starlette(
            debug=False,
            routes=[
                Mount("/mcp", app=handle_streamable_http),
            ],
        )

        async def serve():
            async with session_manager.run():
                config = uvicorn.Config(app, host=host, port=port, log_level="info")
                server = uvicorn.Server(config)
                await server.serve()

        import asyncio

        asyncio.run(serve())


# ─────────────────────────────────────────────
# CLI Argument Parsing
# ─────────────────────────────────────────────
def parse_args():
    """
    Parse CLI arguments for the MCP server.
    """
    parser = argparse.ArgumentParser(description="Run an MCP tool server.")
    parser.add_argument(
        "--server_type",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport to use",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (SSE/HTTP only)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (SSE/HTTP only)"
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        default=False,
        help="Enable stateless (ephemeral) HTTP mode",
    )
    parser.add_argument(
        "--json_response",
        action="store_true",
        default=False,
        help="Enable JSON response mode for HTTP",
    )
    parser.add_argument(
        "--resumable",
        action="store_true",
        default=False,
        help="Enable resumability (event store) for HTTP",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Server Startup Logic
# ─────────────────────────────────────────────
def start_server(args):
    """
    Start the MCP server using the selected transport.
    """
    server = MCPServer()
    try:
        if args.server_type == "stdio":
            server.run_stdio()
        elif args.server_type == "sse":
            server.run_sse(host=args.host, port=args.port)
        elif args.server_type == "streamable-http":
            server.run_streamable_http(
                host=args.host,
                port=args.port,
                stateless=args.stateless,
                resumable=args.resumable,
                json_response=args.json_response,
            )
        else:
            logger.error(f"Unknown server_type: {args.server_type}")
    except Exception as e:
        logger.exception(f"Server failed to start: {e}")


# ─────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────
def main():
    """
    CLI entrypoint for the MCP server.
    """
    args = parse_args()
    start_server(args)


if __name__ == "__main__":
    main()
