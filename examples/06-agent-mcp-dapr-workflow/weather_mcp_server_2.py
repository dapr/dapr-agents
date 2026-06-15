#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import asyncio
import logging
import random

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("weather-mcp-server-2")

mcp = FastMCP("WeatherServer2")


@mcp.tool()
async def get_humidity(location: str) -> str:
    """Get current relative humidity for a location."""
    humidity = random.randint(20, 95)
    return f"{location}: {humidity}% humidity."


@mcp.tool()
async def get_wind(location: str) -> str:
    """Get current wind conditions for a location."""
    speed = random.randint(0, 35)
    direction = random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    return f"{location}: {speed} mph from the {direction}."


def main(host: str = "0.0.0.0", port: int = 8082) -> None:
    session_manager = StreamableHTTPSessionManager(
        app=mcp._mcp_server,
        json_response=False,
        stateless=True,
    )

    async def handle_mcp(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    starlette_app = Starlette(
        debug=False,
        routes=[Mount("/mcp", app=handle_mcp)],
    )

    async def serve() -> None:
        async with session_manager.run():
            config = uvicorn.Config(
                starlette_app, host=host, port=port, log_level="info"
            )
            server = uvicorn.Server(config)
            logger.info(
                "Weather MCP server #2 listening on http://%s:%d/mcp", host, port
            )
            await server.serve()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Weather MCP server #2 (humidity + wind, streamable HTTP)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
