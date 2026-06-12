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
logger = logging.getLogger("weather-mcp-server")

mcp = FastMCP("WeatherServer")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get current weather information for a location."""
    temperature = random.randint(55, 95)
    conditions = random.choice(["sunny", "cloudy", "partly cloudy", "rainy", "windy"])
    return f"{location}: {temperature}°F and {conditions}."


@mcp.tool()
async def get_forecast(location: str, days: int = 3) -> str:
    """Get a multi-day weather forecast for a location."""
    lines = [f"{location} {days}-day forecast:"]
    for i in range(1, days + 1):
        temp = random.randint(55, 95)
        cond = random.choice(["sunny", "cloudy", "rainy", "stormy", "clear"])
        lines.append(f"  Day {i}: {temp}°F, {cond}")
    return "\n".join(lines)


def main(host: str = "0.0.0.0", port: int = 8081) -> None:
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
            logger.info("Weather MCP server listening on http://%s:%d/mcp", host, port)
            await server.serve()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weather MCP server (streamable HTTP)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
