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

"""
Remote Weather MCP Server (SSE transport)
==========================================

Simulates a third-party weather service hosted externally.  Exposes
``get_weather`` and ``get_forecast`` tools over SSE (Server-Sent Events).

In production this would be a remote endpoint managed by another team.
For this example it runs locally on port 8081 to demonstrate the SSE
transport type in the ``weather`` MCPServer resource.

Run::

    python weather_sse_server.py [--host 0.0.0.0] [--port 8081]
"""

import asyncio
import logging
import random

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("weather-sse-server")

mcp = FastMCP("RemoteWeatherService")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get current weather information for a location.

    Args:
        location: City or region name (e.g. 'Seattle', 'London').

    Returns:
        Current temperature and conditions.
    """
    temperature = random.randint(32, 105)
    conditions = random.choice(
        ["sunny", "cloudy", "partly cloudy", "rainy", "windy", "snowy", "foggy"]
    )
    humidity = random.randint(20, 95)
    return f"{location}: {temperature}F, {conditions}, {humidity}% humidity."


@mcp.tool()
async def get_forecast(location: str, days: int = 5) -> str:
    """Get a multi-day weather forecast for a location.

    Args:
        location: City or region name.
        days: Number of days to forecast (default 5, max 10).

    Returns:
        Multi-line forecast summary.
    """
    days = min(days, 10)
    lines = [f"{location} {days}-day forecast:"]
    for i in range(1, days + 1):
        high = random.randint(55, 105)
        low = high - random.randint(10, 25)
        cond = random.choice(
            ["sunny", "cloudy", "rainy", "stormy", "clear", "partly cloudy"]
        )
        lines.append(f"  Day {i}: High {high}F / Low {low}F, {cond}")
    return "\n".join(lines)


def main(host: str = "0.0.0.0", port: int = 8081) -> None:
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0], streams[1], mcp._mcp_server.create_initialization_options()
            )

    starlette_app = Starlette(
        debug=False,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    async def serve() -> None:
        config = uvicorn.Config(starlette_app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        logger.info("Weather SSE MCP server listening on http://%s:%d/sse", host, port)
        await server.serve()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remote weather MCP server (SSE transport)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
