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

"""Minimal MCP server used by the MCPClient tests.

Run as python mcp_echo_server.py stdio or
python mcp_echo_server.py streamable-http <port>.
"""

import sys

from mcp.server.fastmcp import FastMCP


def build_server(port: int = 8000) -> FastMCP:
    server = FastMCP("EchoServer", host="127.0.0.1", port=port, log_level="WARNING")

    @server.tool()
    async def add(a: int, b: int) -> str:
        """Add two integers."""
        return f"sum={a + b}"

    return server


if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    build_server(port).run(transport)
