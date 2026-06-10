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

from .client import MCPClient
from .dapr_workflow_client import mcp_tool_def_to_workflow_tool
from .prompt import convert_prompt_message
from .transport import (
    start_sse_session,
    start_stdio_session,
    start_streamable_http_session,
    start_transport_session,
    start_websocket_session,
)


def __getattr__(name):
    # Lazy re-export so the DeprecationWarning in .schema only fires when the
    # deprecated symbol is actually accessed, not on every package import.
    if name == "create_pydantic_model_from_schema":
        from .schema import create_pydantic_model_from_schema

        return create_pydantic_model_from_schema
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MCPClient",
    "mcp_tool_def_to_workflow_tool",
    "start_stdio_session",
    "start_sse_session",
    "start_streamable_http_session",
    "start_websocket_session",
    "start_transport_session",
    "convert_prompt_message",
    "create_pydantic_model_from_schema",
]
