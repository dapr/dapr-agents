<!--
Copyright 2026 The Dapr Authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# MCPServer Example

A `DurableAgent` that connects to a single `MCPServer` resource (a weather
service) and demonstrates Dapr's built-in middleware pipelines around MCP
tool calls.

## MCPServer Resource

| Resource | Transport | Tools |
|---|---|---|
| `weather` | `sse` | `get_weather`, `get_forecast` |

The MCPServer is defined in [`resources/weather.yaml`](resources/weather.yaml).
The agent doesn't connect to it directly — the sidecar handles transport,
auth, and retries, and the agent auto-discovers the tools via the metadata API.

## Middleware Hooks

The MCPServer spec wires three workflow-based middleware hooks around every
tool call:

| Workflow | Hook Type | Purpose |
|---|---|---|
| `rate_limit_workflow` | `beforeCallTool` | Per-tool rate limit (30/min) via a state-store counter |
| `input_validation_workflow` | `beforeCallTool` | Rejects arguments containing SQL injection / XSS patterns |
| `audit_log_workflow` | `afterCallTool` | Writes every tool invocation + result to the state store |

Implementations: [`middleware_workflows.py`](middleware_workflows.py).

## Prerequisites

- **Redis** running on `localhost:6379`
- **Dapr CLI** installed with `dapr init` completed
- **`OPENAI_API_KEY`** environment variable set
- **Python 3.11+** with `uv` (recommended) or `pip`

## Setup

```bash
# From this directory
uv sync

# Or with pip
pip install -e "../../[dev]"
```

## Run

### 1. Start the weather MCP server

```bash
python weather_sse_server.py
# Listening on http://0.0.0.0:8081/sse
```

### 2. Launch the agent with Dapr

```bash
cd examples/10-mcpserver
dapr run \
  --app-id mcp-agent \
  --resources-path ./resources \
  -- python agent.py
```

The sidecar will:
1. Load the `weather` MCPServer resource from `resources/`
2. Register `dapr.internal.mcp.weather.ListTools` and `dapr.internal.mcp.weather.CallTool.<tool>` workflows
3. Execute middleware hooks on every tool call

## File Overview

| File | Purpose |
|---|---|
| [`agent.py`](agent.py) | Main agent — auto-discovers the `weather` MCPServer and answers weather questions |
| [`weather_sse_server.py`](weather_sse_server.py) | SSE weather MCP server (run locally) |
| [`middleware_workflows.py`](middleware_workflows.py) | Rate limit, input validation, and audit log workflows |
| [`resources/weather.yaml`](resources/weather.yaml) | MCPServer resource (SSE transport) |

## Key Concepts

- **MCPServer resource**: A first-class Dapr resource that declares MCP
  server connection details. The sidecar handles transport, auth, and retries.
- **Auto-discovery**: `DurableAgent` queries the sidecar's metadata API for
  loaded MCPServer resources and registers their tools automatically — no
  manual `DaprMCPClient` wiring required.
- **Middleware hooks**: `beforeCallTool` / `afterCallTool` workflow pipelines
  defined on the MCPServer spec. "Before" hooks abort on error; "after" hooks
  log errors without affecting results.
