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

# MCPServer Auto-Discovery with a Durable Agent

This example shows a `DurableAgent` that automatically discovers and uses MCP
server tools through the Dapr sidecar — no manual `DaprMCPClient` setup required.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/) (`dapr init` once)
- A Dapr runtime built with `MCPServer` support
- OpenAI API key (or an Ollama endpoint configured in `resources/llm-provider.yaml`)

## Setup

```bash
uv venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv sync --active
```

Create a `.env` in this directory:

```env
OPENAI_API_KEY=your_openai_key_here
```

## Single MCP server

First, start the weather MCP server in a separate terminal. It listens at
`http://localhost:8081/mcp` and exposes `get_weather` and `get_forecast`:

```bash
python weather_mcp_server.py
```

A Dapr `MCPServer` resource named `weather` is provided at
`resources/weather-mcp.yaml` and is loaded automatically when you pass
`--resources-path ./resources`:

```bash
dapr run --app-id mcp-agent --resources-path ./resources -- python mcp_dapr_workflow.py
```

## Multiple MCP servers

`mcp_dapr_workflow_multi.py` runs one agent against two MCPServers, each exposing
distinct tools. Start each server in its own terminal:

```bash
python weather_mcp_server.py --port 8081     # MCPServer "weather":  get_weather, get_forecast
python weather_mcp_server_2.py --port 8082   # MCPServer "weather2": get_humidity, get_wind
```

Resources are at `resources/weather-mcp.yaml` (`weather` → `:8081`) and
`resources/weather-mcp-2.yaml` (`weather2` → `:8082`).

```bash
dapr run --app-id mcp-agent-multi --resources-path ./resources -- python mcp_dapr_workflow_multi.py
```

## Expected behavior

The agent answers a weather question by invoking the right tool for each
sub-question. In the multi-server case the LLM picks `get_weather`/`get_forecast`
from the `weather` server and `get_humidity`/`get_wind` from the `weather2`
server — tool names are unique across servers, so there's no ambiguity.

## How it works

1. At startup, the agent queries the Dapr sidecar metadata API and discovers all loaded `MCPServer` resources.
2. For each MCPServer, the agent calls the built-in `dapr.internal.mcp.<server>.ListTools` workflow and caches the tool schemas.
3. Each tool is a `WorkflowContextInjectedTool` that schedules `dapr.internal.mcp.<server>.CallTool.<tool>` as a child workflow when invoked.
4. The Dapr sidecar handles transport, retries, timeouts, and auth.

## Extending this example

To bypass auto-discovery and choose tools manually, drive `DaprMCPClient` yourself:

```python
from dapr.ext.workflow import DaprMCPClient
from dapr_agents.tool.mcp import mcp_tool_def_to_workflow_tool

client = DaprMCPClient(timeout_in_seconds=30)
client.connect("weather")
tools = [mcp_tool_def_to_workflow_tool(t) for t in client.get_all_tools()]
agent = DurableAgent(name="WeatherAgent", tools=tools, ...)
```

## Files

| File | Purpose |
|------|---------|
| `mcp_dapr_workflow.py` | Durable agent with single-server MCP auto-discovery |
| `mcp_dapr_workflow_multi.py` | Same, against two MCPServers |
| `weather_mcp_server.py` | MCP server: `get_weather`, `get_forecast` (default `:8081`) |
| `weather_mcp_server_2.py` | MCP server: `get_humidity`, `get_wind` (default `:8082`) |
| `mcp_tools.py` | Minimal stdio MCP server (standalone demo) |
| `resources/weather-mcp.yaml` | `MCPServer` component → `:8081` |
| `resources/weather-mcp-2.yaml` | `MCPServer` component → `:8082` |
