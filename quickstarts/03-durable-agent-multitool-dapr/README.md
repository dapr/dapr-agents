# Durable Agent Tool Call with Dapr Conversation API

This quickstart mirrors `03-durable-agent-multitool-call/` but uses the Dapr Conversation API Alpha2 as the LLM provider with tool calling.
It tests the Durable Agent with multiple tools using the Dapr Conversation API Alpha2.

## Prerequisites
- Python 3.10+
- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)
- Dapr CLI [installed and initialized](https://docs.dapr.io/getting-started/install-dapr-cli/#step-1-install-the-dapr-cli)

## Setup using uv
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run
```bash
dapr run --app-id durablemultitoolapp \
  --resources-path ./components \
  -- source .venv/bin/activate && python multi_tool_agent_dapr.py
```

## Files
- `multi_tool_agent_dapr.py`: Durable agent using `llm_provider="dapr"`
- `multi_tools.py`: sample tools
- `components/`: Dapr components for LLM and state/pubsub

Notes:
- Alpha2 currently does not support streaming; this example is non-streaming.


