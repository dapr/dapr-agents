# Durable Agent Multi-Tool with Dapr LLM (Alpha2)

This quickstart demonstrates a Durable Agent that may call multiple tools (weather, calculator, web search) using Dapr Conversation API Alpha2 with tool-calling.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment for your Dapr sidecar and LLM component:
```bash
export DAPR_HTTP_PORT=3500
export DAPR_GRPC_PORT=56178
export DAPR_LLM_COMPONENT_DEFAULT=openai   # or google
```

Optionally set API keys for your chosen LLM component (e.g., OPENAI_API_KEY).

## Run
```bash
python multi_tool_agent_dapr.py
```

The agent will orchestrate tool calls to answer a multi-step query.


