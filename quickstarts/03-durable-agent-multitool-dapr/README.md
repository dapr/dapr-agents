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

## Component Configuration
We provide a `components` folder with the Dapr components for the LLM and state/pubsub.
The Conversation component example uses [OpenAI](https://docs.dapr.io/reference/components-reference/supported-conversation/openai/) component. 
Many other LLM providers are compatible with OpenAI to certain extent (DeepSeek, Google AI, etc) so you can use them with the OpenAI component by configuring it with the appropriate parameters.
But Dapr also has [native support](https://docs.dapr.io/reference/components-reference/supported-conversation/) for other providers like Google AI, Anthropic, Mistral, DeepSeek, etc.

One thing you will need to update is the `key` in the [component configuration file](components/openai.yaml):

```yaml
metadata:
  - name: key
    value: "YOUR_OPENAI_API_KEY"
```

Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API (or other provider's API) key.

## Run
```bash
dapr run --app-id durablemultitoolapp \
  --resources-path ./components \
  -- python multi_tool_agent_dapr.py
```

## Files
- `multi_tool_agent_dapr.py`: Durable agent using `llm_provider="dapr"`
- `multi_tools.py`: sample tools
- `components/`: Dapr components for LLM and state/pubsub

Notes:
- Alpha2 currently does not support streaming; this example is non-streaming.


