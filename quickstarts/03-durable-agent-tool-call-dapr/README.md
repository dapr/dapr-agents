# Durable Agent Tool Call with Dapr LLM (Alpha2)

This quickstart mirrors `03-durable-agent-tool-call/` but uses the Dapr Conversation API Alpha2 as the LLM provider with tool calling.

## Prerequisites
- Python 3.10+
- Dapr CLI installed and initialized

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` with any provider-specific keys your chosen Dapr LLM component requires (e.g., OpenAI):
```env
OPENAI_API_KEY=your_api_key_here
```

Set the default Dapr LLM component name (matches a `components/*.yaml`):
```bash
export DAPR_LLM_COMPONENT_DEFAULT=openai
```

## Run
```bash
dapr run --app-id durableweatherapp \
  --resources-path ./components \
  -- python durable_weather_agent_dapr.py
```

## Files
- `durable_weather_agent_dapr.py`: Durable agent using `llm_provider="dapr"`
- `weather_tools.py`: sample tools
- `components/`: Dapr components for LLM and state/pubsub

Notes:
- Alpha2 currently does not support streaming; this example is non-streaming.
- Tool calling is enabled via Alpha2 `converse_alpha2` under the hood.

