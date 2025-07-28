# Durable Agent Tool Call with Dapr Agents

This quickstart demonstrates how to create a **Durable Agent** with custom tools using Dapr Agents. You'll learn how to build a weather assistant that can fetch information and perform actions using defined tools through LLM-powered function calls, with stateful and durable execution.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key

## Environment Setup

```bash
# Create a virtual environment
python3.10 -m venv .venv

# Activate the virtual environment 
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file for your API keys:

```env
OPENAI_API_KEY=your_api_key_here
```
Replace `your_api_key_here` with your actual OpenAI API key.

2. Make sure Dapr is initialized on your system:

```bash
dapr init
```

3. The quickstart includes the necessary Dapr components in the `components` directory:

- `statestore.yaml`: Agent state configuration
- `pubsub.yaml`: Pub/Sub message bus configuration
- `workflowstate.yaml`: Workflow state configuration

## Example: DurableAgent Usage

```python
from dapr_agents import DurableAgent
from dotenv import load_dotenv
from weather_tools import tools
import asyncio
import logging

async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    weather_agent = DurableAgent(
        role="Weather Assistant",
        name="Stevie",
        goal="Help humans get weather and location info using smart tools.",
        instructions=[
            "Respond clearly and helpfully to weather-related questions.",
            "Use tools when appropriate to fetch weather data."
        ],
        message_bus_name="messagepubsub",
        state_store_name="workflowstatestore",
        state_key="workflow_state",
        agents_registry_store_name="agentstatestore",
        agents_registry_key="agents_registry",
        tools=tools,
    )
    await weather_agent.run("What's the weather in Boston tomorrow?")

if __name__ == "__main__":
    asyncio.run(main())
```

## Running the Example

start the agent with Dapr:

```bash
dapr run --app-id durableweatherapp --resources-path ./components -- python durable_weather_agent.py
```

## Other Durable Agent
You can also try the following Durable agents with the same tools using `HuggingFace hub` and `NVIDIA` LLM chat clients. Make sure you add the `HUGGINGFACE_API_KEY` and `NVIDIA_API_KEY` to the `.env` file.
- [HuggingFace Durable Agent](./durable_weather_agent_hf.py)
- [NVIDIA Durable Agent](./durable_weather_agent_nv.py)

## About Durable Agents

Durable agents maintain state across runs, enabling workflows that require persistence, recovery, and coordination. This is useful for long-running tasks, multi-step workflows, and agent collaboration.

## Custom Tools Example

See `weather_tools.py` for sample tool definitions.
