# Test New DurableAgents and Orchestrators

## Prerequisites

- Python 3.10+
- Dapr CLI & Docker
- OpenAI-compatible API key (for the `llm_activity` decorators)

## Setup

Install dependencies in your virtual environment (see repository root instructions) and ensure Dapr is initialised:

```bash
dapr init
```

Provide your OpenAI key via `.env` or by editing `components/openai.yaml`, identical to the earlier quickstart.

## Run the app

```bash
# Terminal 1 – run the multi-agent system
dapr run -f dapr-llm.yaml
```

You should see the workflow started and the generated blog post appear in the app logs.

## How it maps to the new host

- `RouteSpec` couples pub/sub metadata with the Pydantic message schema and the registered workflow name.
- Workflows/activities are registered once when constructing `DaprWorkflowApp`; the host handles playback-safe scheduling and lifecycle.
- The code under `workflow.py` is identical to quickstart 04—only the hosting logic changed.
