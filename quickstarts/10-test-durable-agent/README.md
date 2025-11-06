# Test New DurableAgent (Pub/Sub → Workflow)

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
# Terminal 1 – run the workflow app
dapr run \
  --app-id blog-app-agent \
  --resources-path ./components \
  -- python app.py

# Terminal 2 – publish a message to start the workflow
dapr run \
  --app-id blog-app-client \
  --resources-path ./components \
  -- python message_client.py
```

You should see the workflow started and the generated blog post appear in the app logs.