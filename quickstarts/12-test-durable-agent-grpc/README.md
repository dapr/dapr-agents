# DurableAgent with custom gRPC limits

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

This variant mirrors **10-test-durable-agent** but demonstrates how to raise the
workflow gRPC message size limits (defaults are ~4 MB). The app configures
`WorkflowGrpcOptions` with 32 MB send/receive ceilings before the workflow runtime
starts, which you should see logged when the service boots.

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

You should see the workflow start, log the gRPC override, and print the generated
blog post in the app logs.
