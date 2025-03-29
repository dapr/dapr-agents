# Run Multi agent workflows in Kubernetes

This quickstart demonstrates how to create and orchestrate event-driven workflows with multiple autonomous agents using Dapr Agents running on Kubernetes.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key
- Dapr CLI and Docker installed

## Configuration

1. Create a `.env` file for your API keys:

```env
OPENAI_API_KEY=your_api_key_here
```

## Install through script

The script will:

1. Install Kind with a local registry
1. Install Bitnami Redis
1. Install Dapr
1. Build the images for [05-multi-agent-workflow-dapr-workflows](../05-multi-agent-workflow-dapr-workflows/)
1. Push the images to local in-cluster registry
1. Install the [components for the agents](./components/)
1. Run with `dapr run -f dapr-llm.yaml -k`
