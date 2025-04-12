# MCP Integrated GitHub Agent

This quickstart demonstrates how to create and orchestrate event-driven workflows with multiple autonomous agents using Dapr Agents. You'll learn how to set up agents as services, implement workflow orchestration, and enable real-time agent collaboration through pub/sub messaging.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key
- Dapr CLI and Docker installed
- OpenAI API key
- GitHub PAT (with appropriate permissions to execute the prompt)
- Kind
- Docker
- Helm

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

Create a `.env` file for your API keys:

```env
OPENAI_API_KEY=your_api_key_here
GITHUB_PERSONAL_ACCESS_TOKEN=your_pat_here
```

The quickstart includes the necessary Dapr components in the `components` in the [06-k8s-multi-agent-workflow](../06-k8s-multi-agent-workflow/components) directory.

## Project Structure

```shell
services/                 # Directory for agent services
├── client/               # Client to trigger the workflow
│   └─ k8s_http_client.py # FastAPI app for hobbit
│   └─ requirements.txt   # Requirements file for the client
├── github/               # Agent's service
│   └─ app.py             # FastAPI app for github agent   
└── workflow-llm/         # LLM orchestrator
    └─ app.py             # Workflow service
doker-compose.yaml        # Compose file to build images
install.sh                # Easy install script to run the full demo
requirements.txt          # Requirements file for the agents
```

## Examples

## Install through script

The script will:

1. Install Kind with a local registry
1. Install Bitnami Redis
1. Install Dapr
1. Build the images for the GitHub agent including the [GitHub MCP Server](https://github.com/github/github-mcp-server/blob/main/Dockerfile#L25) binary
1. Push the images to local in-cluster registry
1. Install the [components for the agents](../06-k8s-multi-agent-workflow/components)
1. Create the kubernetes secret from `.env` file
1. Deploy the [manifests for the agents](./manifests/)
1. Port forward the `workload-llm` pod on port `8004`
1. Trigger the workflow for getting to Morder by [k8s_http_client.py](./services/client/k8s_http_client.py)

### Install through manifests

First create a secret from your `.env` file:

```bash
kubectl create secret generic mcp-secrets --from-env-file=".env" --dry-run=client -o yaml | kubectl apply -f -
```

Then build the images locally with `docker-compose`:

```bash
docker-compose -f docker-compose.yaml build --no-cache
```

Deploy the components:

```bash
kubectl apply -f ../06-k8s-multi-agent-workflow/components
```

Then deploy the manifests:

```bash
kubectl apply -f manifests/
```

Port forward the `workload-llm` pod:

```bash
kubectl port-forward -n default svc/workflow-llm 8004:80 &>/dev/null &
```

Trigger the client:

```bash
python3 services/client/k8s_http_client.py
```

## Next Steps

After completing this quickstart, you can:

- Add more agents to the workflow
- Switch to another workflow orchestration pattern (RoundRobin, LLM-based)
- Extend agents with custom tools
- Deploy agents and Dapr to a Kubernetes cluster. For more information on read [Deploy Dapr on a Kubernetes cluster](https://docs.dapr.io/operations/hosting/kubernetes/kubernetes-deploy)
- Check out the [Cookbooks](../../cookbook/)
