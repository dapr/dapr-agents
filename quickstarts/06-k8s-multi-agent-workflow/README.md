# Run Multi agent workflows in Kubernetes

This quickstart demonstrates how to create and orchestrate event-driven workflows with multiple autonomous agents using Dapr Agents running on Kubernetes.

## Prerequisites

- Python 3.10 (recommended)
- Pip package manager
- OpenAI API key
- Kind
- Docker
- Helm

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
1. Create the kubernetes secret form `.env` file
1. Deploy the [manifests for the agents](./manifests/)
1. Port forward the `workload-llm` pod on port `8004`
1. Trigger the workflow for getting to Morder by [k8s_http_client.py](./services/client/k8s_http_client.py)

### Install through manifests

First create a secret from your `.env` file:

```bash
kubectl create secret generic openai-secrets --from-env-file=.env --namespace default --dry-run=client -o yaml | kubectl apply -f -
```

Then build the images locally with `docker-compose`:

```bash
docker-compose -f docker-compose.yaml build --no-cache
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
