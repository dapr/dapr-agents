<!--
Copyright 2026 The Dapr Authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Run Multi agent workflows in Kubernetes

This example demonstrates how to create and orchestrate event-driven workflows with multiple autonomous agents using Dapr Agents running on Kubernetes.

## Prerequisites

- uv package manager
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
1. Build the images for [04-multi-agent-workflows](../04-multi-agent-workflows/)
1. Push the images to local in-cluster registry
1. Install the [components for the agents](./resources/)
1. Create the kubernetes secret from `.env` file
1. Deploy the [manifests for the agents](./manifests/)
1. Port forward the `workflow-llm` pod on port `8004`
1. Trigger the workflow to get to Mordor via [k8s_http_client.py](./services/client/k8s_http_client.py)

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
