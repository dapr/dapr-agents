# Run Multi agent workflows in Kubernetes

## Headless run

The script will:

1. Install Kind with a local registry
1. Install Bitnami Redis
1. Install Dapr
1. Build the images for [05-multi-agent-workflow-dapr-workflows](../05-multi-agent-workflow-dapr-workflows/)
1. Push the images to local in-cluster registry
1. Install the [components for the agents](./components/)
1. Run with `dapr run -f dapr-llm.yaml -k`
