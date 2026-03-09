# demo-otel-k8s

Demonstrates four ways to configure OpenTelemetry observability for Dapr Agents on Kubernetes.

**Note:** The supplied scripts assume a bash terminal and won't work in a regular Windows environment.

## Agent Variants

| Pod | OTel Config Source | Hot-Reload? |
|---|---|---|
| `otel-instantiation` | Code (Python constructor) | No |
| `otel-envvars` | Environment variables | No |
| `otel-statestore` | Dapr state store (`agent-runtime`) | No |
| `otel-configstore` | Dapr Configuration Store (`otel-config`) | **Yes** |

## Prerequisites

- Kubernetes cluster with Dapr installed
- Redis deployed in the target namespace (Helm chart `bitnami/redis` or equivalent)
- A container registry accessible from the cluster (default: `localhost:5001`)

## Quick Start

```bash
# Deploy to a namespace (default: dapr-agents)
./build-and-deploy.sh my-namespace

# Or use the default namespace
./build-and-deploy.sh
```

## Configuration

### Namespace

All scripts accept the namespace as the first argument:

```bash
./build-and-deploy.sh <namespace>
./seed-runtime-store.sh <namespace>
```

Default: `dapr-agents`

### Redis Pod Discovery

The seed script auto-discovers the Redis pod using common label selectors:
1. `app.kubernetes.io/name=redis` (Bitnami Helm chart)
2. `app=dapr-redis` (Dapr quickstart)
3. Pod name matching `redis-master`

If auto-discovery fails, set the pod name explicitly:

```bash
REDIS_POD=my-redis-pod-0 ./seed-runtime-store.sh my-namespace
```

### Redis Host in Components

The Dapr component manifests use `dapr-redis-master:6379` as the Redis host. If your Redis service has a different name or is in a different namespace, edit the `redisHost` value in:
- `manifests/agent-runtime-component.yaml`
- `manifests/otel-configstore-component.yaml`

For cross-namespace access, use the FQDN: `<service>.<namespace>.svc.cluster.local:6379`

### Container Image

Override the image with:

```bash
DEMO_OTEL_IMAGE=my-registry/demo-otel-agent:v1 ./build-and-deploy.sh
```

## Hot-Reloading OTel Config

The `otel-configstore` agent subscribes to the Dapr Configuration Store for real-time OTel config updates.

### Supported Keys

| Key | Type | Description |
|---|---|---|
| `otel_sdk_disabled` | bool | `true` disables OTel entirely |
| `otel_exporter_otlp_endpoint` | string | Collector endpoint (e.g. `http://collector:4317`) |
| `otel_exporter_otlp_headers` | string | Auth token / headers for the exporter |
| `otel_service_name` | string | Service name for traces/logs |
| `otel_tracing_enabled` | bool | Enable/disable tracing |
| `otel_traces_exporter` | string | `otlp_grpc`, `otlp_http`, `zipkin`, `console` |
| `otel_logging_enabled` | bool | Enable/disable log export |
| `otel_logs_exporter` | string | `otlp_grpc`, `otlp_http`, `console` |

### Example: Change Endpoint at Runtime

```bash
NAMESPACE=dapr-agents
REDIS_POD=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=redis -o jsonpath='{.items[0].metadata.name}')

# Switch to a new collector
kubectl exec $REDIS_POD -n $NAMESPACE -- redis-cli SET otel_exporter_otlp_endpoint "http://new-collector:4317"

# Disable OTel
kubectl exec $REDIS_POD -n $NAMESPACE -- redis-cli SET otel_sdk_disabled "true"

# Re-enable with different exporter
kubectl exec $REDIS_POD -n $NAMESPACE -- redis-cli SET otel_sdk_disabled "false"
kubectl exec $REDIS_POD -n $NAMESPACE -- redis-cli SET otel_traces_exporter "otlp_http"
```

The agent logs will confirm the reload:
```
Agent otel-configstore: OTel providers reloaded via hot-reload
```

## Architecture

```
                    +-----------------------+
                    | Dapr Configuration    |
                    | Store (Redis)         |
                    |  otel_sdk_disabled    |
                    |  otel_*_endpoint ...  |
                    +-----------+-----------+
                                |
                        subscription
                                |
                    +-----------v-----------+
                    | otel-configstore      |
                    | DurableAgent          |
                    |  _config_handler()    |
                    |  _reload_observability|
                    +-----------+-----------+
                                |
                        new TracerProvider
                                |
                    +-----------v-----------+
                    | instrumentor          |
                    |  update_providers()   |
                    |  wrapper._tracer = x  |
                    +-----------------------+
```

The reload is zero-gap: new providers are created and wrappers updated before old providers are shut down. In-flight spans complete against their original tracer.
