#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
NAMESPACE="${1:-dapr-agents}"
REDIS_POD="${REDIS_POD:-}"

# Auto-discover Redis pod if not explicitly set
if [ -z "$REDIS_POD" ]; then
  echo "=== Discovering Redis pod in namespace: $NAMESPACE ==="
  REDIS_POD=$(kubectl get pods -n "$NAMESPACE" \
    -l 'app.kubernetes.io/name=redis' \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

  # Fallback: try the dapr-redis label
  if [ -z "$REDIS_POD" ]; then
    REDIS_POD=$(kubectl get pods -n "$NAMESPACE" \
      -l 'app=dapr-redis' \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  fi

  # Fallback: try matching pod name pattern
  if [ -z "$REDIS_POD" ]; then
    REDIS_POD=$(kubectl get pods -n "$NAMESPACE" \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
      | grep -m1 'redis-master' || true)
  fi

  if [ -z "$REDIS_POD" ]; then
    echo "ERROR: Could not find a Redis pod in namespace '$NAMESPACE'."
    echo "Set REDIS_POD env var explicitly, e.g.:"
    echo "  REDIS_POD=my-redis-master-0 $0 $NAMESPACE"
    exit 1
  fi
fi

echo "=== Using Redis pod: $REDIS_POD ==="

# --- State store seed (otel-statestore agent) ---
echo "=== Seeding agent-runtime state store for otel-statestore agent ==="

# The key follows the pattern: <agent-name>||observability-config
# Value is JSON with OTEL configuration
kubectl exec "$REDIS_POD" -n "$NAMESPACE" -- redis-cli SET \
  "otel-statestore||observability-config" \
  '{"enabled":true,"tracing_enabled":true,"tracing_exporter":"otlp_grpc","endpoint":"http://dapr-agents-opentelemetry-collector.'"$NAMESPACE"'.svc.cluster.local:4317","service_name":"otel-statestore-agent"}'

echo "=== Verifying state store seed ==="
kubectl exec "$REDIS_POD" -n "$NAMESPACE" -- redis-cli GET \
  "otel-statestore||observability-config"

# --- Configuration store seed (otel-configstore agent) ---
echo ""
echo "=== Seeding OTel configuration store for otel-configstore agent ==="

# These keys use the standard OTEL env var names (lowercase).
# The Configuration Store subscription watches these keys for hot-reload.
kubectl exec "$REDIS_POD" -n "$NAMESPACE" -- redis-cli SET \
  "otel_sdk_disabled" "false"

kubectl exec "$REDIS_POD" -n "$NAMESPACE" -- redis-cli SET \
  "otel_exporter_otlp_endpoint" \
  "http://dapr-agents-opentelemetry-collector.${NAMESPACE}.svc.cluster.local:4317"

kubectl exec "$REDIS_POD" -n "$NAMESPACE" -- redis-cli SET \
  "otel_traces_exporter" "otlp_grpc"

kubectl exec "$REDIS_POD" -n "$NAMESPACE" -- redis-cli SET \
  "otel_service_name" "otel-configstore-agent"

kubectl exec "$REDIS_POD" -n "$NAMESPACE" -- redis-cli SET \
  "otel_tracing_enabled" "true"

echo "=== Verifying configuration store seed ==="
for key in otel_sdk_disabled otel_exporter_otlp_endpoint otel_traces_exporter otel_service_name otel_tracing_enabled; do
  echo -n "  $key = "
  kubectl exec "$REDIS_POD" -n "$NAMESPACE" -- redis-cli GET "$key"
done

echo ""
echo "=== Done ==="
echo ""
echo "To hot-reload OTel config at runtime, run:"
echo "  kubectl exec $REDIS_POD -n $NAMESPACE -- redis-cli SET otel_exporter_otlp_endpoint \"http://new-collector:4317\""
echo "  kubectl exec $REDIS_POD -n $NAMESPACE -- redis-cli SET otel_sdk_disabled \"true\"  # disable OTel"
