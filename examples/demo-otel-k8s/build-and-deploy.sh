#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
NAMESPACE="${1:-dapr-agents}"
IMAGE="${DEMO_OTEL_IMAGE:-localhost:5001/demo-otel-agent:latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Using namespace: $NAMESPACE ==="

echo "=== Building Docker image from local source ==="
docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"

echo "=== Pushing to local registry ==="
docker push "$IMAGE"

echo "=== Deploying agent-runtime component ==="
kubectl apply -f "$SCRIPT_DIR/manifests/agent-runtime-component.yaml" -n "$NAMESPACE"

echo "=== Deploying OTel configstore component ==="
kubectl apply -f "$SCRIPT_DIR/manifests/otel-configstore-component.yaml" -n "$NAMESPACE"

echo "=== Seeding runtime state store ==="
"$SCRIPT_DIR/seed-runtime-store.sh" "$NAMESPACE"

echo "=== Deploying all OTEL agent pods ==="
kubectl apply -f "$SCRIPT_DIR/manifests/" -n "$NAMESPACE"

echo "=== Waiting for pods to be ready ==="
kubectl rollout status deployment/otel-instantiation -n "$NAMESPACE" --timeout=120s
kubectl rollout status deployment/otel-envvars -n "$NAMESPACE" --timeout=120s
kubectl rollout status deployment/otel-statestore -n "$NAMESPACE" --timeout=120s
kubectl rollout status deployment/otel-configstore -n "$NAMESPACE" --timeout=120s

echo "=== All pods deployed ==="
kubectl get pods -n "$NAMESPACE" | grep otel-
