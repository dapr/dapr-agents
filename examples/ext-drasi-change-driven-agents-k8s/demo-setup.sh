#!/usr/bin/env bash
#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -euo pipefail

BASE_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$BASE_DIR/../.." && pwd)
ENV_FILE="$PROJECT_DIR/.env"

install_with_retries() {
    local tool_name="$1"
    local install_cmd="$2"
    local uninstall_cmd="$3"
    local max_attempts="$4"
    
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "=== $tool_name installation attempt $attempt of $max_attempts... ==="
        
        if eval "$install_cmd"; then
            echo "=== $tool_name installed successfully! ==="
            return 0
        fi
        
        echo "=== $tool_name installation failed. ==="
        
        if [ $attempt -lt $max_attempts ]; then
            echo "=== Uninstalling $tool_name before retry... ==="
            eval "$uninstall_cmd" 2>/dev/null || true
            sleep 5
        else
            echo "=== ERROR: Failed to initialize $tool_name after $max_attempts attempts. ==="
            return 1
        fi
        
        attempt=$((attempt + 1))
    done
}

init_cluster() {
  echo "=== Creating cluster... ==="
  # Delete cluster if it already exists
  k3d cluster delete dapr-agents 2>/dev/null
  k3d cluster create dapr-agents \
    --port "8080:30080@server:0" \
    --port "5432:30432@server:0" \
    --k3s-arg "--disable=traefik@server:*" \
    --registry-create local-registry:0.0.0.0:5001
  echo "=== Cluster created successfully! ==="

  echo "=== Switching active context to the cluster... ==="
  kubectl config use-context k3d-dapr-agents

  echo "=== Waiting for cluster to be ready... ==="
  kubectl wait --for=condition=ready node --all --timeout=60s
}

install_dapr() {
  echo "=== Adding Dapr Helm chart repository... ==="
  helm repo add dapr https://dapr.github.io/helm-charts/
  helm repo update
  echo "=== Dapr Helm chart repository added successfully! ==="

  echo "=== Installing Dapr via Helm... ==="
  local install_cmd="helm upgrade \
    --install dapr dapr/dapr \
    --version=1.18.1 \
    --namespace dapr-system \
    --create-namespace \
    --set global.tag=1.18.1-mariner \
    --set daprd.logLevel=DEBUG \
    --wait"
  install_with_retries \
    "Dapr" \
    "$install_cmd" \
    "helm uninstall dapr -n dapr-system" \
    3
}

install_drasi() {
  echo "=== Configuring Drasi environment to use the active context... ==="
  if drasi env kube 2>/dev/null; then
    echo "=== Drasi environment configured successfully! ==="
  else
    echo "=== WARNING: Failed to configure Drasi environment to use the active context. ==="
    echo "=== Continuing with setup anyway... ==="
  fi

  echo "=== Installing Drasi... ==="
  install_with_retries \
    "Drasi" \
    "drasi init --version 0.10.0" \
    "drasi uninstall -y" \
    3
}

install_redis() {
  echo "=== Installing Redis via Helm... ==="
  local install_cmd="helm upgrade \
    --install redis oci://registry-1.docker.io/cloudpirates/redis \
    --version 0.30.7 \
    --set auth.enabled=false \
    --wait"
  install_with_retries \
    "Redis" \
    "$install_cmd" \
    "helm uninstall redis" \
    3
}

install_postgres() {
  echo "=== Installing Postgres via Helm... ==="
  local install_cmd="helm upgrade \
    --install postgres oci://registry-1.docker.io/cloudpirates/postgres \
    --version 0.19.6 \
    -f products/values.yaml \
    --wait"
  install_with_retries \
    "Postgres" \
    "$install_cmd" \
    "helm uninstall postgres" \
    3
}

build_push_images() {
  echo "=== Building images... ==="
  docker compose -f docker-compose.yaml build --no-cache
  echo "=== Images built successfully! ==="

  echo "=== Pushing images to local registry... ==="
  docker push localhost:5001/inventory-agent:latest
  echo "=== Images pushed successfully! ==="
}

create_secrets() {
  echo "=== Creating secrets... ==="

  if [ -z "$DAPR_CONVERSATION_API_KEY" ]; then
    echo "=== ERROR: DAPR_CONVERSATION_API_KEY environment variable not set. ==="
    echo "=== Please add it to your root .env file or set it before running this script: ==="
    echo "=== 'export DAPR_CONVERSATION_API_KEY=<YOUR_API_KEY>' ==="
    exit 1
  fi

  # Unset optional fields fall back to provider-specific defaults
  kubectl create secret generic dapr-conversation-secrets \
    --from-literal=key="$DAPR_CONVERSATION_API_KEY" \
    --from-literal=endpoint="${DAPR_CONVERSATION_ENDPOINT:-}" \
    --from-literal=model="${DAPR_CONVERSATION_MODEL:-}" \
    --from-literal=apiType="${DAPR_CONVERSATION_API_TYPE:-}" \
    --from-literal=apiVersion="${DAPR_CONVERSATION_API_VERSION:-}" \
    --dry-run=client -o yaml | kubectl apply -f -

  echo "=== Secrets created successfully! ==="
}

load_secrets() {
  echo "=== Loading secrets... ==="
  if [ -f "$ENV_FILE" ]; then
    # Optional — use .env file at the project root if present
    echo "=== .env file found at project root. ==="
    set -a
    source "$ENV_FILE"
    set +a
  else
    echo "=== .env file not found at project root. Using environment variables... ==="
  fi
}

deploy_components() {
  echo "=== Deploying Dapr components... ==="
  kubectl apply -f "${BASE_DIR}/drasi/components/"
  kubectl apply -f "${BASE_DIR}/inventory-agent/components/"
  echo "=== Dapr components deployed successfully! ==="
}

deploy_apps() {
  echo "=== Deploying apps... ==="
  kubectl apply -f "${BASE_DIR}/inventory-agent/inventory-agent.yaml"
  kubectl apply -f "${BASE_DIR}/workflow-dashboard/workflow-dashboard.yaml"
  echo "=== Apps deployed successfully! ==="
}

wait_for_workloads_ready() {
  echo "=== Waiting for all workloads to be ready... ==="
  kubectl rollout status deployment/inventory-agent --namespace default --timeout=300s
  kubectl rollout status deployment/workflow-dashboard --namespace default --timeout=300s
  kubectl wait --for=condition=Ready pod --all --namespace default --timeout=300s
  echo "=== All workloads ready! ==="
}

echo "=== Beginning setup... ==="

init_cluster

install_dapr
install_drasi
install_redis
install_postgres

build_push_images

load_secrets
create_secrets

deploy_components
deploy_apps
wait_for_workloads_ready

echo "=== Setup complete! ==="
echo "=== View the workflow dashboard at http://localhost:8080 ==="
