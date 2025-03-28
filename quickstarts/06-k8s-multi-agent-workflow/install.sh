#!/bin/bash

unset DOCKER_DEFAULT_PLATFORM

BASE_DIR=$(dirname "$0")

# Create Registry
REG_NAME='dapr-registry'
REG_PORT='5001'
if [ "$(docker inspect -f '{{.State.Running}}' "${REG_NAME}" 2>/dev/null || true)" != 'true' ]; then
  docker run \
    -d --restart=always -p "127.0.0.1:${REG_PORT}:5000" --network bridge --name "${REG_NAME}" \
    registry:2
fi

# Create kind cluster with registry config
CLUSTER_NAME='dapr-agents'
cat <<EOF | kind create cluster --config=-
kind: Cluster
name: ${CLUSTER_NAME}
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  image: kindest/node:v1.32.3@sha256:b36e76b4ad37b88539ce5e07425f77b29f73a8eaaebf3f1a8bc9c764401d118c
- role: worker
  image: kindest/node:1.32.3@sha256:b36e76b4ad37b88539ce5e07425f77b29f73a8eaaebf3f1a8bc9c764401d118c
containerdConfigPatches:
- |-
  [plugins."io.containerd.grpc.v1.cri".registry]
    config_path = "/etc/containerd/certs.d"
EOF

# Add the registry to the nodes
REGISTRY_DIR="/etc/containerd/certs.d/localhost:${REG_PORT}"
for node in $(kind get nodes -n ${CLUSTER_NAME}); do
  docker exec "${node}" mkdir -p "${REGISTRY_DIR}"
  cat <<EOF | docker exec -i "${node}" cp /dev/stdin "${REGISTRY_DIR}/hosts.toml"
[host."http://${REG_NAME}:5000"]
EOF
done

# Connect the registry to the cluster network
if [ "$(docker inspect -f='{{json .NetworkSettings.Networks.kind}}' "${REG_NAME}")" = 'null' ]; then
  docker network connect "kind" "${REG_NAME}"
fi

# Document the local registry
# https://github.com/kubernetes/enhancements/tree/master/keps/sig-cluster-lifecycle/generic/1755-communicating-a-local-registry
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:${REG_PORT}"
    help: "https://kind.sigs.k8s.io/docs/user/local-registry/"
EOF

build_images () {
  echo "Building images... This takes a while..."
  docker-compose -f docker-compose.yaml build --no-cache
  echo "Images built!"
  echo "Pushing images to local registry... This takes a while..."
  docker push localhost:5001/dapr-client:latest 
  docker push localhost:5001/workflow-llm:latest
  docker push localhost:5001/elf:latest 
  docker push localhost:5001/hobbit:latest
  docker push localhost:5001/wizard:latest
  echo "Images pushed!"
}

echo "Installing Dapr..."
helm repo add dapr https://dapr.github.io/helm-charts/ &>/dev/null && \
  helm repo update &>/dev/null && \
  helm upgrade --install dapr dapr/dapr \
  --version=1.15 \
  --namespace dapr-system \
  --create-namespace \
  --set global.tag=1.15.2-mariner \
  --wait &>/dev/null
echo "Dapr installed!"

echo "Installing components..."
kubectl apply -f ../05-multi-agent-workflow-dapr-workflows/components/ &>/dev/null
echo "Components installed!"

build_images

echo "Installing Dapr agents..."
kubectl apply -f manifests/
echo "Dapr agents installed!"
