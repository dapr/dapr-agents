#!/bin/bash

unset DOCKER_DEFAULT_PLATFORM

BASE_DIR=$(dirname "$0")

# Create Registry
echo "### Creating local registry... ###"
REG_NAME='dapr-registry'
REG_PORT='5001'
if [ "$(docker inspect -f '{{.State.Running}}' "${REG_NAME}" 2>/dev/null || true)" != 'true' ]; then
  docker run \
    -d --restart=always -p "127.0.0.1:${REG_PORT}:5000" --network bridge --name "${REG_NAME}" \
    registry:2
fi
echo "### Local registry created! ###s"

# Create kind cluster with registry config
echo "### Creating kind cluster... ###"
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
echo "### Kind cluster created! ###"

# Add the registry to the nodes
echo "### Adding registry to nodes... ###"
REGISTRY_DIR="/etc/containerd/certs.d/localhost:${REG_PORT}"
for node in $(kind get nodes -n ${CLUSTER_NAME}); do
  docker exec "${node}" mkdir -p "${REGISTRY_DIR}"
  cat <<EOF | docker exec -i "${node}" cp /dev/stdin "${REGISTRY_DIR}/hosts.toml"
[host."http://${REG_NAME}:5000"]
EOF
done
echo "### Registry added to nodes! ###"

# Connect the registry to the cluster network
echo "### Connecting registry to cluster network... ###"
if [ "$(docker inspect -f='{{json .NetworkSettings.Networks.kind}}' "${REG_NAME}")" = 'null' ]; then
  docker network connect "kind" "${REG_NAME}"
fi
echo "### Registry connected to cluster network! ###"

# Document the local registry
# https://github.com/kubernetes/enhancements/tree/master/keps/sig-cluster-lifecycle/generic/1755-communicating-a-local-registry
echo "### Documenting local registry... ###"
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
echo "### Documenting local registry done! ###"

build_images () {
  echo "### Building images... This takes a while... ###"
  docker-compose -f docker-compose.yaml build --no-cache
  echo "### Images built! ###"
  echo "### Pushing images to local registry... This takes a while... ###"
  docker push localhost:5001/workflow-llm:latest
  docker push localhost:5001/elf:latest 
  docker push localhost:5001/hobbit:latest
  docker push localhost:5001/wizard:latest
  echo "#### Images pushed! ####"
}

echo "### Installing Bitnami Redis... ###"
helm install dapr-redis oci://registry-1.docker.io/bitnamicharts/redis \
  --wait &>/dev/null
echo "### Bitnami Redis installed! ###"

echo "### Installing Dapr... ####"
helm repo add dapr https://dapr.github.io/helm-charts/ &>/dev/null && \
  helm repo update &>/dev/null && \
  helm upgrade --install dapr dapr/dapr \
  --version=1.15 \
  --namespace dapr-system \
  --create-namespace \
  --set global.tag=1.15.2-mariner \
  --set daprd.logLevel=DEBUG \
  --wait &>/dev/null
echo "### Dapr installed! ###"

echo "### Installing components... ###"
kubectl apply -f "${BASE_DIR}/components/" &>/dev/null
echo "### Components installed! ###"

build_images

echo "### Creating Kubernetes secret from .env file... ###"
kubectl create secret generic openai-secrets \
  --from-env-file="${BASE_DIR}/.env" \
  --dry-run=client -o yaml | kubectl apply -f -
echo "### Kubernetes secret created! ###"

echo "### Creating manifests... ###"
kubectl apply -f "${BASE_DIR}/manifests/" &>/dev/null
echo "### Manifests created! ###"

echo "### Port forwarding the workflow-llm pod... ###"
while [[ $(kubectl get pods -l app=workflow-llm -o 'jsonpath={..status.conditions[?(@.type=="Ready")].status}') != "True" ]]; do
  sleep 1
done
kubectl port-forward -n default svc/workflow-llm 8004:80 &>/dev/null &
echo "### Port forwarded the workflow-llm pod... ###"

echo "### Trigger workflow... ###"
python3.10 -m pip install -r "${BASE_DIR}/services/client/requirements.txt" &>/dev/null
python3.10 "${BASE_DIR}/services/client/k8s_http_client.py"
