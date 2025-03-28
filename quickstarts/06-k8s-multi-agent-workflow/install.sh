#!/bin/bash

export DOCKER_DEFAULT_PLATFORM=linux/arm64/v8

BASE_DIR=$(dirname "$0")

OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    echo "Detected macOS"
    export DOCKER_DEFAULT_PLATFORM="linux/arm64/v8"
elif [ "$OS" = "Linux" ]; then
    echo "Detected Linux"
    export DOCKER_DEFAULT_PLATFORM="linux/amd64"
    # Prompt for sudo access upfront, so we can add the certificate to the system trust store
    sudo ls / &> /dev/null
else
    echo "Unknown OS: $OS"
fi

build_images () {
  echo "Building images... This takes a while..."
  docker-compose -f docker-compose.yaml build --no-cache &> /dev/null
  echo "Images built!"
  echo "Pushing images to local registry... This takes a while..."
  docker push registry.registry:5500/dapr-client:latest 
  docker push registry.registry:5500/workflow-llm:latest
  docker push registry.registry:5500/elf:latest 
  docker push registry.registry:5500/hobbit:latest
  docker push registry.registry:5500/wizard:latest
  echo "Images pushed!"
}

echo "Creating kind cluster..."
kind create cluster --config ${BASE_DIR}/kind-cluster.yaml &>/dev/null
echo "Cluster is ready!"

# install network
echo "installing Calico..."
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.0/manifests/tigera-operator.yaml &> /dev/null
kubectl  create -f https://raw.githubusercontent.com/projectcalico/calico/v3.27.0/manifests/custom-resources.yaml &> /dev/null
echo "Calico installed!"

### Create local kind registry ####
mkdir -p ${BASE_DIR}/.tmp &> /dev/null
openssl genpkey -algorithm RSA -out ${BASE_DIR}/.tmp/registry.key &> /dev/null
openssl req -new -x509 -days 365 -keyout ${BASE_DIR}/.tmp/registry.key -out ${BASE_DIR}/.tmp/registry.crt -config ${BASE_DIR}/openssl.cnf -nodes &> /dev/null


kubectl create namespace registry &> /dev/null
kubectl create secret tls registry-tls \
  --cert=${BASE_DIR}/.tmp/registry.crt \
  --key=${BASE_DIR}/.tmp/registry.key -n registry -o yaml --dry-run=client | kubectl apply -f - &> /dev/null
kubectl apply -f ${BASE_DIR}/registry.yaml &> /dev/null

echo "Installing registry certificate to local system trust store"
if [ "$OS" = "Darwin" ]; then
    sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain ${BASE_DIR}/.tmp/registry.crt
elif [ "$OS" = "Linux" ]; then
    sudo cp ${BASE_DIR}/.tmp/registry.crt /usr/local/share/ca-certificates/registry-ca.crt &> /dev/null
    sudo update-ca-certificates &> /dev/null
else
    echo "Registry certificate is not installed due to unsupported OS: $OS"
fi

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