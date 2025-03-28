# Run Multi agent workflows in Kubernetes

## Headless run

To get going quickly, visit [the install script](./install.sh). The install script will ask for your password to use `sudo` for the following

- insert `127.0.0.1   registry.registry` into `/etc/hosts`
- store certificates into your keychain 

The script will

1. Install Kind with Calico CNI
1. Create certificates for the private registry
1. Create a private registry in the cluster
1. Install Dapr
1. Build the images for [05-multi-agent-workflow-dapr-workflows](../05-multi-agent-workflow-dapr-workflows/)
1. Push the images to local in-cluster registry
1. Install the [05-multi-agent-workflow-dapr-workflows components](../05-multi-agent-workflow-dapr-workflows/components/)
1. Install the [manifests for the agents](./manifests/)
