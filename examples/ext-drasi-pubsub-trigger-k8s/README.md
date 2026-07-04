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

# Drasi pub/sub-triggered agent

This example demonstrates how to build business event-driven agent workflows with Drasi and Dapr Agents on Kubernetes. With only a handful of lines of code, users can subscribe agents to Drasi queries and allow agents to take action when complex conditions in data sources (databases, streaming platforms) are satisfied.

## Why use Drasi?

Every system needs to interact with other sytems to be useful. Traditional polling may either result in unnecessary load on source systems, or stale data/unacceptable delays depending on the polling frequency. Custom change data capture (CDC) pipelines can power real-time, change-driven systems at scale, but can be expensive (if managed) or difficult to set up and maintain (if self-hosted).

For many use cases, these tradeoffs can be eliminated by using [Drasi](https://drasi.io/), which offers a platform centered around detecting and reacting to changes. The high-level architecture is simple:
- Sources to ingest data from existing systems
- Continuous Queries to track high-level changes ("business conditions") that downstream consumers are interested in
- Reactions to push events to downstream consumers

## How it works (with this example)

This example focuses on an inventory agent that automatically generates purchase orders. The agent should be triggered when it receives an event indicating that a product is "low" in stock (having stock less than some threshold), or "critical" (having no stock).

Data lives in a Postgres instance — when a product's stock level dips below its threshold (or reaches zero), it emits a low-level change event that is tracked by two Drasi queries: one for "low" and one for "critical". All of the low-level changes are transparent to downstream consumers, as queries **only** emit change events when their conditions are satisfied. The inventory agent does not have to do any additional processing.

## Prerequisites

- [Docker](https://docs.docker.com/get-started/get-docker/)
- [k3d](https://k3d.io/#installation)
- [kubectl](https://kubernetes.io/docs/reference/kubectl/)
- [Helm](https://helm.sh/docs/intro/install/)
- [Drasi CLI](https://drasi.io/drasi-kubernetes/reference/command-line-interface/#get-the-drasi-cli)
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key (Azure and direct OpenAI are supported)

## Setup

### Configure OpenAI credentials

NOTE: Azure OpenAI and direct OpenAI are the only two supported providers — the setup script fails if an API endpoint is given and it is not supported by one of the two providers.

**Option 1 — .env file**

Create a `.env` file in the project root with your OpenAI configuration.

For direct OpenAI, only the API key is required (the API endpoint is optional and defaults to `https://api.openai.com/v1`):

```properties
OPENAI_API_KEY=<YOUR_API_KEY>
```

For Azure OpenAI, the API key and API endpoint are both required:

```properties
OPENAI_API_KEY=<YOUR_API_KEY>
OPENAI_ENDPOINT=<YOUR_ENDPOINT>
```

The following configuration is optional:

```properties
OPENAI_MODEL=<YOUR_MODEL>                    # Default: "gpt-4.1-nano" for Azure, otherwise "gpt-4.1-nano-2025-04-14"
OPENAI_API_TYPE=<YOUR_API_TYPE>              # Default: "azure" for Azure, otherwise "openai"
OPENAI_API_VERSION=<YOUR_API_VERSION>        # Default: "2025-04-01-preview"
```

**Option 2 — environment variables**

Alternatively, you can export your OpenAI configuration as environment variables.
Required and optional variables are the same as in the first option.

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
export OPENAI_ENDPOINT=<YOUR_ENDPOINT>
export OPENAI_MODEL=<YOUR_MODEL>
export OPENAI_API_TYPE=<YOUR_API_TYPE>
export OPENAI_API_VERSION=<YOUR_API_VERSION>
```

### Create cluster

Ensure that the current working directory is the directory containing this `README`:

```bash
echo $(pwd)
```

Then, run the following initialization script:

```bash
./demo-setup.sh
```

This creates a `k3d`-managed cluster, installs Dapr and Drasi control plane services, and spins up several apps:
  - A Postgres instance with initialized with a default `productsdb` database, a `products` table, and seed data
  - An inventory agent configured to consume "low" and "critical" stock events from Drasi
  - A Diagrid Dashboard instance to observe agent workflow executions

Once the script completes, the dashboard and Postgres instance should be accessible from the host machine on ports `8080` and `5432`, respectively.

Before proceeding, ensure that the Diagrid Dashboard is accessible at http://localhost:8080. Notice that there are no workflow executions at this stage.

### Bring up Drasi resources

**Option 1 — Drasi CLI**

All of the Drasi resource manifests (sources, queries, reactions) are found in `./manifests/drasi/`.

First, bring up the sources and wait for them to be ready:

```bash
drasi apply -f ./manifests/drasi/sources/products.yaml
drasi wait -f ./manifests/drasi/sources/products.yaml -t 120
```

Bring up the queries and wait for them to be ready:

```bash
drasi apply -f ./manifests/drasi/queries/critical-stock-event-query.yaml
drasi wait -f ./manifests/drasi/queries/critical-stock-event-query.yaml -t 120

drasi apply -f ./manifests/drasi/queries/low-stock-event-query.yaml
drasi wait -f ./manifests/drasi/queries/low-stock-event-query.yaml -t 120
```

Bring up the reactions and wait for them to be ready:

```bash
drasi apply -f ./manifests/drasi/reactions/inventory-events-publisher.yaml
drasi wait -f ./manifests/drasi//reactions/inventory-events-publisher.yaml -t 120
```

You can verify that all of the resources are up with the following commands:

```bash
drasi list source
drasi list query
drasi list reaction
```

**Option 2 — Drasi VS Code Extension**

If you're using VS Code, you can manage Drasi resources interactively via the [Drasi VS Code extension](https://marketplace.visualstudio.com/items?itemName=DrasiProject.drasi).

For a guide on how to use the extension, see the [Drasi documentation](https://drasi.io/drasi-kubernetes/reference/vscode-extension/).

## View workflows

Once the Drasi resources are up and running, open http://localhost:8080 once again to view the Diagrid Dashboard.

At this point, you should be able to see several completed and/or in-flight workflow executions — the seed data in `./manifests/apps/products-db.yaml` contains several products that satisfy the "low" and "critical" stock conditions tracked by the Drasi queries. This causes Drasi to emit stock events, which are eventually consumed by the inventory agent.

This demonstrates the drop-in capabilities of Drasi — it can work with existing data sources, while downstream services can be developed independently.

## Experiment

### Trigger more workflows

Insert products directly into the `products` table and notice how workflow executions only occur for products that satisfy the "low" and "critical" stock conditions.

You can use the following parameters to connect to the Postgres instance:
- Host: `localhost`
- Port: `5432`
- Database: `productsdb`
- Username: `postgres`
- Password: `postgres`

### Adjust business conditions

Update the Drasi queries in `./manifests/drasi/queries/` by setting new thresholds for "low" and "critical" stock — no code changes necessary.

For the new queries to take effect, you must first delete the existing resources in the cluster. For example, to delete the resource for `./manifests/drasi/queries/low-stock-event.yaml`, run:

```bash
drasi delete -f ./manifests/drasi/queries/low-stock-event.yaml
```

Then, bring up the new query and wait for it to be ready:
```bash
drasi apply -f ./manifests/drasi/queries/low-stock-event-query.yaml
drasi wait -f ./manifests/drasi/queries/low-stock-event-query.yaml -t 120
```

With the Drasi VS Code extension, simply click the trash icon next to the query you want to delete, then apply the new query.

### Adapt for different use cases (advanced)

Modify the inventory agent to perform another task, write a new Drasi query, or point Drasi to different sources with different entities. You can use the existing files as a starting point.

Some useful documentation:
- [Drasi extension for Dapr Agents](../../ext/dapr-agents-ext-drasi/README.md)
- [Drasi query syntax](https://drasi.io/reference/query-language/)
- [Supported Drasi sources](https://drasi.io/drasi-kubernetes/how-to-guides/configure-sources/)

## Cleanup

Once you're done experimenting, run the cleanup script to delete the `k3d`-managed cluster:

```bash
./demo-cleanup.sh
```