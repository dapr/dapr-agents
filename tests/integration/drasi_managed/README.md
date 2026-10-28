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

# Managed Drasi integration

Credential-free integration coverage for agent-managed Drasi subscriptions. This suite exercises `drasi_subscription_lifecycle()` with the actual built-in DaprAgentRouter, shared protocol package, Dapr service invocation, streaming Pub/Sub, persistent state, and workflow execution.

The model is a local scripted HTTP fixture, not an external LLM. It deterministically invokes the generated subscription-list tool and the agent's ordinary `record_event` tool through the normal durable chat/tool loop. The temporary receiver records actions for inspection; it is not the autonomous demonstration or a production idempotency service.

The suite injects packed, non-aggregation query-result fixtures through the router's real inbound Pub/Sub component. It does not install the Drasi query engine or validate Kubernetes installation. The PostgreSQL-to-query-engine Kubernetes example is separate work.

## Run

Requirements: Python supported by the workspace, uv, Git, and a running Docker engine with Compose/Buildx support for additional build contexts. No Dapr CLI, Kubernetes context, existing cluster, Azure deployment, or API key is needed.

From the repository root:

```sh
uv sync --group dev --group test --extra drasi
uv run pytest \
  --confcutdir=tests/integration/drasi_managed \
  tests/integration/drasi_managed \
  --run-drasi-managed \
  -o log_cli_level=WARNING
```

Run one scenario by appending `-k catalog_tools`, `-k interrupted`, or another pytest selector. The real-runtime scenarios carry the existing `integration` marker and skip without `--run-drasi-managed`.

**Run real-runtime scenarios in their own pytest process with `--confcutdir`.** Core `tests/conftest.py` replaces SDK modules globally, and the shared integration fixtures can initialize or reuse developer-wide Dapr resources. Neither is appropriate here. An explicitly requested run fails rather than skips if Docker, image builds, or a runtime dependency is unavailable.

`test_harness.py` contains offline regressions for partial startup, artifact capture, and the scripted model's current-turn behavior. These tests use controlled boundaries, require no Docker, and run in the normal core unit suite. Run them alone with:

```sh
uv run pytest \
  --confcutdir=tests/integration/drasi_managed \
  tests/integration/drasi_managed/test_harness.py
```

Images build once per pytest session. Each case then creates a fresh, randomly named Compose project, a private bridge network, temporary query files, and project-owned state volumes. Every published port binds to an ephemeral loopback port. The agent, router, and broker never adopt an existing container or application state.

Dapr's built-in `nameformat` resolver maps application IDs to private Compose DNS aliases and the sidecar's internal gRPC port. Service invocation still passes through both real sidecars; the fixture does not depend on multicast DNS support in the host's container network.

The pinned Dapr SDK waits for `/v1.0/healthz/outbound` during `DaprClient` construction, before the agent's metadata RPC. The lifecycle also checks `WorkflowRuntime.wait_for_worker_ready()` before opening the dynamic inbox or exposing application readiness.

Compose is invoked with an empty environment file and an allowlisted CLI/daemon environment, not a copy of the user's environment. The agent build has an explicit file allowlist that excludes `.env`, `.env.*`, virtual environments, and Git metadata. Azure/model credentials and build-version overrides are not inherited by the fixture.

## Reference pins and evidence

| Boundary | Selection |
|---|---|
| Dapr runtime, placement, scheduler | `1.18.4` |
| Redis Streams and Redis state stores | `7.4.2-alpine` |
| Built-in router and Reaction SDK | Drasi Platform revision `7e3f4f2aec6574ba1f487d910315e4265670ddc8` |
| Shared Python contract | PyPI `drasi-agent-router-contracts==0.1.0a1` |
| Agent SDK and dependencies | Workspace `uv.lock` resolved by `uv sync`, installed with `--frozen` in the image |
| Agent Dapr/workflow SDK | Bundled in `dapr==1.19.0rc4`; recorded as the `dapr` package version |
| Agent container Python / uv | `3.12.12` / `0.11.27` |

The router image is built directly from the Platform's own Dockerfile and Reaction SDK at the same pinned public Git revision. Docker/Buildx fetches both contexts; no local Platform checkout is needed. There is no copied router or local replacement protocol.

Before the image build, the fixture naturally builds a local core wheel with `uv build`. It reads the version from that wheel and passes that exact value into the restricted Docker context, which excludes Git metadata. Use a checkout with upstream release tags available so setuptools-scm can derive the real source version. The build fails dependency checks rather than inventing a release number. The session's `drasi-build*` directory retains the wheel and `source-build.json` with its version and SHA-256.

The agent image installs the checked-out core and extension as non-editable packages and runs `uv pip check`. Ordinary package imports, not a namespace bootstrap or test SDK mocks, are used at runtime. Source Git identity, dirty status, package versions, and actual image IDs identify the tested build.

Each case retains these files in its pytest temporary directory:

| File | Contents |
|---|---|
| `compose.env` | Non-secret project/image/query coordinates, written before startup for scoped recovery |
| `versions.json` | Actual runtime/Redis/package versions, Git revisions and dirty status, project name, and container image IDs |
| `containers.log` | Router, agent, sidecar, broker, placement, and scheduler logs |
| `artifacts.json` | Scheduling observations, model/tool evidence, and inbox/DLT CloudEvents |

Logs are also captured when startup fails; full artifacts are collected only after endpoint discovery, application readiness, and version recording all complete. Pytest reports its temporary directory with normal verbose output. To choose an artifact location, use a **new, dedicated** `--basetemp` directory: pytest replaces an existing directory supplied through that option.

## Scenarios

| Scenario | Observable boundary |
|---|---|
| Startup catalog | Real MCP through Dapr invocation; shared parsing and omitted optional fields |
| Router metadata validation | Built router rejects boolean, string, and floating-point sequence/timestamp values before publication or agent scheduling |
| Subscribe/update/unsubscribe | Generated tools, durable agent intent, and real router rules |
| Operation filtering | Router filters a mixed packed batch; admission also rejects a retained delivery excluded by a later filter |
| Router and agent restart | Durable rules/instructions survive; incarnation is retained; no startup model turn |
| Interrupted mutations | Stop the real router, observe persisted pending intent, kill the agent, then restart and reconcile subscribe/update/unsubscribe |
| Inactive/stale/unavailable input | Already-published deliveries are consumed without scheduling or model work |
| Retired-query cleanup | After catalog removal and restart, a generated unsubscribe-only tool clears saved intent whether its router rule still exists or was already removed; unrelated monitoring remains active |
| ACK boundary | Hold scheduling before acceptance and observe Redis pending state; then accept while the model remains blocked and observe broker acknowledgement before workflow completion |
| Frozen instructions | Updating a subscription while scheduling is held cannot rewrite an already admitted task |
| Intent failures | Invalid persisted intent and an actual store outage request retry, not success/absence |
| Scheduler failure | A real SDK client connects to a reserved, non-listening local port; restoring the normal client allows retry |
| Poison input and retry exhaustion | Original data reaches the configured DLT; poison never invokes the model |
| Active/terminal duplicates | Replay reaches the native scheduler and gets `ALREADY_EXISTS` while active; terminal ID reuse starts a new execution and repeats the ordinary action |
| Mode exclusion and tools | Both static/dynamic registration orders reject mixing; ordinary tools and all generated tools remain available in event workflows |
| Lifecycle ownership | A dedicated inbox Dapr client unblocks and joins streaming intake before workflow and router cleanup |
| In-flight shutdown | A real generated subscription tool drains before router close while inbox shutdown remains clean |

The only scheduling instrumentation is a `DaprWorkflowClient` subclass that records calls, optionally holds a call before acceptance, or selects the deliberately unreachable real client. Successful calls always delegate to the native scheduler. There are no fabricated workflow IDs, fake state clients, mocked router responses, or synthetic broker acknowledgements.

The scripted model considers tool calls only after the latest user message. Previous completed turns in a reused workflow ID's history cannot suppress the new task's scripted actions. Repeated receiver records demonstrate possible duplicate execution, not an idempotent business action.

For retirement/admission checks, a test changes only its temporary startup catalog or its own scoped intent document through Dapr's state API. CAS and strong reads remain enabled. Restart recovery uses fresh application processes, not reuse of a stopped registration.

## Bounded guarantees

The fixture configures three inbound retries at two-second intervals. Redis broker redelivery and SDK retries are separate layers; a configured retry count is not a universal bound on all possible attempts. The suite observes actual Redis consumer cursors and pending entries instead of treating `Subscription.respond()` as a durable broker receipt.

Poison is intentionally dropped with a configured dead-letter topic. Retriable failures can reach that same DLT after the selected runtime's retry budget is exhausted. DLT publication itself can fail; this suite does not promise unconditional redelivery, loss-free transport, exactly-once execution, permanent deduplication, or preservation across destruction of the reference environment.

Incarnations fence already-published inbox messages, not upstream query generations. Fresh query IDs remain necessary when recreating a query or changing its meaning. No migration or backward-compatibility layer is provided for this pre-release contract.

This is one logical agent, one router, and one application replica per isolated project. The state components use Redis and support ETags; this does not certify every Dapr state component or production broker configuration.

## Cleanup and ownership

Fixture teardown runs `docker compose down --volumes` for the exact generated project and verifies that no containers, networks, or volumes with that project's label remain. Session teardown removes only the two uniquely tagged images it built. Existing Docker images/build cache, clusters, Dapr installations, and unrelated resources are never pruned or uninstalled. Temporary evidence is retained for diagnosis.

A forcibly killed pytest process cannot run teardown. Copy the exact `COMPOSE_PROJECT_NAME` from that case's `compose.env` into `--project-name` below:

```sh
docker compose \
  --env-file /absolute/path/to/test-artifacts/compose.env \
  --project-name drasi-i2-<exact-project-id> \
  --file tests/integration/drasi_managed/compose.yaml \
  down --volumes --remove-orphans
```

Once all of that session's projects are down, abandoned test images can be removed with `docker image rm` using only the two exact image references from `compose.env`. Do not delete resources by a broad prefix or use `docker system prune`, `dapr uninstall`, or cluster-wide cleanup.

This directory owns its harness, fixtures, and local operating instructions. Shared CI and dependency resolution remain repository-level concerns; the Kubernetes reference example is not part of this suite.

Repository type checking uses its configured Python 3.11 target. Repeated mypy runs under a newer interpreter can install newer stub packages into the active environment, so synchronize the documented development environment before comparing type-check results. Applications run in pinned Python 3.12 containers.

## Prior art

The Compose topology, packed-change shape, and Redis acknowledgement observations follow the real-runtime tests in the [Platform router package](https://github.com/drasi-project/drasi-platform/tree/7e3f4f2aec6574ba1f487d910315e4265670ddc8/reactions/dapr/agent-router/tests). That Apache-2.0 implementation remains authoritative for router behavior; these tests add the composed Dapr Agents boundary.
