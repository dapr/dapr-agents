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

# dapr-agents-ext-drasi

[Drasi](https://drasi.io/) query-result changes can start independent Dapr Agent workflows through this extension.

## Features

- Register author-configured Drasi query triggers through Dapr Pub/Sub.
- Let an agent select persistent subscriptions from an operator-curated router catalog.

## Getting Started

### Availability

The APIs documented here describe this source checkout. The published `dapr-agents-ext-drasi==0.1.0` package predates this work: it exposes `drasi_trigger()` and does not include `register_drasi_trigger()` or `drasi_subscription_lifecycle()`.

The next extension release version has not been selected; the unchanged `0.1.0` source version is not a publication target. This layer declares `dapr-agents>=1.0.7,<2.0.0`: core `1.0.6` loses explicit tool-choice settings when the lifecycle adds tools to an initially tool-less agent. Recheck the minimum against the final assembled contribution before publication. No compatibility alias or migration layer is provided for these pre-release API changes.

To use the current APIs, install this checkout from the repository root:

```bash
uv sync --group dev --group test --extra drasi
```

Agent-managed subscriptions use [`drasi-agent-router-contracts==0.1.0a1`](https://pypi.org/project/drasi-agent-router-contracts/0.1.0a1/), generated from the Drasi Platform TypeSpec source. A local Platform checkout and schema-generation tools are not required. This dependency is not installed by the core framework unless the Drasi extension is selected.

### Public API

```python
from dapr_agents.ext.drasi import (
    register_drasi_trigger,  # Register author-configured Drasi query triggers
    drasi_subscription_lifecycle,  # Host agent-managed persistent subscriptions
    DrasiSubscriptionLifecycle,
    DrasiLifecycleError,
    DrasiChangeEvent,  # Type for Drasi change events
    DrasiOperation,  # Operation type for Drasi change events
)
```

The modes are alternatives on one logical agent:

| Mode | Owner of the subscription | Public API | Use when |
|---|---|---|---|
| Author-configured | Application author or deployment | `register_drasi_trigger()` | Query IDs and task mapping are fixed before hosting. |
| Agent-managed | Agent using generated tools | `drasi_subscription_lifecycle()` | The agent selects from an operator-curated router catalog at runtime. |

`register_drasi_trigger()` replaces the pre-release `drasi_trigger()` name without an alias. It preserves the existing unpacked change-event format and task-mapping behavior. Static and agent-managed modes cannot be combined on one agent.

### Author-configured triggers

Register a fixed Drasi query on an agent before hosting. Call the helper several times to register different fixed queries on the same agent.

```python
agent = DurableAgent(...)

register_drasi_trigger(
    agent,
    query_id="<YOUR_DRASI_QUERY_ID>",
    task_mapper=lambda event, ctx: TriggerAction(task="<AGENT_TASK_MESSAGE>"),
)

runner = AgentRunner()
try:
    runner.subscribe(agent)
    await wait_for_shutdown()
finally:
    runner.shutdown(agent)
```

### Agent-managed subscriptions

Agent-managed mode requires:

- A fresh `DurableAgent` using the ordinary LLM/tool loop.
- `AgentStateConfig.store` backed by a state component that supports ETags and strong reads.
- An `AgentPubSubConfig` or an explicit `pubsub=` override.
- A deployed `DaprAgentRouter` with an operator-curated query catalog.
- Stable Dapr application and agent names.

The lifecycle prepares Drasi, starts the workflow worker, opens event intake only after readiness, and yields an ordinary `AgentRunner`. Keep the lifecycle object if the application may need to retry unresolved cleanup:

```python
from dapr_agents.workflow.utils.core import wait_for_shutdown

lifecycle = drasi_subscription_lifecycle(
    agent,
    router_id="drasi-system/sre-router-reaction",
    namespace="applications",
)

async with lifecycle as runner:
    await runner.run(
        agent,
        {"task": "Monitor service errors and record an assessment when they change."},
        wait=False,
    )
    await wait_for_shutdown()
```

For an application-owned FastAPI service, pass the app to make `serve()` the first normal hosting operation. This preserves the normal service routes, streaming/input routes, Pub/Sub wiring, and HITL endpoints:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with drasi_subscription_lifecycle(
        agent,
        router_id="drasi-system/sre-router-reaction",
        namespace="applications",
        app=app,
    ):
        yield


app = FastAPI(lifespan=lifespan)
```

#### Lifecycle configuration

| Argument | Default | Meaning |
|---|---|---|
| `router_id` | Required | Router identity in `<namespace>/<app-id>` form. |
| `namespace` | Required | Namespace of the subscriber application. |
| `app` | `None` | With a FastAPI app, the lifecycle hosts through `AgentRunner.serve()`; otherwise it starts workflow hosting only. |
| `workflow_client` | `None` | Optional caller-owned workflow client. Injected clients are not closed by the lifecycle. |
| `pubsub` | Agent's configured message bus | Application-facing Pub/Sub component used for the derived inbox and DLT. |
| `dapr_http_port` | `DAPR_HTTP_PORT` | Local sidecar HTTP port used for router service invocation. |
| `router_timeout_seconds` | `30.0` | Deadline for each router MCP exchange. |
| `startup_timeout_seconds` | `30.0` | Deadline for workflow-worker readiness. |

Application ID, exact logical agent name, state-store name, and workflow name come from the agent. Infrastructure identity, topics, and broker coordinates are never model arguments.

Startup revalidates the supported profile on context entry, runs public MCP discovery, revalidates again, resolves configuration, claims the application, and verifies sidecar identity and components. It then loads the catalog, collision-checks all generated tools, performs one final pre-mutation validation, initializes and reconciles durable intent, attaches tools, performs normal hosting, and verifies `WorkflowRuntime.wait_for_worker_ready()`. Configuration and tool registrations are checked again before the derived inbox/DLT opens last. An empty catalog still exposes `list_drasi_subscriptions()`. A failed catalog/state read or readiness check fails startup, rather than exposing a partially prepared consumer. No special startup model turn is performed.

Generated tools are added to the existing executor without replacing author tools or prompts. They remain available in independent event workflows. Initially tool-less agents are supported: the framework's explicit `none` and `required` choices are preserved, while the normal default becomes `auto` once tools exist.

The initial dynamic profile requires a fresh agent constructed without `runtime=`. A previously runner-hosted agent is rejected even after shutdown, using the core's closed registration window. Do not share or replace the runtime. Runtime ownership is a documented caller precondition because the current public framework API cannot inspect it. Dynamic mode also rejects `RuntimeSubscriptionConfig` and pre-existing activation callbacks. These are initial integration restrictions, not framework-wide rules. External `executor=` runtimes and orchestrators are also unsupported.

Use one active dynamic lifecycle per Dapr application and one logical agent per application in deployment. The application guard is process-local, not a distributed lease: deploy one replica. The POC supports one router connection and a small deployment-time catalog, not catalog polling. Static and dynamic modes cannot be mixed on one agent. Hosting is one-shot: after failure or shutdown, create fresh agent and runner objects. Durable intent and router rules survive ordinary shutdown and are reconciled by the fresh lifecycle.

An initial monitoring task may finish normally. A later matching event starts a new workflow using the same logical agent definition and the saved, self-contained handling instructions. It does not resume the original Python call stack or conversation. Use the generated unsubscribe tool to stop monitoring; leaving the context only cleans up hosting resources.

If context exit reports unresolved cleanup, retain the lifecycle object from the example above. After addressing the reported dependency or shutdown problem, retry cleanup as a separate operation:

```python
# Run later from the application's shutdown supervisor.
await lifecycle.aclose()
```

`aclose()` never rehosts the agent or removes durable intent and router rules.

The lifecycle creates and yields its runner. `workflow_client=` may inject a caller-owned workflow client; neither the runner nor the lifecycle closes that client. Otherwise, the lifecycle creates and owns the workflow client, lending it to the runner so close failures remain visible and retryable. The dynamic inbox owns a dedicated Dapr client created through `agent.client_factory`. Shutdown marks intake intentional, closes dynamic and runner Pub/Sub transports to unblock their streams, joins consumers, drains workflow execution, lets the runner perform its normal cleanup, then closes the owned workflow client and router resources. Unresolved cleanup retains the affected resources and application guard for a later `aclose()` attempt. It never unsubscribes or deletes durable intent. If application code raises or is cancelled, cleanup diagnostics are attached without replacing the original exception or cancellation.

The intent component must support ETags and honor strong reads. State/router errors fail preparation instead of producing an empty subscription set. Tool-name collisions and missing state, bus, or router configuration are actionable errors, not partially usable capabilities.

### Examples

- [Author-configured Drasi triggers on Kubernetes](../../examples/ext-drasi-change-driven-agents-k8s/README.md) demonstrates fixed query subscriptions and author-provided task mapping.
- [Credential-free runtime suite](../../tests/integration/drasi_managed/README.md) exercises the actual router, Dapr, and a deterministic local model. A new Kubernetes agent-managed example is separate work.

## Development

### Install extension in editable mode

From the project root:

```bash
uv venv
source .venv/bin/activate
uv sync --active --group dev --group test --extra drasi
```

### Run extension tests

```bash
DAPR_AGENTS_REQUIRE_DRASI=1 uv run pytest ext -m "not integration"
uv run pytest tests -m "not integration"
```

Run core and extension tests in separate processes. The core fixtures replace SDK modules and can hide extension imports; the required-import guard must remain enabled for extension validation.

### Extension code quality

Use the formatting, lint, and Python 3.11 mypy checks in the root [developer guide](../../AGENTS.md), including `ext` in the flake8 paths.

### Agent-managed architecture

`_models.py` and `_interfaces.py` define the private component contracts. The public `drasi_subscription_lifecycle()` context composes the router client, intent repository, subscription manager, generated tools, admission handler, and delivery adapter.

#### Configuration and ownership

`SubscriptionScope(router_id, namespace, app_id, agent_name)` is immutable and preserves exact identity strings. It validates identities through the shared router package and returns fresh shared `Subscriber` models. Its `inbox_topic` and `dead_letter_topic` properties use the shared identity helpers. Queries, operation filters, and incarnations do not change those topics.

`ResolvedDrasiConfig` contains that scope, `pubsub_name`, `state_store_name`, the already-registered `workflow_name`, and `dapr_http_port`. All values are resolved before construction; this model does not read environment variables, discover resources, or create clients. Its `router_mcp_url` addresses the configured router through the local Dapr sidecar.

All domain methods are synchronous and return completed results, not tasks, futures, or enqueue acknowledgements. Blocking state/network I/O belongs in preparation, a subscription worker, or an ordinary tool activity. Async callers must offload blocking calls rather than block their application event loop. Deterministic workflow bodies and replayed hooks must not perform this I/O. `MCPRouterClient` adapts the asynchronous MCP machinery while keeping each session, task, and event loop within one bounded exchange.

The public lifecycle owns preparation and resource lifetime: discover MCP tools, prepare the router catalog, initialize or load intent, reconcile the manager, attach tools, verify worker readiness, and only then admit inbox work. It closes partially prepared resources on failure. The inbox owns a dedicated Dapr client; tools and the manager borrow their router and state dependencies. The agent continues to own its configured state-store primitive. `RouterClient.close()` is idempotent and releases runtime resources only. Shutdown never removes intent or router rules.

#### Component signatures

The complete signatures are in `_interfaces.py`. `Operation` below is imported from the shared contract package, not the static trigger's separate operation enum.

| Boundary | Arguments and result | Semantics |
|---|---|---|
| `RouterClient.list_queries()` | Returns shared `ListQueriesResponse` | Prepare/cache the complete startup catalog for `scope`; an empty catalog is valid, a failed fetch is not an empty catalog. |
| `RouterClient.subscribe(request)` | Shared `SubscribeRequest` -> `SubscribeResponse` | A validated confirmation of the query, subscriber scope, incarnation, effective operation set, and derived topic. Both `created` and `updated` succeed. |
| `RouterClient.unsubscribe(request)` | Shared `UnsubscribeRequest` -> `UnsubscribeResponse` | A validated confirmation for the request; both `removed=True` and `removed=False` confirm absence. Retired queries remain removable. |
| `RouterClient.close()` | Returns `None` | The lifecycle owns cleanup, including after failed preparation. Calls after close are invalid lifecycle usage, not successful no-ops. |
| `IntentReader.get(query_id: str)` | Returns `SubscriptionIntent` or `None` | An owned local snapshot; `None` means genuinely absent intent. Reading may perform persistent I/O. |
| `IntentRepository.load()` | Returns `IntentSnapshot` or `None` | The whole scoped document and its ETag, including queries absent from the current catalog. `None` means the document is absent. |
| `IntentRepository.initialize(document)` | `IntentDocument` -> `None` | Unconditional initialization only during single-owner preparation, after checking absence and before admitting work. Not atomic create-if-absent. |
| `IntentRepository.save(document, *, expected_etag: str)` | Returns `None` | Conditional replacement of the whole document. Reload to obtain its next ETag; the underlying state primitive does not return one from save. |
| `SubscriptionManager.subscribe(query_id: str, *, operations: tuple[Operation, ...], instructions: str)` | Returns persisted, active `SubscriptionIntent` | New subscription or same-query update. Require a non-empty unique operation subset and non-blank, self-contained instructions. |
| `SubscriptionManager.unsubscribe(query_id: str)` | Returns `None` | Success only after router absence is confirmed and local intent is removed; already absent local intent is a successful no-op. |
| `SubscriptionManager.list_subscriptions()` | Returns `tuple[SubscriptionIntent, ...]` | Local intent, filters, instructions and status, not a claim about live router health. |
| `SubscriptionManager.reconcile(catalog)` | Shared `ListQueriesResponse` -> `None` | Preparation-time reconciliation; retain an owned catalog copy for later commands. No periodic refresh or startup model turn. |
| `AdmissionHandler.admit(data: object)` | Returns `AdmissionResult` | Decoded M2 CloudEvent **data**, not the outer CloudEvent; no scheduling or acknowledgement occurs here. |

The gateway consumes `parse`, `parse_catalog`, and `to_wire` from `drasi_agent_router_contracts`; generated Pydantic validation alone is insufficient. It inspects MCP `isError` before success parsing and treats operation arrays as sets. It never receives handling instructions. The current catalog contains `protocol_version`, `router_id`, and `queries`; there is no capability-negotiation or separate delivery-version field.

`router_client.MCPRouterClient(config, timeout_seconds=30.0)` implements that private gateway. Preparation calls `list_queries()` to validate and cache the catalog; mutations also establish the configured router identity before their first write if preparation has not already run. Each MCP exchange has a deadline and creates, initializes, and closes a stateless HTTP session on the client's worker thread. The SDK's typed request API leaves result validation to the shared contract so both structured and JSON-text results work. Calls are serialized with `close()`, and shutdown waits for the current exchange without removing router rules. The client does not retry mutations; reconciliation and retry decisions remain with the subscription manager.

The client honors the Dapr SDK's `DAPR_API_TOKEN` setting through the `dapr-api-token` header. Environment HTTP proxies are disabled because requests target the local sidecar. Tokens are not included in the client's URLs or diagnostic messages.

`RouterError.category` distinguishes `invalid_arguments`, `unknown_query`, `incarnation_conflict`, `state_unavailable`, `transport`, and `invalid_response`. The first three mean a rejected mutation. The latter three mean an uncertain mutation: a write may already have committed. A malformed or mismatched success response is not evidence of rejection. `mutation_outcome` makes this distinction explicit; it does not decide retry policy. For a catalog read, these categories describe retrieval failure, not a state mutation.

`IntentStoreError.category` distinguishes `unavailable`, `corrupt`, `unsupported_version`, and `conflict`. None may be converted into absent/empty state. An existing document without a usable ETag is a storage failure. `SubscriptionCommandError.category` distinguishes local `invalid_input`, `unknown_query`, `pending_operation`, and `unavailable` rejections. Manager commands may also propagate classified router/store errors. These errors carry safe categories, not raw responses, rows, or instructions; callers log operation context without logging sensitive payloads. Programming errors, cancellation, and invalid lifecycle usage are not converted into successful results.

#### Intent document and concurrency

`IntentDocument` has required `format_version=1`, a `SubscriptionScope`, and an `intents` mapping keyed by query ID. Each `SubscriptionIntent` contains its query ID, selected operations, handling instructions, historical full catalog snapshot, opaque non-empty incarnation, status, and optional `RouterOperationOutcome`. Record keys and catalog/router identities must agree. The format version is local persistence metadata, not a new router wire field.

`IntentSnapshot.etag` belongs to the **entire router/subscriber document**, never an individual query. A successful write to query A invalidates the token held by a simultaneous writer for query B. On conflict, the subscription manager must reread the document, reevaluate its targeted transition, and merge only that query's intended change into the fresh document before conditionally saving again. It must not resubmit an unchanged stale document or overwrite unrelated queries. The repository surfaces the conflict rather than silently performing that overwrite.

Initial creation is serialized by single-owner preparation. `initialize` is explicitly an unconditional write, not an `If-None-Match` operation and not a way around a failed CAS. Preparation reloads the created document to obtain an ETag before admitting work. Every subsequent update, including removal of a completed subscription, uses the whole-document ETag. An empty document can remain after the last unsubscribe. This design needs no state-store query API, per-query index, transaction framework, or distributed lease.

Statuses are `pending_subscribe`, `pending_update`, `active`, `pending_unsubscribe`, and `unavailable`. The subscription manager persists pending intent before calling the router and reports success only after its final local commit. Updates and retries retain incarnation; unsubscribe followed by a new subscription changes it. An uncertain mutation retains pending intent. A failed local write can also have committed, so recovery must reload rather than assume the previous state.

A definitive router rejection (`invalid_arguments`, `unknown_query`, or `incarnation_conflict`) leaves that intent `unavailable`, preserving its instructions, incarnation, and `last_outcome`. Explicit commands still report the error, but startup reconciliation continues with other queries once the rejection is durably recorded. Unavailable deliveries are discarded without scheduling work. Timeouts, malformed confirmations, router state failures, and a failure to persist a rejection still abort reconciliation; none implies rollback.

Recover unavailable intent through explicit unsubscribe, then subscribe again. Unsubscribe retains the existing incarnation and removes local intent only after the router confirms removal or absence. An incarnation conflict requires an operator to inspect and resolve the conflicting rule using the router's administration API before retrying unsubscribe. It never authorizes forced replacement or blind local deletion. Listing exposes the last router outcome and recovery guidance, not live router health.

Startup reasserts active or pending subscriptions, completes pending unsubscriptions even when a query has left the catalog, and otherwise marks removed queries unavailable while retaining their historical metadata. Query reappearance does not automatically revive unavailable intent. Ordinary shutdown does not unsubscribe. The subscription manager implements transition ordering and same-query serialization.

#### Durable intent storage

The private `intent_store.DaprIntentRepository(scope=..., store=...)` implements the repository boundary using the agent's configured `AgentStateConfig.store` (also exposed as `agent.state_store`). It borrows the existing raw Dapr state primitive and its configured factory, retaining the service's key prefix without applying its optional workflow model, local mirroring, or blanket retries. It does not change or close the configured service.

The storage key is `<service key prefix>drasi:intent:<scope.inbox_topic>`. The shared inbox identity includes the router, subscriber namespace, application, and exact agent name; queries live inside that document. Reads also validate the embedded scope. The format version remains in the document so an unsupported version cannot be mistaken for a new, absent key. The component must support ETags; nonempty data without a usable ETag fails as unavailable, and empty data with a nonempty ETag is corrupt. The native SDK represents absence with empty data and `etag=""`, not `None`. Its GetState response has no separate existence flag, so a backend with empty, untagged records cannot be distinguished from absence and is unsupported. Reads explicitly request strong consistency and return detached snapshots, including pending and retired-query intent. Because the pinned SDK's `get_state()` wrapper does not expose consistency, the adapter sends a native GetState request through the configured client's channel and unchanged read retry policy. The component must honor strong reads; ETags alone cannot prevent admission against stale intent.

Initialization is unconditional and restricted by the caller to exclusive preparation. Subsequent saves pass the expected document ETag with first-write-wins and strong write consistency. A gRPC `ABORTED` conditional save is a conflict; other transport failures are unavailable, and may follow a committed write. Each write uses a fresh client from the configured primitive's factory and explicitly disables SDK retries on that client. Otherwise, a committed write with a lost reply could be retried with its stale ETag and incorrectly reported as a conflict. The configured factory, read retry policy, and runtime-side resiliency policies are unchanged. The adapter does not retry or repair writes: callers reload after success or uncertain failure and merge against fresh state after conflicts. It adds no TTL, automatic deletion, router calls, or catalog filtering.

Malformed JSON, invalid records, mismatched scope, and missing or incorrectly typed versions fail as corrupt; an unsupported integer format version fails separately. Error messages and logs contain safe categories rather than backend responses or stored instructions. Compatibility cases in `tests/test_intent_store.py` construct real storage wrappers and SDK clients, mocking only health-check and state RPC I/O rather than injecting private client attributes. Missing private storage/client methods fail these cases visibly. The admission-level stale-replica regression uses the same normally constructed clients.

#### Subscription commands and recovery

The private `subscription_manager.DrasiSubscriptionManager(repository=..., router=...)` borrows matching scoped dependencies. Preparation initializes the repository and calls `reconcile(catalog)` before admitting work. Reconciliation retains an owned startup catalog, finishes pending unsubscriptions first (including retired queries), reasserts active/pending subscriptions with their existing incarnations, and marks other retired intent unavailable without deleting its historical context. It continues after durably recorded definitive router rejections, but uncertain outcomes and storage failures still fail preparation. Subscribe commands remain unavailable until preparation completes. Listing and unsubscribe do not need a current catalog, so retained intent can still be inspected and removed.

Subscribe and update persist pending intent before the router call and report success only after the active state is committed. An identical active command is a local no-op, not a router-health check. An identical pending command resumes the existing lifecycle; conflicting commands are rejected while that operation is unresolved. Updates retain incarnation. Unsubscribe persists pending-unsubscribe before the router call and removes intent only after confirmed absence, including an already-absent router response. Definitive router failures retain unavailable intent and a safe rejected outcome; uncertain failures retain pending intent. Local write failures remain visible and may require reloading after an uncertain commit.

An unavailable intent is never revived by catalog reappearance or a subscribe call. Subscribe returns an actionable error to unsubscribe first, then subscribe again. Unsubscribe uses the retained incarnation and the normal durable deletion flow; uncertain deletion retains pending-unsubscribe, while rejected deletion returns to unavailable. Only confirmed router deletion or absence permits removing local intent. The subsequent subscribe persists a fresh incarnation, while ordinary active-subscription updates retain their incarnation. This adds no automatic cleanup, replacement state machine, or permanent tombstones.

A fresh incarnation rejects already-published inbox messages carrying the old incarnation. It does not fence query generations: an upstream packed event retried later can be routed through the new rule and stamped with the new incarnation. Continue using fresh query IDs for deleted/recreated queries or changed query semantics. Explicit unsubscribe/re-subscribe is subscription-lifecycle hygiene, not query-generation fencing.

The single-owner POC manager serializes mutating commands within its process; listing reads persisted snapshots without waiting for a router call. Whole-document ETag conflicts reload and merge only the targeted query, with at most ten write attempts. A changed same-query intent is never overwritten. The manager adds no router discovery, periodic refresh, startup model turn, resource cleanup, or shutdown unsubscribe; the public lifecycle owns composition and resource lifetimes. Its synchronous methods perform I/O and belong in preparation or ordinary tool activities, never deterministic workflow bodies or replayed hooks.

#### Snapshot and admission boundaries

Return values are detached caller-owned snapshots. Components deep-copy models they return and inputs they retain; callers may modify their own copies without changing stored or prepared state. The configuration's immutable scalar identity is separate from these mutable owned snapshots. There is no second recursively immutable copy of the router model hierarchy.

`SubscriptionIntent` parses an owned catalog through the shared package and serializes it with `to_wire`, preserving omitted optional fields such as `usage`. Do not replace that serialization with an ordinary nested `model_dump()` that introduces unsupported explicit nulls. Persist documents using their model serialization and validate on load; do not bypass shared wire validation.

| Condition after protocol parsing | Admission result |
|---|---|
| No intent, unavailable intent, pending unsubscribe, or stale incarnation | `Discard` |
| Matching pending subscribe/update | `Retry`, before evaluating the requested operation filter |
| Intent storage unavailable, corrupt, or unsupported | `Retry`, not absent intent |
| Active intent excludes the operation | `Discard` |
| Malformed/unsupported delivery or row, or wrong router | `Poison` |
| Valid active matching intent | `SchedulingInput(instance_id, event_id, task)` |

`SchedulingInput.task` is complete, owned text containing the admitted instructions, catalog context, operation, event identity and canonical event data. The task builder delimits projected data as untrusted and derives a bounded deterministic instance ID from the router/subscriber scope, incarnation and canonical row event ID. Later intent changes cannot rewrite that text. No current time, model-generated token, or outer publication ID changes the agreed row identity.

The private `DrasiAdmissionHandler(scope=..., intents=...)` in `admission.py` implements this boundary using an `IntentReader` with the same scope. It accepts decoded M2 CloudEvent data only, validates finite JSON values before shared protocol parsing, verifies that the event can be encoded as task JSON before reading intent, and resolves lifecycle status before operation filtering. Delivery encoding failures are poison input, not corrupt-state retries; interpreter conversion limits remain unchanged. `task_builder.build_event_task()` freezes the admitted context and event into separately delimited JSON text. Its 70-character workflow ID is `drasi-` followed by the SHA-256 hex digest of the compact ASCII JSON tuple `("drasi-agent-workflow/v1", scope.inbox_topic, incarnation, event_id)`. This is correlation, not a permanent deduplication guarantee.

The delivery adapter maps these results to transport responses and schedules the configured workflow. A `SchedulingInput` is **not** an acknowledgement: success follows scheduling acceptance or an explicit existing-active-instance conflict, not local enqueue or workflow completion. Arbitrary scheduler errors retry. Duplicate workflows after terminal ID reuse remain possible.

The private `delivery.subscribe_drasi_inbox(config=..., admission=..., dapr_client=..., workflow_client=...)` opens the derived inbox and DLT on the configured application Pub/Sub component and returns an owned inbox handle. It decodes the SDK message's raw CloudEvent **data** without using the outer publication ID as workflow identity. A single consumer waits for `schedule_new_workflow()` acceptance before responding; it never waits for workflow completion or deduplicates against a local cache. Only the scheduler's structured gRPC `ALREADY_EXISTS` conflict is acknowledged as an existing active instance. Other errors or mismatched scheduling confirmations request retry.

Poison input, including JSON that exceeds the decoder's nesting limit, returns `DROP` on a subscription with its derived `dead_letter_topic` configured, instructing Dapr to forward the original message to that DLT without terminating the consumer. The adapter does not independently republish another copy. Intentional discards return `SUCCESS`; pending/infrastructure failures return `RETRY`. Configure the broker and Dapr inbound resiliency policy for the desired retry budget: retry exhaustion and DLT publication failure remain runtime/broker concerns, not a loss-free-delivery promise.

The pinned SDK buffers acknowledgement responses and logs rather than propagates `respond()` failures; its return is not proof of broker receipt. The adapter does not record a response as permanently delivered or suppress redelivery. An inactive SDK stream is surfaced by the next read and the closer; otherwise missing responses are governed by the configured runtime/broker timeout and retry policies. A redelivered event is admitted and scheduled again, including the normal active-ID conflict handling.

The inbox handle marks shutdown intentional, closes its owned Dapr client to interrupt the streaming iterator, closes the SDK's current subscription, and joins its consumer. Final consumer cleanup closes the current stream again, including any replacement opened by an in-flight reconnect. It records transport, subscription, and consumer completion separately; a later `close()` retries only unresolved resources. Stream/consumer failures are logged immediately. A consumer still running after the ten-second shutdown deadline raises `DrasiDeliveryError` and remains the lifecycle's responsibility. Long-running SDK calls must be bounded by deployment timeout/resiliency settings; shutdown does not cancel accepted workflows.

#### Component map

Extension-local `tests/conftest.py` and `tests/fakes.py` provide contract-backed catalogs and deliveries, pending-state examples, a scripted router, a scripted manager, and a document-level in-memory repository. They exercise component boundaries without becoming a second subscription engine. Scripted outcomes must be explicit; storage errors are not empty results, stale ETags conflict, initialization is not create-only, and returned mutable values are detached.

| Component | Production files and responsibilities |
|---|---|
| Private contracts | `_models.py`, `_interfaces.py`, shared fixtures and contract coverage |
| Router client | `router_client.py`: MCP lifecycle, parsing, reply validation and async adaptation |
| Intent repository | `intent_store.py`: configured runtime-store adapter, document codec and conditional writes |
| Subscription manager | `subscription_manager.py`: commands, transition serialization and reconciliation |
| Generated tools | `subscription_tools.py`: schemas, descriptions and delegation to the manager |
| Admission | `admission.py`, `task_builder.py`: persistent read, classification and detached event task |
| Delivery | `delivery.py`: inbox/DLT lifecycle, transport dispositions and scheduling acceptance |
| Public lifecycle | `subscriptions.py`: one-shot composition, tool attachment, mode exclusion and public export |

Composition is extension-owned rather than an activation hook. Dynamic mode rejects runtime configuration subscriptions and pre-existing activation callbacks, prepares before the first normal hosting call, verifies worker readiness, and opens the inbox last. Generated tools remain attached because failed or stopped agent/runner objects are one-shot and are not reused. Delivery uses the extension-owned adapter rather than the generic filter/mapper path.

#### Generated subscription tools

The private `subscription_tools.build_subscription_tools(catalog, manager)` factory returns ordinary synchronous `AgentTool` objects without performing network/state I/O or changing an agent. Each cached query gets subscribe/unsubscribe tools with bounded, deterministic names containing a digest of the exact query ID. Subscribe accepts only explicit, distinct `i/u/d` operations and non-blank, self-contained handling instructions. `list_drasi_subscriptions()` is always present, including for an empty catalog, and reports local intent rather than live router state.

The tools borrow the subscription manager interface and preserve classified command failures as error results, including the unsubscribe-before-resubscribe guidance for retained unavailable intent. Their diagnostics exclude handling instructions and raw payloads. This is an extension-tool logging boundary, not an end-to-end redaction guarantee: existing core console output, debug logging, and tracing may include tool arguments and results. The public lifecycle owns attachment to the existing executor, collision checks, and lifecycle preparation. The factory is not publicly re-exported and does not enable dynamic subscriptions by itself.

### Regenerate Drasi models

See the [provenance file](./PROVENANCE.md) for context.

```bash
./scripts/regen-drasi-models.sh
```
