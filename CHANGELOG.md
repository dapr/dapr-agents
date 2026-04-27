# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Streaming responses across all execution modes** (`.run()`, `.serve()`,
  `.subscribe()`, `.workflow()`). Opt in via `AgentExecutionConfig(streaming=True)`
  and subscribe through one of the built-in consumers.
  - `AgentRunner.run_stream(agent, payload, *, listener=None, include_complete_message=False)`
    returns an `AsyncIterator[AgentStreamChunk]`.
  - `.serve()` exposes `POST {entry_path}/stream` with content-negotiated
    SSE or NDJSON formatting (via the `Accept` header), plus
    `GET {status_path}/stream` to reattach to an in-flight pub/sub session
    after a client reconnect, and `POST {entry_path}/input` for the
    mid-stream `ask_user` flow.
  - Pluggable `StreamListener` transport protocol with built-in
    `PubSubListener`, `InProcessQueueListener`, `WebhookListener`, and
    `CompositeListener` implementations; `register_stream_listener()`
    registers custom transports by name.
- **Built-in `ask_user` tool** (opt-in via `AgentExecutionConfig(builtin_tools=["ask_user"])`
  together with `streaming=True`) that suspends the workflow on
  `wait_for_external_event` and emits `USER_INPUT_REQUESTED` chunks so a
  UI can route the question to a human. Answers are length-capped
  (4096 chars by default) and passed back to the LLM wrapped in neutral
  framing.
- **Multi-agent stream attribution** — child workflows inherit the parent's
  stream session and emit chunks on the same topic with
  `parent_agent` / `parent_instance_id` / `depth` / `call_path`
  populated. Orchestrators emit `ORCHESTRATION_DECISION`, `TURN_PAUSED`,
  and `TURN_RESUMED` envelopes around sub-agent dispatch.
- **Observability**: `LLMWrapper` is now streaming-aware. Token usage and
  `gen_ai.*` attributes are set at stream end via a shared
  `AssistantMessageAccumulator`, with no duplicate ingest when the
  observability wrapper and `StreamEmitter` are both active on the same
  iterator.

### Changed

- `DaprChatClient.generate(stream=True)` no longer raises
  `ValueError("Streaming is not supported by DaprChatClient.")`. The call
  now silently falls back to non-streaming and tags the returned
  `LLMChatResponse.metadata` with `dapr_streaming_fallback=True` so
  callers can detect the fallback. Code that caught the old `ValueError`
  for feature-detection should now inspect `metadata.dapr_streaming_fallback`.
- `StreamConsumer` Protocol gained `astart() -> None`. Existing
  implementations without it remain usable at runtime (Python structural
  typing does not enforce Protocol methods) but fail
  `isinstance(obj, StreamConsumer)` and trigger static-type-checker
  errors. Built-in consumers implement it; custom consumers should add a
  no-op for in-process transports or an eager-subscribe call for
  long-distance transports.
- `build_listener(config, *, allow_custom=False, **kwargs)` now takes an
  `allow_custom` kwarg. The default is `False` so that HTTP-reachable
  paths never invoke the `"custom"` dotted-import factory. Python
  callers with a trusted config can pass `allow_custom=True`; otherwise
  pre-register custom transports via
  `register_stream_listener("my-type", factory)` and reference them by
  name.
- `StreamEmitter.__init__` gained an `owns_listener: bool = True` kwarg.
  When `False`, `close()` is a no-op — used by the per-session listener
  cache where the listener's lifetime is owned by `DurableAgent`.
- `AssistantMessageAccumulator` moved from
  `dapr_agents.streaming.emitter` to `dapr_agents.types.streaming`. The
  `dapr_agents.streaming` package continues to re-export it; imports from
  the old path continue to work.
- `PubSubListener._run` was rebuilt to reuse a single Dapr client and
  event loop across the listener's lifetime and publishes in batches via
  `asyncio.gather`. Expect 3-6× lower per-chunk publish overhead on warm
  sidecars.
- `InProcessQueueConsumer` now awaits directly on an `asyncio.Queue` fed
  by `loop.call_soon_threadsafe(put_nowait, ...)` from the activity
  thread, replacing the previous `run_in_executor` thread-pool hop per
  chunk.
- `DaprInferenceClient.get_metadata()` is now cached per-instance with a
  configurable TTL (default 60s). Use `invalidate_metadata_cache()` when
  a component is known to have hot-reloaded faster than the window.
- **Backwards-compatibility safeguards added for the streaming feature**:
  - `AgentExecutionConfig.builtin_tools` defaults to `[]` (empty); upgrading
    agents do not suddenly see `ask_user` in their LLM tool schema.
  - The `ask_user` tool is only registered when
    `execution.streaming=True`, preventing the LLM from calling a tool
    that would block on a never-arriving event.

### Security

- The `"custom"` listener type (dotted-import factory) now requires
  `allow_custom=True` on `build_listener(...)`. HTTP ingress paths
  always pass `allow_custom=False`, so `POST /stream` bodies that try to
  invoke arbitrary `importlib.import_module` factories are rejected with
  HTTP 400. Trusted custom transports should register at startup via
  `register_stream_listener("my-type", factory)`.
- `POST {entry_path}/input` validates `answer` type and enforces a 4096
  character maximum; oversized answers return HTTP 413.

### Fixed

- `PubSubStreamConsumer` subscription now opens eagerly in
  `AgentRunner.run_stream` before the workflow is scheduled, closing a
  race where a fast activity could emit `START` before the consumer was
  ready.
- `TURN_COMPLETE.complete_message` no longer leaks the full assistant
  message on the Dapr Conversation API fallback path. Consumers must
  opt in via `include_complete_message=True` on `run_stream` to receive
  the assembled body through the stream topic; durable state remains
  the authoritative record.
- The SSE/NDJSON stream handler synthesises a terminal `ERROR` chunk on
  transport failure so clients can distinguish a clean end-of-stream
  from a socket EOF.
- `StreamingResponse`s now include `Cache-Control: no-store` and
  `X-Accel-Buffering: no` headers to prevent proxies from buffering the
  stream.
