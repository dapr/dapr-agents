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

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import uuid
from threading import Lock, Thread
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    Union,
)

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dapr_agents.agents.durable import DurableAgent
from dapr_agents.streaming.consumers import (
    InProcessQueueConsumer,
    PubSubStreamConsumer,
)
from dapr_agents.streaming.keys import (
    INCLUDE_COMPLETE_MESSAGE,
    MESSAGE_METADATA,
    STREAM_LISTENER_CONFIG,
    USER_INPUT_EVENT_PREFIX,
)
from dapr_agents.tool.workflow.ask_user_tool import ASK_USER_ANSWER_MAX_LEN
from dapr_agents.streaming.listeners import (
    register_in_process_queue,
)
from dapr_agents.tool.workflow.agent_tool import AgentWorkflowTool
from dapr_agents.types.streaming import AgentStreamChunk, StreamChunkType
from dapr_agents.types.activation import ActivationContext
from dapr_agents.types.workflow import PubSubRouteSpec
from dapr_agents.utils import DaprClientFactory
from dapr_agents.workflow.runners.base import WorkflowRunner
from dapr_agents.workflow.utils.core import get_decorated_methods
from dapr_agents.workflow.utils.registration import (
    register_http_routes,
    register_message_routes,
)
from dapr_agents.workflow.utils.subscription import TTLDedupeBackend

logger = logging.getLogger(__name__)

# Agent names we've already warned about the unsafe in_process streaming default
# — warn once per agent (not per session) to avoid log spam.
_WARNED_IN_PROCESS_DEFAULT: set = set()

R = TypeVar("R")


class _HitlRespondBody(BaseModel):
    approved: bool
    reason: Optional[str] = None


def _payload_as_dict(
    payload: Optional[Union[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Normalize a scheduler-style payload into a mutable dict."""

    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        # Treat bare strings as a prompt. Callers who want structured input
        # should pass a dict directly.
        return {"task": payload}
    raise TypeError(
        f"run_stream payload must be dict, str, or None; got {type(payload).__name__}"
    )


def _format_sse(chunk: AgentStreamChunk) -> str:
    """Serialize a chunk as a Server-Sent Events frame."""

    return f"event: {chunk.type.value}\ndata: {chunk.model_dump_json()}\n\n"


def _format_ndjson(chunk: AgentStreamChunk) -> str:
    """Serialize a chunk as a newline-delimited JSON line."""

    return chunk.model_dump_json() + "\n"


def _build_terminal_error_chunk(
    *, sequence: int, exc: BaseException
) -> AgentStreamChunk:
    """Construct a minimal ``ERROR`` chunk for an HTTP stream handler.

    Attribution is intentionally sparse: we don't know which agent /
    workflow instance the failure belongs to when the exception originated
    in the consumer plumbing itself (e.g., before any real chunk
    arrived). The ``error`` payload is the authoritative signal.
    """

    return AgentStreamChunk(
        sequence=sequence,
        type=StreamChunkType.ERROR,
        agent="runner",
        workflow_instance_id="",
        turn=0,
        root_instance_id="",
        depth=0,
        error={"type": type(exc).__name__, "message": str(exc)},
    )


class AgentRunner(WorkflowRunner):
    """
    Runner specialized for Agent classes.
    """

    def __init__(
        self,
        *,
        name: str = "agent-runner",
        wf_client=None,
        timeout_in_seconds: int = 600,
        auto_install_signals: bool = False,
        client_factory: Optional[DaprClientFactory] = None,
    ) -> None:
        """
        Initialize an AgentRunner.

        Args:
            name: Logical name used in logs (defaults to "agent-runner").
            wf_client: Optional injected DaprWorkflowClient. If omitted, a new one is created.
            timeout_in_seconds: Default timeout used when waiting for workflow completion.
            auto_install_signals: If True, installs SIGINT/SIGTERM handlers automatically
                when used as a context manager (with/async with) and removes them on exit.
            client_factory: Optional sync Dapr client factory forwarded to the
                underlying workflow runner. Defaults to the env-driven factory.
        """
        super().__init__(
            name=name,
            wf_client=wf_client,
            timeout_in_seconds=timeout_in_seconds,
            auto_install_signals=auto_install_signals,
            client_factory=client_factory,
        )
        self._default_http_paths: set[str] = set()

        # In-memory store of managed agents - used for handling shutdown
        self._managed_agents: List[DurableAgent] = []
        self._lock: Lock = Lock()
        # Activation-hook state (see _attach_agent / DurableAgent.add_activation):
        #   _activated_agent_ids — fire-once guard keyed by id(agent)
        #   _activation_closers  — teardown closers per agent, drained on shutdown
        self._activated_agent_ids: set[int] = set()
        self._activation_closers: Dict[int, List[Callable[[], None]]] = {}

    @staticmethod
    async def _ensure_mcp_connected(agent: DurableAgent) -> None:
        """Connect MCPServer tools if the agent supports auto-discovery."""
        connect_fn = getattr(agent, "connect_mcpservers", None)
        if connect_fn and not getattr(agent, "_mcp_tools_connected", True):
            await connect_fn()

    def _ensure_mcp_connected_sync(self, agent: DurableAgent) -> None:
        """Sync wrapper for MCP auto-discovery (for non-async entry points)."""
        if getattr(agent, "_mcp_tools_connected", True):
            return
        try:
            asyncio.get_running_loop()
            is_in_event_loop = True
        except RuntimeError:
            is_in_event_loop = False

        coroutine = self._ensure_mcp_connected(agent)
        if is_in_event_loop:
            self._run_coro_in_new_loop_thread(coroutine)
        else:
            asyncio.run(coroutine)

    async def run(
        self,
        agent: DurableAgent,
        payload: Optional[Union[str, Dict[str, Any]]] = None,
        *,
        instance_id: Optional[str] = None,
        wait: bool = True,
        timeout_in_seconds: Optional[int] = None,
        fetch_payloads: bool = True,
        log: bool = True,
    ) -> Union[str, Optional[str]]:
        """
        Run an Agent's workflow entry.

        Args:
            agent: Agent instance containing exactly one bound method marked with `@workflow_entry`.
            payload: Workflow input (JSON-serializable dict or string).
            instance_id: Workflow instance id; if omitted, a new UUID is generated.
            wait: If True, wait for completion and return serialized output; otherwise return instance id immediately.
            timeout_in_seconds: Max time to wait when wait=True. If omitted (Runner's timeout), defaults to the runner's configured timeout.
                Ignored when wait=False.
            fetch_payloads: Whether to fetch input/output payloads when waiting.
            log: If True, log the final outcome (sync if `wait=True`, background if `wait=False`).

        Returns:
            - If `wait=False`: the workflow instance id (str).
            - If `wait=True`: the serialized output string, or `None` on timeout/error.

        Raises:
            RuntimeError: If zero or multiple entry methods are found on the Agent.
        """
        logger.debug(
            "[%s] Start run: agent=%s payload=%s wait=%s timeout=%s",
            self._name,
            type(agent).__name__,
            payload,
            wait,
            timeout_in_seconds,
        )
        await self._ensure_mcp_connected(agent)
        self._attach_agent(agent)

        entry = self.discover_entry(agent)
        logger.debug("[%s] Discovered workflow entry: %s", self._name, entry.__name__)

        return await self.run_workflow_async(
            entry,
            payload,
            instance_id=instance_id,
            timeout_in_seconds=timeout_in_seconds,
            fetch_payloads=fetch_payloads,
            detach=not wait,
            log=log,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def run_stream(
        self,
        agent: DurableAgent,
        payload: Optional[Union[str, Dict[str, Any]]] = None,
        *,
        instance_id: Optional[str] = None,
        listener: Optional[Dict[str, Any]] = None,
        include_complete_message: bool = False,
        timeout_in_seconds: Optional[int] = None,
    ) -> AsyncIterator[AgentStreamChunk]:
        """Schedule an agent workflow and yield ``AgentStreamChunk`` until
        ``SESSION_COMPLETE`` or the workflow reaches a terminal state.

        Args:
            agent: Durable agent whose workflow entry to invoke.
            payload: Workflow input (dict preferred so streaming metadata can be
                attached). Strings are wrapped in ``{"task": payload}``.
            instance_id: Optional explicit instance id. Autogenerated otherwise
                and used as the session's ``root_instance_id``.
            listener: Optional listener config dict
                (``{"type": "pubsub" | "in_process" | ...}``). When omitted,
                the runner picks a topology-safe default based on the agent's
                registered peers.
            include_complete_message: If True, terminal chunks carry the full
                ``AssistantMessage`` in ``complete_message``. Default False —
                clients reconstruct from deltas.
            timeout_in_seconds: Maximum time to wait for ``SESSION_COMPLETE``
                before the iterator stops itself.

        Yields:
            ``AgentStreamChunk`` objects in arrival order.
        """

        try:
            agent.start()
        except RuntimeError:
            pass
        with self._lock:
            if agent not in self._managed_agents:
                self._managed_agents.append(agent)

        chosen_instance_id = instance_id or uuid.uuid4().hex
        listener_config = self._resolve_default_listener(
            agent=agent,
            explicit=listener,
            root_instance_id=chosen_instance_id,
        )

        payload_dict = _payload_as_dict(payload)
        metadata = dict(payload_dict.get(MESSAGE_METADATA) or {})
        metadata[STREAM_LISTENER_CONFIG] = listener_config
        if include_complete_message:
            metadata[INCLUDE_COMPLETE_MESSAGE] = True
        payload_dict[MESSAGE_METADATA] = metadata

        consumer = self._build_stream_consumer(
            listener_config=listener_config,
            root_instance_id=chosen_instance_id,
        )

        entry = self.discover_entry(agent)
        logger.debug(
            "[%s] run_stream: instance_id=%s listener=%s",
            self._name,
            chosen_instance_id,
            listener_config.get("type"),
        )
        # Open the consumer (subscribe to the pub/sub topic, or no-op for
        # in-process registry entries) BEFORE scheduling the workflow. A
        # fast activity would otherwise emit ``START`` before the pubsub
        # subscription is live, and the first chunk would be lost.
        try:
            await consumer.astart()
        except Exception:
            await consumer.aclose()
            raise

        try:
            await self.run_workflow_async(
                entry,
                payload_dict,
                instance_id=chosen_instance_id,
                timeout_in_seconds=timeout_in_seconds,
                detach=True,
                log=False,
            )
        except Exception:
            await consumer.aclose()
            raise

        try:
            async for chunk in consumer:
                yield chunk
        finally:
            await consumer.aclose()

    def _resolve_default_listener(
        self,
        *,
        agent: DurableAgent,
        explicit: Optional[Dict[str, Any]],
        root_instance_id: str,
    ) -> Dict[str, Any]:
        """Pick a listener config when the caller didn't specify one.

        Order of precedence: explicit > agent.execution.stream_listener >
        mode-based default. The mode-based default prefers ``pubsub`` when the
        agent has cross-app peers, is horizontally scaled, or has a message bus
        configured (all safe across a multi-replica handoff), and only falls
        back to ``in_process`` when no bus is available (local / direct-run).
        """

        if explicit:
            return self._materialize_listener_config(
                explicit,
                root_instance_id,
                infra=agent._infra,
                agent_name=agent.name,
            )

        configured = getattr(agent.execution, "stream_listener", None)
        if configured:
            return self._materialize_listener_config(
                configured,
                root_instance_id,
                infra=agent._infra,
                agent_name=agent.name,
            )

        # Fall-through default. Prefer pubsub when the agent has cross-app peers
        # OR the deployment is horizontally scaled: the in-process listener is
        # bound to one process's registry, so after a rolling restart rehydrates
        # the workflow on a different replica, emits would land nowhere.
        if self._agent_has_cross_app_peers(agent) or self._is_multi_replica():
            return self._materialize_listener_config(
                {"type": "pubsub"},
                root_instance_id,
                infra=agent._infra,
                agent_name=agent.name,
            )
        # Safe-by-default when deployed: if a message bus is configured, stream
        # over pubsub so the session survives a multi-replica handoff even when
        # the operator hasn't set a replica-count env var. Fall through to
        # in_process only when there is no bus (local / direct-run), which
        # cannot cross processes anyway.
        infra = getattr(agent, "_infra", None)
        if infra is not None and getattr(infra, "message_bus_name", None):
            return self._materialize_listener_config(
                {"type": "pubsub"},
                root_instance_id,
                infra=infra,
                agent_name=agent.name,
            )
        # in_process is the last resort (no bus available). Warn once per agent
        # so horizontally-scaled deployments know it is unsafe across handoff.
        if agent.name not in _WARNED_IN_PROCESS_DEFAULT:
            _WARNED_IN_PROCESS_DEFAULT.add(agent.name)
            logger.warning(
                "Agent '%s' is streaming over the default 'in_process' listener "
                "(no message bus configured). This is unsafe across multi-replica "
                "handoff: a workflow rehydrated on another replica after a restart "
                "emits to a dead in-process queue. For horizontally-scaled "
                "deployments configure a message bus or set "
                "stream_listener={'type': 'pubsub'}.",
                agent.name,
            )
        return self._materialize_listener_config(
            {"type": "in_process"}, root_instance_id, agent_name=agent.name
        )

    @staticmethod
    def _agent_has_cross_app_peers(agent: DurableAgent) -> bool:
        executor = getattr(agent, "tool_executor", None)
        if executor is None:
            return False
        tools = getattr(executor, "tools", None) or []
        for tool in tools:
            if isinstance(tool, AgentWorkflowTool) and tool.target_app_id:
                return True
        return False

    @staticmethod
    def _is_multi_replica() -> bool:
        """Best-effort detection that this app runs more than one replica.

        In-process stream listeners are bound to a single process, so a
        horizontally-scaled agent must use pubsub to survive replica handoff.
        There is no universal runtime signal for replica count, so honor an
        explicit operator-set env: ``DAPR_AGENTS_MULTI_REPLICA`` (truthy) or a
        replica-count env (``DAPR_AGENTS_REPLICA_COUNT`` / ``REPLICA_COUNT``) > 1.
        """
        flag = os.getenv("DAPR_AGENTS_MULTI_REPLICA", "").strip().lower()
        if flag in ("1", "true", "yes", "on"):
            return True
        for var in ("DAPR_AGENTS_REPLICA_COUNT", "REPLICA_COUNT"):
            raw = os.getenv(var)
            if raw and raw.strip().isdigit() and int(raw) > 1:
                return True
        return False

    @staticmethod
    def _materialize_listener_config(
        config: Dict[str, Any],
        root_instance_id: str,
        *,
        infra: Optional[Any] = None,
        agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fill in session-scoped fields (topic, registry_key) on the config.

        ``pubsub`` entries get ``pubsub_name`` and ``topic`` derived from the
        agent's ``DaprInfra``; ``in_process`` entries get ``registry_key`` set
        to the root instance id. Any user-provided values win.

        Fails fast when a ``pubsub`` listener cannot be materialised (no
        ``infra`` and no explicit ``pubsub_name``/``topic``) so the caller
        sees a clear error before scheduling the workflow rather than an
        opaque ``KeyError`` inside ``PubSubListener.__init__`` later.
        """

        result = dict(config)
        kind = result.get("type")
        if kind == "in_process":
            result.setdefault("registry_key", root_instance_id)
        elif kind == "pubsub":
            if infra is not None:
                if "pubsub_name" not in result and infra.message_bus_name:
                    result["pubsub_name"] = infra.message_bus_name
                if "topic" not in result:
                    topic = infra.stream_topic_name(root_instance_id)
                    if topic:
                        result["topic"] = topic
            missing = [f for f in ("pubsub_name", "topic") if not result.get(f)]
            if missing:
                who = f"agent {agent_name!r} " if agent_name else ""
                raise ValueError(
                    f"pubsub listener requires {missing}; "
                    f"{who}has no AgentPubSubConfig and none were supplied "
                    f"in the listener config"
                )
        return result

    def _build_stream_consumer(
        self,
        *,
        listener_config: Dict[str, Any],
        root_instance_id: str,
    ):
        """Create a consumer matching the listener type.

        For ``in_process`` we also register the process-local queue here —
        before the workflow schedules — so the activity's ``START`` chunk has
        somewhere to land.
        """

        kind = listener_config.get("type")
        if kind == "in_process":
            registry_key = listener_config["registry_key"]
            try:
                register_in_process_queue(registry_key)
            except ValueError:
                # Already registered (edge case: caller reused an instance_id).
                pass
            return InProcessQueueConsumer(
                registry_key=registry_key,
                root_instance_id=root_instance_id,
            )
        if kind == "pubsub":
            return PubSubStreamConsumer(
                pubsub_name=listener_config["pubsub_name"],
                topic=listener_config["topic"],
                root_instance_id=root_instance_id,
            )
        raise ValueError(
            f"run_stream cannot consume from listener type '{kind}'. "
            "Use in_process or pubsub, or subscribe to the webhook/custom "
            "transport yourself."
        )

    def run_sync(
        self,
        agent: DurableAgent,
        payload: Optional[Union[str, Dict[str, Any]]] = None,
        *,
        instance_id: Optional[str] = None,
        timeout_in_seconds: Optional[int] = None,
        fetch_payloads: bool = True,
        log: bool = True,
    ) -> Optional[str]:
        """
        Synchronously run an Agent's workflow entry and wait for completion.

        Args:
            agent: Agent instance containing exactly one bound method marked with `@workflow_entry`.
            payload: Workflow input (JSON-serializable dict or string).
            instance_id: Workflow instance id; if omitted, a new UUID is generated.
            timeout_in_seconds: Max time to wait when wait=True. If omitted (Runner's timeout), defaults to the runner's configured timeout.
                Ignored when wait=False.
            fetch_payloads: Whether to fetch input/output payloads when waiting.
            log: If True, log the final outcome.

        Returns:
            Serialized output string, or `None` on timeout/error.
        """
        coro = self.run(
            agent,
            payload,
            instance_id=instance_id,
            wait=True,
            timeout_in_seconds=timeout_in_seconds,
            fetch_payloads=fetch_payloads,
            log=log,
        )
        try:
            asyncio.get_running_loop()
            return self._run_coro_in_new_loop_thread(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def workflow(
        self,
        agent: DurableAgent,
    ) -> "AgentRunner":
        """
        Start the agent's workflow runtime without wiring pub/sub or HTTP routes.

        Use this when the agent is triggered by external Dapr workflows or the
        Dapr Workflow API rather than pub/sub messages or HTTP requests. Call
        ``await wait_for_shutdown()`` after this to keep the runtime alive.

        Args:
            agent: Durable agent instance.

        Returns:
            The runner (to allow fluent chaining).
        """
        self._ensure_mcp_connected_sync(agent)
        self._attach_agent(agent)

        return self

    def discover_entry(self, agent: Any) -> Callable[..., Any]:
        """
        Locate exactly one bound method on `agent` marked with `@workflow_entry`.

        If the agent exposes an ``agent_workflow_name`` property (as
        ``DurableAgent`` does), a lightweight stub with that name is returned so
        that ``schedule_new_workflow`` schedules the correct registered workflow
        (e.g. ``"frodo_agent_workflow"``) rather than the generic method name.
        The Dapr SDK accepts either a callable or a plain string; passing a stub
        with the right ``__name__`` keeps the call-site uniform.

        Returns:
            A callable whose ``__name__`` matches the registered workflow name.

        Raises:
            RuntimeError: If zero or multiple @workflow_entry methods are found.
        """
        candidates: list[Callable[..., Any]] = []
        for attr in dir(agent):
            fn = getattr(agent, attr)
            if callable(fn) and getattr(fn, "_is_workflow_entry", False):
                # Ensure it's bound to THIS instance (not a function on the class)
                if getattr(fn, "__self__", None) is agent:
                    candidates.append(fn)

        if not candidates:
            raise RuntimeError("Agent has no @workflow_entry method.")
        if len(candidates) > 1:
            names = ", ".join(getattr(c, "__name__", "<callable>") for c in candidates)
            raise RuntimeError(f"Agent has multiple @workflow_entry methods: {names}")

        # If the agent knows its registered workflow name, use that directly so
        # schedule_new_workflow resolves to the right Dapr registration.
        registered_name: Optional[str] = getattr(agent, "agent_workflow_name", None)
        if registered_name:

            def _stub(*_) -> None:
                pass

            _stub.__name__ = registered_name
            return _stub

        return candidates[0]

    @staticmethod
    def _run_coro_in_new_loop_thread(
        coro: "asyncio.Future[R] | asyncio.coroutines.Coroutine[Any, Any, R]",
    ) -> R:
        """
        Execute an async coroutine in a brand-new event loop on a background thread,
        then return its result to the current thread (which may already be running a loop).

        This enables `run_sync` to work in notebooks and ASGI servers.

        Args:
            coro: The coroutine to run.

        Returns:
            The coroutine's result, or raises its exception.
        """
        fut: "concurrent.futures.Future[R]" = concurrent.futures.Future()

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(coro)
                fut.set_result(result)
            except Exception as exc:  # noqa: BLE001
                fut.set_exception(exc)
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    loop.close()

        t = Thread(target=_runner, daemon=True)
        t.start()
        return fut.result()

    def register_routes(
        self,
        agent: DurableAgent,
        *,
        fastapi_app: Optional[FastAPI] = None,
        delivery_mode: Literal["sync", "async"] = "sync",
        queue_maxsize: int = 1024,
        await_result: bool = False,
        await_timeout: Optional[int] = None,
        fetch_payloads: bool = True,
        log_outcome: bool = False,
    ) -> None:
        """
        Register message/HTTP routes for a single durable agent instance.

        Args:
            agent: The agent instance whose routes should be registered.
            fastapi_app: Optional FastAPI app to register HTTP routes on. If omitted, no HTTP routes are registered.
            delivery_mode: "sync" or "async" delivery for message handlers.
            queue_maxsize: Max size of internal message queues.
            await_result: If True, message handlers will await workflow results.
            await_timeout: Max time to wait for workflow results when `await_result=True`. If omitted (None), waits indefinitely.
            fetch_payloads: Whether to fetch input/output payloads for awaited workflows.
            log_outcome: Whether to log the final outcome of awaited workflows.
        """
        self._ensure_mcp_connected_sync(agent)
        self._attach_agent(agent, app=fastapi_app)

        self._wire_pubsub_routes(
            agent=agent,
            delivery_mode=delivery_mode,
            queue_maxsize=queue_maxsize,
            await_result=await_result,
            await_timeout=await_timeout,
            fetch_payloads=fetch_payloads,
            log_outcome=log_outcome,
        )

        if fastapi_app is not None:
            self._wire_http_routes(agent=agent, fastapi_app=fastapi_app)

    def _build_pubsub_specs(
        self, agent: DurableAgent, config: Any
    ) -> list[PubSubRouteSpec]:
        handlers = get_decorated_methods(agent, "_is_message_handler")
        if not handlers:
            return []

        specs: list[PubSubRouteSpec] = []
        for _, handler in handlers.items():
            meta = getattr(handler, "_message_router_data", {})
            is_broadcast = meta.get("is_broadcast", False)
            topic: Optional[str] = (
                config.broadcast_topic if is_broadcast else config.agent_topic
            )
            if not topic:
                kind = "broadcast" if is_broadcast else "direct"
                raise ValueError(
                    f"AgentPubSubConfig missing topic for {kind} handler {handler.__name__}"
                )

            schemas = meta.get("message_schemas") or []
            message_model = schemas[0] if schemas else None

            # Use the agent's registered workflow name so schedule_new_workflow
            # resolves to the correct Dapr registration (e.g. "frodo_agent_workflow").
            registered_name: Optional[str] = (
                getattr(agent, "broadcast_workflow_name", None)
                if is_broadcast
                else getattr(agent, "agent_workflow_name", None)
            )
            if registered_name is not None:

                def _stub(*_) -> None:
                    pass

                _stub.__name__ = registered_name
                named_handler: Any = _stub
            else:
                named_handler = handler

            specs.append(
                PubSubRouteSpec(
                    pubsub_name=config.pubsub_name,
                    topic=topic,
                    handler_fn=named_handler,
                    message_model=message_model,
                )
            )

        return specs

    def _wire_pubsub_routes(
        self,
        *,
        agent: DurableAgent,
        delivery_mode: Literal["sync", "async"],
        queue_maxsize: int,
        await_result: bool,
        await_timeout: Optional[int],
        fetch_payloads: bool,
        log_outcome: bool,
    ) -> None:
        config = getattr(agent, "pubsub", None)
        if config is None:
            logger.debug(
                "[%s] Agent %s has no pubsub; skipping pub/sub route registration.",
                self._name,
                getattr(agent, "name", agent),
            )
            return

        specs = self._build_pubsub_specs(agent, config)
        if not specs:
            return

        self._ensure_dapr_client()
        if self._wired_pubsub or self._dapr_client is None:
            return

        try:
            deduper = TTLDedupeBackend()
        except ImportError:
            logger.warning(
                "cachetools not installed; disabling pub/sub message deduplication for agent %s",
                getattr(agent, "name", agent),
            )
            deduper = None

        closers = register_message_routes(
            routes=specs,
            dapr_client=self._dapr_client,
            delivery_mode=delivery_mode,
            queue_maxsize=queue_maxsize,
            wf_client=self._wf_client,
            await_result=await_result,
            await_timeout=await_timeout,
            fetch_payloads=fetch_payloads,
            log_outcome=log_outcome,
            deduper=deduper,
            client_factory=self._client_factory,
        )
        self._pubsub_closers.extend(closers)
        self._wired_pubsub = True

    def _wire_http_routes(
        self,
        *,
        agent: DurableAgent,
        fastapi_app: Optional[FastAPI],
    ) -> None:
        if fastapi_app is None or self._wired_http:
            return

        register_http_routes(
            app=fastapi_app,
            targets=[agent],
            routes=None,
        )
        self._wired_http = True

    def subscribe(
        self,
        agent: DurableAgent,
        *,
        delivery_mode: Literal["sync", "async"] = "sync",
        queue_maxsize: int = 1024,
        await_result: bool = False,
        await_timeout: Optional[int] = None,
        fetch_payloads: bool = True,
        log_outcome: bool = False,
    ) -> "AgentRunner":
        """
        Wire the agent's pub/sub triggers without exposing HTTP routes.

        Args:
            agent: Durable agent instance.
            delivery_mode: Delivery mode for pub/sub handlers.
            queue_maxsize: Queue size when delivery_mode="async".
            await_result: Whether message handlers wait for workflow completion.
            await_timeout: Timeout applied when awaiting workflow completion.
            fetch_payloads: Include input/output payloads when awaiting.
            log_outcome: Log workflow outcome on completion.

        Returns:
            The runner (to allow fluent chaining).
        """
        self._ensure_mcp_connected_sync(agent)
        self._attach_agent(agent)

        self._wire_pubsub_routes(
            agent=agent,
            delivery_mode=delivery_mode,
            queue_maxsize=queue_maxsize,
            await_result=await_result,
            await_timeout=await_timeout,
            fetch_payloads=fetch_payloads,
            log_outcome=log_outcome,
        )
        return self

    def serve(
        self,
        agent: DurableAgent,
        *,
        app: Optional[FastAPI] = None,
        host: str = "0.0.0.0",
        port: int = 8001,
        expose_entry: bool = True,
        entry_path: str = "/agent/run",
        status_path: str = "/agent/instances/{instance_id}",
        workflow_component: str = "dapr",
        fetch_status_payloads: bool = True,
        delivery_mode: Literal["sync", "async"] = "sync",
        queue_maxsize: int = 1024,
    ) -> FastAPI:
        """
        Host the agent as a service: subscribe to pub/sub triggers and expose HTTP endpoints.

        Args:
            agent: Durable agent instance.
            app: Optional FastAPI application to mount routes on. When omitted a default
                 FastAPI app is created and uvicorn is started automatically.
            host: Host address when auto-running the FastAPI app.
            port: Port when auto-running the FastAPI app.
            expose_entry: Mount a default POST endpoint that schedules the workflow entry.
            entry_path: HTTP path for the default POST endpoint.
            status_path: HTTP path for the status endpoint (must include `{instance_id}`).
            workflow_component: Workflow component name used in the returned status URL.
            fetch_status_payloads: Include payloads when fetching workflow status.
            delivery_mode: Delivery mode forwarded to `subscribe`.
            queue_maxsize: Queue size forwarded to `subscribe` for async delivery.

        Returns:
            The FastAPI application with the workflow routes.
        """

        fastapi_app = app or FastAPI(title="Dapr Agent Service", version="1.0.0")

        self._ensure_mcp_connected_sync(agent)
        # Attach here (before the nested subscribe) so the activation context
        # carries the FastAPI app; the nested subscribe() then no-ops.
        self._attach_agent(agent, app=fastapi_app)

        self.subscribe(
            agent,
            delivery_mode=delivery_mode,
            queue_maxsize=queue_maxsize,
        )

        self._wire_http_routes(agent=agent, fastapi_app=fastapi_app)

        if expose_entry:
            self._mount_service_routes(
                fastapi_app=fastapi_app,
                agent=agent,
                entry_path=entry_path,
                status_path=status_path,
                workflow_component=workflow_component,
                fetch_status_payloads=fetch_status_payloads,
            )

        self._mount_hitl_routes(fastapi_app=fastapi_app, agent=agent)

        auto_run = app is None
        if auto_run:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeError(
                    "AgentRunner.serve() cannot auto-run uvicorn inside an active event loop. "
                    "Pass your own FastAPI app and run it separately when calling from async code."
                )

            try:
                import uvicorn
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "uvicorn is required to auto-run AgentRunner.serve(); "
                    "install uvicorn or pass an existing FastAPI app."
                ) from exc

            uvicorn.run(
                fastapi_app,
                host=host,
                port=port,
                log_level=logging.getLevelName(logger.getEffectiveLevel()).lower(),
            )

        return fastapi_app

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return path

    def _mount_service_routes(
        self,
        *,
        fastapi_app: FastAPI,
        agent: DurableAgent,
        entry_path: str,
        status_path: str,
        workflow_component: str,
        fetch_status_payloads: bool,
    ) -> None:
        entry_path = self._normalize_path(entry_path)
        status_path = self._normalize_path(status_path)

        if "{instance_id}" not in status_path:
            raise ValueError("status_path must include '{instance_id}'.")

        if entry_path not in self._default_http_paths:
            self._default_http_paths.add(entry_path)

            async def _start_workflow(
                body: dict[str, str] = Body(default_factory=dict),
            ) -> dict[str, str]:
                payload = body or None
                instance_id = await self.run(
                    agent,
                    payload=payload,
                    wait=False,
                    log=True,
                )
                return {
                    "instance_id": instance_id if instance_id else "",
                    "status_url": status_path.replace(
                        "{instance_id}", instance_id or ""
                    ),
                }

            async def _terminate_workflow(instance_id: str) -> dict[str, str]:
                await asyncio.to_thread(
                    self._wf_client.terminate_workflow,
                    instance_id,
                    output="Terminated by user request",
                )
                if hasattr(agent, "mark_workflow_terminated"):
                    await asyncio.to_thread(
                        agent.mark_workflow_terminated,
                        instance_id,
                    )
                return {"instance_id": instance_id, "status": "terminated"}

            async def _purge_workflow(instance_id: str) -> dict[str, str]:
                self._wf_client.purge_workflow(
                    instance_id=instance_id,
                )
                purge_fn = getattr(agent, "purge", None)
                if callable(purge_fn):
                    purge_fn(instance_id)
                else:
                    agent._infra.purge_state(instance_id)
                    if getattr(agent, "memory", None) is not None:
                        agent.memory.purge_memory(instance_id)
                return {"instance_id": instance_id, "status": "purged"}

            stream_path = entry_path + "/stream"
            input_path = entry_path + "/input"

            async def _start_workflow_stream(
                request: Request,
                body: dict = Body(default_factory=dict),
            ) -> StreamingResponse:
                accept = (request.headers.get("accept") or "").lower()
                use_sse = "text/event-stream" in accept
                listener_cfg = body.pop("listener", None) if body else None
                # Block the custom dotted-import factory from HTTP callers;
                # arbitrary ``factory`` strings would trigger
                # ``importlib.import_module`` inside the framework process.
                # Trusted custom listeners must be pre-registered at startup
                # via ``register_stream_listener()`` and referenced by name.
                if isinstance(listener_cfg, Mapping):
                    if listener_cfg.get("type") == "custom":
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "Custom listener factories must be registered "
                                "at startup via register_stream_listener() and "
                                "referenced by name; dotted-import factory "
                                "configs are not accepted from HTTP callers."
                            ),
                        )
                include_complete = (
                    bool(body.pop("include_complete_message", False)) if body else False
                )
                iterator = self.run_stream(
                    agent,
                    payload=body or None,
                    listener=listener_cfg,
                    include_complete_message=include_complete,
                )
                formatter = _format_sse if use_sse else _format_ndjson
                media_type = "text/event-stream" if use_sse else "application/x-ndjson"

                async def event_source():
                    last_seq = 0
                    try:
                        async for chunk in iterator:
                            last_seq = chunk.sequence
                            yield formatter(chunk)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "Stream endpoint terminated with error: %s", exc
                        )
                        # Synthesize a terminal ERROR chunk so clients can
                        # distinguish "stream failed" from "stream ended
                        # cleanly". Attribution fields are minimal because
                        # the failure may have occurred before any real
                        # chunk arrived.
                        try:
                            error_chunk = _build_terminal_error_chunk(
                                sequence=last_seq + 1, exc=exc
                            )
                            yield formatter(error_chunk)
                        except Exception:  # noqa: BLE001
                            logger.exception("Failed to emit terminal ERROR chunk")

                return StreamingResponse(
                    event_source(),
                    media_type=media_type,
                    headers={
                        "Cache-Control": "no-store",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    },
                )

            async def _submit_input(
                body: dict = Body(...),
            ) -> dict[str, str]:
                try:
                    request_id = body["request_id"]
                    target_instance_id = body["target_instance_id"]
                    answer = body["answer"]
                except KeyError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing required field: {exc.args[0]}",
                    ) from exc
                if not isinstance(answer, str):
                    raise HTTPException(
                        status_code=400,
                        detail="'answer' must be a string",
                    )
                if len(answer) > ASK_USER_ANSWER_MAX_LEN:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"'answer' exceeds maximum length of "
                            f"{ASK_USER_ANSWER_MAX_LEN} characters"
                        ),
                    )
                await asyncio.to_thread(
                    self._wf_client.raise_workflow_event,
                    instance_id=target_instance_id,
                    event_name=f"{USER_INPUT_EVENT_PREFIX}:{request_id}",
                    data=answer,
                )
                return {
                    "request_id": request_id,
                    "target_instance_id": target_instance_id,
                    "status": "accepted",
                }

            fastapi_app.add_api_route(
                entry_path,
                _start_workflow,
                methods=["POST"],
                summary="Schedule workflow entry",
                tags=["workflow"],
            )
            fastapi_app.add_api_route(
                stream_path,
                _start_workflow_stream,
                methods=["POST"],
                summary="Schedule workflow entry and stream chunks (SSE or NDJSON via Accept)",
                tags=["workflow"],
            )
            fastapi_app.add_api_route(
                input_path,
                _submit_input,
                methods=["POST"],
                summary="Submit a UserInputResponse to a waiting agent",
                tags=["workflow"],
            )
            fastapi_app.add_api_route(
                status_path + "/terminate",
                _terminate_workflow,
                methods=["POST"],
                summary="Terminate workflow instance",
                tags=["workflow"],
            )
            fastapi_app.add_api_route(
                status_path + "/purge",
                _purge_workflow,
                methods=["POST"],
                summary="Purge agent instance",
                tags=["agent"],
            )

            logger.info("Mounted default agent run endpoint at %s", entry_path)
            logger.info("Mounted streaming endpoint at %s", stream_path)
            logger.info("Mounted input endpoint at %s", input_path)
        else:
            logger.debug("Workflow entry endpoint already mounted at %s", entry_path)

        if status_path in self._default_http_paths:
            logger.debug("Workflow status endpoint already mounted at %s", status_path)
            return

        self._default_http_paths.add(status_path)

        async def _get_status(instance_id: str) -> dict:
            state = await asyncio.to_thread(
                self._wf_client.get_workflow_state,
                instance_id,
                fetch_payloads=fetch_status_payloads,
            )
            if state is None:
                raise HTTPException(
                    status_code=404, detail="Workflow instance not found."
                )

            payload = state.to_json()
            payload["runtime_status"] = getattr(
                state.runtime_status, "name", str(state.runtime_status)
            )
            for field in ("created_at", "last_updated_at"):
                ts = payload.get(field)
                if ts:
                    payload[field] = ts.isoformat()
            # Surface outstanding ask_user prompts so a client that lost the
            # original USER_INPUT_REQUESTED chunk (e.g. after a restart) can
            # recover request_id + target_instance_id and answer via /input.
            try:
                entry = await asyncio.to_thread(agent._infra.get_state, instance_id)
                payload["pending_inputs"] = list(
                    (getattr(entry, "pending_inputs", None) or {}).values()
                )
            except Exception:  # noqa: BLE001
                payload["pending_inputs"] = []
            return payload

        fastapi_app.add_api_route(
            status_path,
            _get_status,
            methods=["GET"],
            summary="Get workflow status",
            tags=["workflow"],
        )
        logger.info("Mounted default workflow status endpoint at %s", status_path)

        # GET {status_path}/stream — reattach to an already-scheduled
        # session's pub/sub stream. Useful for page reloads where the
        # original ``POST /stream`` HTTP connection dropped but the
        # workflow is still running. In-process listener sessions cannot
        # be reattached (the queue is ephemeral and bound to the original
        # connection); those callers see 409.
        reconnect_path = status_path + "/stream"
        if reconnect_path not in self._default_http_paths:
            self._default_http_paths.add(reconnect_path)

            async def _reattach_stream(
                instance_id: str, request: Request
            ) -> StreamingResponse:
                state = await asyncio.to_thread(
                    self._wf_client.get_workflow_state,
                    instance_id,
                    fetch_payloads=False,
                )
                if state is None:
                    raise HTTPException(
                        status_code=404, detail="Workflow instance not found."
                    )

                # Reject terminal workflows early. Without this check,
                # ``astart`` succeeds (the pub/sub topic exists) but no
                # more chunks will arrive — the consumer would block
                # until the 30s subscription-ready watchdog fires, and
                # the client would see an empty stream that's
                # indistinguishable from a healthy "no activity yet"
                # session. Return 410 so callers can fall back to the
                # status endpoint to fetch the final message.
                runtime_status = getattr(state, "runtime_status", None)
                status_name = getattr(runtime_status, "name", str(runtime_status))
                if status_name in (
                    "COMPLETED",
                    "FAILED",
                    "TERMINATED",
                    "Completed",
                    "Failed",
                    "Terminated",
                ):
                    raise HTTPException(
                        status_code=410,
                        detail=(
                            "Workflow already in terminal state "
                            f"({status_name}); no stream to reattach. "
                            f"Fetch the final message via "
                            f"{status_path.replace('{instance_id}', instance_id)}."
                        ),
                    )

                topic_name = getattr(agent._infra, "stream_topic_name", None)
                pubsub_name = getattr(agent._infra, "message_bus_name", None)
                topic = topic_name(instance_id) if callable(topic_name) else None
                if not topic or not pubsub_name:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Reattach requires a pub/sub listener session; "
                            "in-process sessions cannot be reattached."
                        ),
                    )

                consumer = PubSubStreamConsumer(
                    pubsub_name=pubsub_name,
                    topic=topic,
                    root_instance_id=instance_id,
                )
                try:
                    await consumer.astart()
                except Exception:
                    await consumer.aclose()
                    raise

                accept = (request.headers.get("accept") or "").lower()
                use_sse = "text/event-stream" in accept
                formatter = _format_sse if use_sse else _format_ndjson
                media_type = "text/event-stream" if use_sse else "application/x-ndjson"

                async def event_source():
                    last_seq = 0
                    # Replay any outstanding ask_user prompts first: a client
                    # reconnecting after a restart never saw the original
                    # USER_INPUT_REQUESTED chunk (it was a cached activity and
                    # pub/sub doesn't retain it), so recover request_id +
                    # target_instance_id from durable state before tailing live.
                    try:
                        _entry = await asyncio.to_thread(
                            agent._infra.get_state, instance_id
                        )
                        _pending = getattr(_entry, "pending_inputs", None) or {}
                    except Exception:  # noqa: BLE001
                        _pending = {}
                    for _req_id, _data in _pending.items():
                        yield formatter(
                            AgentStreamChunk(
                                sequence=0,
                                chunk_id=uuid.uuid5(
                                    uuid.NAMESPACE_DNS, f"replay:{_req_id}"
                                ).hex,
                                type=StreamChunkType.USER_INPUT_REQUESTED,
                                agent=agent.name,
                                workflow_instance_id=instance_id,
                                turn=int(_data.get("turn", 0) or 0),
                                root_instance_id=instance_id,
                                event_data=dict(_data),
                                metadata={"replayed": True},
                            )
                        )
                    try:
                        async for chunk in consumer:
                            last_seq = chunk.sequence
                            yield formatter(chunk)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "Reattach stream terminated with error: %s", exc
                        )
                        try:
                            yield formatter(
                                _build_terminal_error_chunk(
                                    sequence=last_seq + 1, exc=exc
                                )
                            )
                        except Exception:  # noqa: BLE001
                            logger.exception("Failed to emit reattach ERROR chunk")
                    finally:
                        await consumer.aclose()

                return StreamingResponse(
                    event_source(),
                    media_type=media_type,
                    headers={
                        "Cache-Control": "no-store",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    },
                )

            fastapi_app.add_api_route(
                reconnect_path,
                _reattach_stream,
                methods=["GET"],
                summary="Reattach to an existing streaming session via pub/sub",
                tags=["workflow"],
            )
            logger.info("Mounted reconnect stream endpoint at %s", reconnect_path)

    def _mount_hitl_routes(
        self,
        *,
        fastapi_app: FastAPI,
        agent: DurableAgent,
    ) -> None:
        """
        Register the two HITL HTTP endpoints on the FastAPI app.

        GET  /hitl/approvals                            — list pending approval requests
        POST /hitl/approvals/{approval_request_id}/respond — submit a human decision

        These endpoints work regardless of whether pub/sub is configured. When
        pub/sub is not configured, polling GET /hitl/approvals is the only way a
        human reviewer discovers pending requests. When pub/sub IS configured, these
        endpoints are still registered as a secondary interface.

        For workflow-only agents (no HTTP server), responders can call the Dapr
        sidecar raiseEvent API directly instead of using these endpoints.
        """

        async def _list_approvals() -> List[Dict[str, Any]]:
            return agent.list_pending_approvals()

        async def _respond_to_approval(
            approval_request_id: str, body: _HitlRespondBody
        ) -> Dict[str, Any]:
            pending = agent._pending_approvals.get(approval_request_id)
            if pending is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Approval request '{approval_request_id}' not found. "
                        "It may have already been responded to, or this process was restarted. "
                        "You can still submit a response directly via the Dapr sidecar: "
                        "POST <sidecar-host>/v1.0-beta1/workflows/dapr/{instance_id}"
                        f"/raiseEvent/approval_response_{approval_request_id}"
                    ),
                )
            instance_id = pending.get("instance_id", "")
            agent.raise_approval_event(
                instance_id=instance_id,
                approval_request_id=approval_request_id,
                approved=body.approved,
                reason=body.reason,
            )
            return {
                "status": "ok",
                "approval_request_id": approval_request_id,
                "approved": body.approved,
            }

        fastapi_app.add_api_route(
            "/hitl/approvals",
            _list_approvals,
            methods=["GET"],
            summary="List pending human approval requests",
            tags=["hitl"],
        )
        fastapi_app.add_api_route(
            "/hitl/approvals/{approval_request_id}/respond",
            _respond_to_approval,
            methods=["POST"],
            summary="Submit a human approval decision",
            tags=["hitl"],
        )
        logger.info("Mounted HITL endpoints at /hitl/approvals")

    # ------------------------------------------------------------------
    # Activation hooks (extension seam)
    # ------------------------------------------------------------------
    def _attach_agent(self, agent: DurableAgent, app: Optional[FastAPI] = None) -> None:
        """Start, register, and activate an agent for hosting — once per agent.

        Shared by every host entry point (run/workflow/register_routes/subscribe/
        serve). It starts the agent's runtime, records it in ``_managed_agents``,
        and fires its activation callbacks exactly once. Re-hosting the same agent
        (e.g. the nested ``serve() -> subscribe()`` path, or a retry) is a no-op
        for the already-started/managed/activated agent.

        Callbacks run OUTSIDE ``self._lock`` so a callback may safely re-enter the
        runner; returned closers are tracked for teardown by ``shutdown()``. If
        activation fails, only what THIS call created is unwound (a pre-existing
        host or shared runtime is left intact) before the error is re-raised.

        Args:
            agent: The agent being hosted.
            app: FastAPI app exposed on the context, only under serve()/
                register_routes(); ``None`` otherwise.
        """
        started_here = False
        try:
            agent.start()
            started_here = True
        except RuntimeError:
            # Already started — idempotent; not ours to stop on rollback.
            pass

        with self._lock:
            if id(agent) in self._activated_agent_ids:
                return  # already attached; re-host (e.g. serve()->subscribe()) no-ops
            added_here = agent not in self._managed_agents
            if added_here:
                self._managed_agents.append(agent)
            callbacks = list(agent.activations)
            self._activated_agent_ids.add(id(agent))
            # Close the registration window so a late add_activation() fails loudly.
            agent._activation_window_open = False

        if not callbacks:
            return

        collected: List[Callable[[], None]] = []
        try:
            # Extensions may open a streaming subscription, so guarantee a Dapr
            # client even under workflow()/run(), which never wire pub/sub.
            self._ensure_dapr_client()
            if self._dapr_client is None:
                # Unreachable: _ensure_dapr_client() always sets the client or raises.
                raise RuntimeError(
                    "Dapr client unavailable after _ensure_dapr_client(); bug in AgentRunner."
                )
            context = ActivationContext(
                agent=agent,
                runner=self,
                dapr_client=self._dapr_client,
                wf_client=self._wf_client,
                app=app,
            )
            for callback in callbacks:
                label = getattr(callback, "__qualname__", repr(callback))
                try:
                    closer = callback(context)
                except Exception as exc:
                    raise RuntimeError(
                        f"Activation callback {label!r} for agent "
                        f"{getattr(agent, 'name', agent)!r} failed during hosting: {exc}"
                    ) from exc
                if closer is None:
                    continue
                if not callable(closer):
                    raise TypeError(
                        f"Activation callback {label!r} must return a callable "
                        f"closer or None, got {type(closer).__name__!r}"
                    )
                collected.append(closer)
        except Exception:
            # Unwind only what this call created, then re-raise.
            self._abort_attach(
                agent, collected, started_here=started_here, added_here=added_here
            )
            raise

        with self._lock:
            self._activation_closers[id(agent)] = collected

    def _abort_attach(
        self,
        agent: DurableAgent,
        collected: List[Callable[[], None]],
        *,
        started_here: bool,
        added_here: bool,
    ) -> None:
        """Roll back a failed attach: close collected closers, release the
        activation guard, and undo the managed/started state only if THIS call
        created it — so a pre-existing host or shared runtime is left intact."""
        self._rollback_activation_closers(collected)
        with self._lock:
            self._activated_agent_ids.discard(id(agent))
            if added_here and agent in self._managed_agents:
                self._managed_agents.remove(agent)
        if started_here:
            try:
                agent.stop()
            except Exception:
                logger.exception("Error stopping agent during attach rollback")

    def _rollback_activation_closers(self, closers: List[Callable[[], None]]) -> None:
        """Best-effort close (reverse order) of closers from a failed attach."""
        for close in reversed(closers):
            try:
                close()
            except Exception:
                logger.exception(
                    "Error while rolling back activation closer after failure"
                )

    def _close_activations(self, agent: Optional[DurableAgent] = None) -> None:
        """Invoke and clear activation teardown closers, resetting the guard.

        With no ``agent`` every tracked closer runs and the guard is cleared (full
        shutdown). With an ``agent`` only that agent's closers run and its guard
        entry is dropped, so it re-activates if hosted again.
        """
        with self._lock:
            if agent is None:
                groups = list(self._activation_closers.values())
                self._activation_closers.clear()
                self._activated_agent_ids.clear()
            else:
                groups = [self._activation_closers.pop(id(agent), [])]
                self._activated_agent_ids.discard(id(agent))
        for closers in groups:
            for close in reversed(closers):
                try:
                    close()
                except Exception:
                    logger.exception("Error while closing activation")

    def shutdown(self, agent: Optional[DurableAgent] = None) -> None:
        """
        Unwire subscriptions and close owned clients.

        Args:
            agent: Durable agent instance.

        Returns:
            None
        """

        if agent:
            # Shut down a single managed agent. Mutate shared state under the
            # lock, then run teardown (instrument/stop/closers) unlocked so a
            # self-locking _close_activations cannot deadlock.
            with self._lock:
                managed = agent in self._managed_agents
                if managed:
                    self._managed_agents.remove(agent)
                last = len(self._managed_agents) == 0
            if managed:
                try:
                    if agent.instrumentor is not None:
                        agent.instrumentor.uninstrument()
                except AttributeError:
                    # this happens if the agent has no instrumentor
                    pass
                agent.stop()  # This is safe as they'll return None if not started
                self._close_activations(agent)
            if last:
                try:
                    self.unwire_pubsub()
                    self._close_activations()
                finally:
                    self._close_wf_client()
                    self._close_dapr_client()
            return
        try:
            self.unwire_pubsub()
            self._close_activations()
        finally:
            with self._lock:
                agents = list(self._managed_agents)
            for ag in agents:
                try:
                    if ag.instrumentor is not None:
                        ag.instrumentor.uninstrument()
                except AttributeError:
                    # this happens if the agent has no instrumentor
                    pass
                ag.stop()
            self._close_wf_client()
            self._close_dapr_client()
