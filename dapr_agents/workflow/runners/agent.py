from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Callable, Dict, Literal, Optional, TypeVar, Union

from fastapi import Body, FastAPI, HTTPException

from dapr_agents.types.workflow import PubSubRouteSpec
from dapr_agents.workflow.runners.base import WorkflowRunner
from dapr_agents.workflow.utils.core import get_decorated_methods
from dapr_agents.workflow.utils.registration import (
    register_http_routes,
    register_message_routes,
)

logger = logging.getLogger(__name__)

R = TypeVar("R")


def workflow_entry(func: Callable[..., R]) -> Callable[..., R]:
    """
    Mark a method/function as the workflow entrypoint for an Agent.

    This decorator does not wrap the function; it simply annotates the callable
    with `_is_workflow_entry = True` so AgentRunner can discover it on the agent
    instance via reflection.

    Usage:
        class MyAgent:
            @workflow_entry
            def my_workflow(self, ctx: DaprWorkflowContext, wf_input: dict) -> str:
                ...

    Returns:
        The same callable (unmodified), with an identifying attribute.
    """
    setattr(func, "_is_workflow_entry", True)  # type: ignore[attr-defined]
    return func


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
    ) -> None:
        """
        Initialize an AgentRunner.

        Args:
            name: Logical name used in logs (defaults to "agent-runner").
            wf_client: Optional injected DaprWorkflowClient. If omitted, a new one is created.
            timeout_in_seconds: Default timeout used when waiting for workflow completion.
            auto_install_signals: If True, installs SIGINT/SIGTERM handlers automatically
                when used as a context manager (with/async with) and removes them on exit.
        """
        super().__init__(
            name=name,
            wf_client=wf_client,
            timeout_in_seconds=timeout_in_seconds,
            auto_install_signals=auto_install_signals,
        )
        self._default_http_paths: set[str] = set()

    async def run(
        self,
        agent: Any,
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

    def run_sync(
        self,
        agent: Any,
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

    def discover_entry(self, agent: Any) -> Callable[..., Any]:
        """
        Locate exactly one bound method on `agent` marked with `@workflow_entry`.

        Returns:
            The bound method to schedule.

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

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return fut.result()

    def register_routes(
        self,
        agent: Any,
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

    def _build_pubsub_specs(self, agent: Any, config: Any) -> list[PubSubRouteSpec]:
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

            specs.append(
                PubSubRouteSpec(
                    pubsub_name=config.pubsub_name,
                    topic=topic,
                    handler_fn=handler,
                    message_model=message_model,
                )
            )

        return specs

    def _wire_pubsub_routes(
        self,
        *,
        agent: Any,
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
        )
        self._pubsub_closers.extend(closers)
        self._wired_pubsub = True

    def _wire_http_routes(self, *, agent: Any, fastapi_app: Optional[FastAPI]) -> None:
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
        agent: Any,
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
        agent: Any,
        *,
        app: Optional[FastAPI] = None,
        host: str = "0.0.0.0",
        port: int = 8001,
        expose_entry: bool = True,
        entry_path: str = "/run",
        status_path: str = "/run/{instance_id}",
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

            uvicorn.run(fastapi_app, host=host, port=port)

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
        agent: Any,
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

            async def _start_workflow(body: dict = Body(default_factory=dict)) -> dict:
                payload = body or None
                instance_id = await self.run(
                    agent,
                    payload=payload,
                    wait=False,
                    log=True,
                )
                return {
                    "instance_id": instance_id,
                    "status_url": f"/v1.0/workflows/{workflow_component}/{instance_id}",
                }

            fastapi_app.add_api_route(
                entry_path,
                _start_workflow,
                methods=["POST"],
                summary="Schedule workflow entry",
                tags=["workflow"],
            )
            logger.info("Mounted default workflow entry endpoint at %s", entry_path)
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
            return payload

        fastapi_app.add_api_route(
            status_path,
            _get_status,
            methods=["GET"],
            summary="Get workflow status",
            tags=["workflow"],
        )
        logger.info("Mounted default workflow status endpoint at %s", status_path)
