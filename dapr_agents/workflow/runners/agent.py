from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Callable, Dict, Literal, Optional, TypeVar, Union

from fastapi import FastAPI

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
        config = getattr(agent, "pubsub", None)
        if config is None:
            logger.debug(
                "[%s] Agent %s has no pubsub; skipping pub/sub route registration.",
                self._name,
                getattr(agent, "name", agent),
            )
        else:
            specs = self._build_pubsub_specs(agent, config)
            if specs:
                self._ensure_dapr_client()

                if not self._wired_pubsub and self._dapr_client is not None:
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

        if fastapi_app is not None and not self._wired_http:
            register_http_routes(
                app=fastapi_app,
                targets=[agent],
                routes=None,
            )
            self._wired_http = True

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
