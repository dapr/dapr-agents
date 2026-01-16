from __future__ import annotations

import asyncio
import functools
import logging
import threading
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
    overload,
)

from dapr.clients import DaprClient
from dapr.ext.workflow import DaprWorkflowClient
from dapr.ext.workflow.workflow_state import WorkflowState
from fastapi import FastAPI

from dapr_agents.types.workflow import HttpRouteSpec, PubSubRouteSpec
from dapr_agents.utils.signal.mixin import SignalMixin
from dapr_agents.workflow.utils.registration import (
    register_http_routes,
    register_message_routes,
)

logger = logging.getLogger(__name__)


class WorkflowRunner(SignalMixin):
    """
    Host around DaprWorkflowClient with workflow scheduling and optional route wiring.

    Provides:
      • Sync/async workflow scheduling + completion waiting.
      • Optional wiring for `@message_router` (pub/sub) and `@http_router` (FastAPI).
      • Graceful shutdown and OS signal handling.
    """

    def __init__(
        self,
        *,
        name: str = "dapr-workflow-app",
        wf_client: Optional[DaprWorkflowClient] = None,
        timeout_in_seconds: int = 600,
        auto_install_signals: bool = False,
        dapr_client: Optional[DaprClient] = None,
    ) -> None:
        """
        Initialize the runner.

        Args:
            name: Logical name used in logs.
            wf_client: Existing DaprWorkflowClient. If omitted, a new client is created and owned.
            timeout_in_seconds: Default timeout when waiting for completion.
            auto_install_signals: Install SIGINT/SIGTERM handlers on context entry.
            dapr_client: Optional ready-to-use DaprClient. If omitted, a default one is created/owned.

        Returns:
            None
        """
        super().__init__()
        self._name = name
        self._wf_client: DaprWorkflowClient = wf_client or DaprWorkflowClient()
        self._wf_client_owned = wf_client is None
        self._timeout_in_seconds = timeout_in_seconds
        self._client_lock = threading.Lock()
        self._auto_install_signals = auto_install_signals
        self._signals_installed_by_us = False

        # Router wiring state
        self._dapr_client: Optional[DaprClient] = dapr_client
        self._dapr_client_owned: bool = dapr_client is None
        self._pubsub_closers: List[Callable[[], None]] = []
        self._wired_pubsub = False
        self._wired_http = False

    def __enter__(self) -> "WorkflowRunner":
        """
        Enter a synchronous context.

        Returns:
            Self, so callers can schedule workflows or register routes.
        """
        if self._auto_install_signals:
            self.install_signal_handlers()
            self._signals_installed_by_us = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Exit a synchronous context, removing signal handlers and shutting down.

        Args:
            exc_type: Exception type if any.
            exc: Exception instance if any.
            tb: Traceback if any.

        Returns:
            None
        """
        if exc_type:
            logger.error(
                "[%s] Context exited with exception",
                self._name,
                exc_info=(exc_type, exc, tb),
            )
        else:
            logger.debug("[%s] Context exited cleanly (sync).", self._name)

        try:
            if self._signals_installed_by_us:
                self.remove_signal_handlers()
                self._signals_installed_by_us = False
        finally:
            self.shutdown()

    async def __aenter__(self) -> "WorkflowRunner":
        """
        Enter an asynchronous context.

        Returns:
            Self, so callers can schedule workflows or register routes.
        """
        if self._auto_install_signals:
            self.install_signal_handlers()
            self._signals_installed_by_us = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """
        Exit an asynchronous context, removing signal handlers and shutting down.

        Args:
            exc_type: Exception type if any.
            exc: Exception instance if any.
            tb: Traceback if any.

        Returns:
            None
        """
        if exc_type:
            logger.error(
                "[%s] Async context exited with exception",
                self._name,
                exc_info=(exc_type, exc, tb),
            )
        else:
            logger.debug("[%s] Async context exited cleanly.", self._name)

        try:
            await self.graceful_shutdown()
        finally:
            if self._signals_installed_by_us:
                self.remove_signal_handlers()
                self._signals_installed_by_us = False
            self.shutdown()

    def __del__(self) -> None:
        """Best-effort GC close; never raises."""
        try:
            self.shutdown()
        except Exception:
            pass

    def shutdown(self) -> None:
        """
        Unwire subscriptions and close owned clients.

        Returns:
            None
        """
        try:
            self.unwire_pubsub()
        finally:
            self._close_wf_client()
            self._close_dapr_client()

    def _ensure_dapr_client(self) -> None:
        """
        Ensure a DaprClient exists; create and own a default client if absent.

        Returns:
            None
        """
        if self._dapr_client is None:
            self._dapr_client = DaprClient()
            self._dapr_client_owned = True

    def _close_dapr_client(self) -> None:
        """
        Close the DaprClient if it is owned by this runner.

        Returns:
            None
        """
        if self._dapr_client is not None and self._dapr_client_owned:
            try:
                self._dapr_client.close()
            except Exception:
                logger.debug("Ignoring error closing DaprClient", exc_info=True)
        self._dapr_client = None
        self._dapr_client_owned = False

    def _close_wf_client(self) -> None:
        """
        Close the DaprWorkflowClient if it is owned by this runner.

        Returns:
            None
        """
        if self._wf_client is not None and self._wf_client_owned:
            try:
                self._wf_client.close()
            except Exception:
                logger.debug(
                    "Ignoring error while closing DaprWorkflowClient", exc_info=True
                )

    def register_routes(
        self,
        *,
        targets: Optional[Iterable[Any]] = None,
        routes: Optional[Iterable[Union[PubSubRouteSpec, HttpRouteSpec]]] = None,
        delivery_mode: Literal["sync", "async"] = "sync",
        deduper: Optional[Any] = None,
        subscribe: Optional[Callable[..., Callable[[], None]]] = None,
        await_result: bool = False,
        await_timeout: Optional[int] = None,
        fetch_payloads: bool = True,
        log_outcome: bool = False,
        fastapi_app: Optional[FastAPI] = None,
    ) -> None:
        """
        Wire routes in one of two modes (mutually exclusive):

        1) Discovery mode: provide `targets` to auto-discover `@message_router`
        and `@http_router` handlers.
        2) Explicit mode: provide `routes` (list of PubSubRouteSpec | HttpRouteSpec)
        to create subscriptions/endpoints directly. In explicit mode, if a spec
        omits `message_model`/`request_model` and `handler_fn` is decorated, the
        decorator's schema is used; otherwise `dict`.

        Args:
            targets: Instances/functions to scan for decorator metadata. (Discovery mode)
            routes: Explicit route specs to wire. (Explicit mode)
            delivery_mode: "sync" blocks Dapr thread; "async" enqueues to a worker.
            deduper: Optional idempotency backend with `seen(key)` / `mark(key)`.
            subscribe: Optional custom subscriber (defaults to DaprClient.subscribe_with_handler).
            await_result: If True (sync only), wait for completion and ACK/NACK accordingly.
            await_timeout: Timeout (seconds) for completion wait when `await_result=True`.
            fetch_payloads: Include payloads when waiting for completion.
            log_outcome: Log final workflow outcome (awaited or detached).
            fastapi_app: If provided, mount HTTP endpoints.

        Returns:
            None

        Raises:
            ValueError: If both `targets` and `routes` are provided, or neither is provided.
            RuntimeError: If `delivery_mode="async"` without a running event loop.
        """
        self._ensure_dapr_client()

        use_targets = targets is not None
        use_routes = routes is not None

        if use_targets and use_routes:
            raise ValueError(
                "Provide either `targets` (discovery) OR `routes` (explicit), not both."
            )
        if not use_targets and not use_routes:
            raise ValueError(
                "You must provide `targets` (discovery) OR `routes` (explicit)."
            )

        # ---- Discovery mode (targets) ----
        if use_targets:
            if not self._wired_pubsub and self._dapr_client is not None:
                closers = register_message_routes(
                    dapr_client=self._dapr_client,
                    targets=targets or [],
                    routes=None,
                    delivery_mode=delivery_mode,
                    deduper=deduper,
                    subscribe=subscribe,
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
                    targets=targets or [],
                    routes=None,
                )
                self._wired_http = True
            return

        # ---- Explicit mode (routes) ----
        specs = list(routes or [])
        pubsub_specs = [r for r in specs if isinstance(r, PubSubRouteSpec)]
        http_specs = [r for r in specs if isinstance(r, HttpRouteSpec)]

        if pubsub_specs and not self._wired_pubsub and self._dapr_client is not None:
            closers = register_message_routes(
                routes=pubsub_specs,
                dapr_client=self._dapr_client,
                delivery_mode=delivery_mode,
                deduper=deduper,
                subscribe=subscribe,
                wf_client=self._wf_client,
                await_result=await_result,
                await_timeout=await_timeout,
                fetch_payloads=fetch_payloads,
                log_outcome=log_outcome,
            )
            self._pubsub_closers.extend(closers)
            self._wired_pubsub = True

        if http_specs and fastapi_app is not None and not self._wired_http:
            register_http_routes(
                routes=http_specs,
                app=fastapi_app,
            )
            self._wired_http = True

    def unwire_pubsub(self) -> None:
        """
        Unsubscribe all pub/sub handlers wired by this runner.

        Returns:
            None
        """
        for close in self._pubsub_closers:
            try:
                close()
            except Exception:
                logger.exception("Error while closing subscription")
        self._pubsub_closers.clear()
        self._wired_pubsub = False

    # -------------------- workflow scheduling APIs ----------------------

    def workflow_client(self) -> DaprWorkflowClient:
        """
        Get the underlying DaprWorkflowClient.

        Returns:
            DaprWorkflowClient: The active workflow client.
        """
        return self._wf_client

    def run_workflow(
        self,
        workflow: Callable[..., Any],
        payload: Optional[Union[str, Dict[str, Any]]] = None,
        instance_id: Optional[str] = None,
    ) -> str:
        """
        Schedule a registered workflow and return its instance id.

        Args:
            workflow: Callable pointing to a registered workflow.
            payload: Workflow input (dict or serialized string).
            instance_id: Optional explicit instance id; autogenerated if omitted.

        Returns:
            str: The new workflow instance id.

        Raises:
            ValueError: If `workflow` is not callable.
            Exception: Any error bubbling up from the Dapr client while scheduling.
        """
        if not callable(workflow):
            raise ValueError("workflow must be a callable (already registered).")

        chosen_id = instance_id or uuid.uuid4().hex
        logger.debug(
            "[%s] Scheduling workflow %s id=%s",
            self._name,
            getattr(workflow, "__name__", workflow),
            chosen_id,
        )

        try:
            with self._client_lock:
                result = self._wf_client.schedule_new_workflow(
                    workflow=workflow,
                    input=payload,
                    instance_id=chosen_id,
                )
            logger.debug("[%s] Scheduled workflow id=%s", self._name, result)
            return result
        except Exception as e:
            logger.error("[%s] Failed to schedule workflow: %s", self._name, str(e))
            raise

    @overload
    async def run_workflow_async(
        self,
        workflow: Callable[..., Any],
        payload: Optional[Union[str, Dict[str, Any]]] = ...,
        instance_id: Optional[str] = ...,
        *,
        timeout_in_seconds: Optional[int] = ...,
        fetch_payloads: bool = ...,
        detach: Literal[True],
        log: bool = ...,
    ) -> str:
        ...

    @overload
    async def run_workflow_async(
        self,
        workflow: Callable[..., Any],
        payload: Optional[Union[str, Dict[str, Any]]] = ...,
        instance_id: Optional[str] = ...,
        *,
        timeout_in_seconds: Optional[int] = ...,
        fetch_payloads: bool = ...,
        detach: Literal[False] = ...,
        log: bool = ...,
    ) -> Optional[str]:
        ...

    async def run_workflow_async(
        self,
        workflow: Callable[..., Any],
        payload: Optional[Union[str, Dict[str, Any]]] = None,
        instance_id: Optional[str] = None,
        *,
        timeout_in_seconds: Optional[int] = None,
        fetch_payloads: bool = True,
        detach: bool = False,
        log: bool = False,
    ) -> Union[str, Optional[str]]:
        """
        Schedule a workflow and optionally wait for completion.

        Args:
            workflow: Callable pointing to a registered workflow.
            payload: Workflow input (dict or serialized string).
            instance_id: Optional explicit instance id; autogenerated if omitted.
            timeout_in_seconds: Wait timeout when `detach=False` (defaults to Runner's timeout).
            fetch_payloads: Include payloads when waiting for completion.
            detach: If True, return instance id immediately; otherwise, wait and return output.
            log: If True, log the final outcome (COMPLETED/FAILED).

        Returns:
            str | None:
              - If `detach=True`: the workflow instance id.
              - If `detach=False`: the serialized output string, or `None` on timeout/error.
        """
        schedule = functools.partial(self.run_workflow, workflow, payload, instance_id)
        instance = await asyncio.to_thread(schedule)

        effective_timeout = timeout_in_seconds or self._timeout_in_seconds

        if detach:
            logger.info("[%s] Running in detached mode", self._name)
            if log:
                asyncio.create_task(
                    self._await_and_log_state(
                        instance, effective_timeout, fetch_payloads
                    )
                )
            return instance

        logger.info("[%s] Waiting for workflow completion...", self._name)
        state = await self._await_state(instance, effective_timeout, fetch_payloads)

        if log:
            self._log_state(instance, state)
        return getattr(state, "serialized_output", None) if state else None

    def wait_for_workflow_completion(
        self,
        instance_id: str,
        *,
        fetch_payloads: bool = True,
        timeout_in_seconds: Optional[int] = None,
    ) -> Optional[WorkflowState]:
        """
        Block until a workflow completes and return its final state.

        Args:
            instance_id: Workflow instance id to wait on.
            fetch_payloads: Include payloads in the returned state.
            timeout_in_seconds: Per-call timeout (defaults to Runner's timeout).

        Returns:
            WorkflowState | None: Final state, or None on timeout/error.
        """
        effective_timeout = timeout_in_seconds or self._timeout_in_seconds
        try:
            with self._client_lock:
                return self._wf_client.wait_for_workflow_completion(
                    instance_id,
                    fetch_payloads=fetch_payloads,
                    timeout_in_seconds=effective_timeout,
                )
        except Exception as exc:
            logger.error("Error while waiting for %s completion: %s", instance_id, exc)
            return None

    # ----------------------- internal helpers ---------------------------

    async def _await_state(
        self,
        instance_id: str,
        timeout_in_seconds: int,
        fetch_payloads: bool,
    ) -> Optional[WorkflowState]:
        """
        Await a workflow's completion using a thread offload.

        Args:
            instance_id: Workflow instance id.
            timeout_in_seconds: Timeout in seconds.
            fetch_payloads: Include payloads.

        Returns:
            WorkflowState | None: Final state, or None on timeout/error.
        """

        def _wait() -> Optional[WorkflowState]:
            with self._client_lock:
                return self._wf_client.wait_for_workflow_completion(
                    instance_id,
                    fetch_payloads=fetch_payloads,
                    timeout_in_seconds=timeout_in_seconds,
                )

        return await asyncio.to_thread(_wait)

    async def _await_and_log_state(
        self,
        instance_id: str,
        timeout_in_seconds: int,
        fetch_payloads: bool,
    ) -> None:
        """
        Await and log a workflow's final state (fire-and-forget).

        Args:
            instance_id: Workflow instance id.
            timeout_in_seconds: Timeout in seconds.
            fetch_payloads: Include payloads.

        Returns:
            None
        """
        try:
            state = await self._await_state(
                instance_id, timeout_in_seconds, fetch_payloads
            )
            self._log_state(instance_id, state)
        except Exception:
            logger.exception(
                "[%s] %s: error while monitoring workflow outcome",
                self._name,
                instance_id,
            )

    def _log_state(self, instance_id: str, state: Optional[WorkflowState]) -> None:
        """
        Compact logger for final workflow state.

        Args:
            instance_id: Workflow instance id.
            state: Final state (may be None on timeout/error).

        Returns:
            None
        """
        if not state:
            logger.warning(
                "[%s] %s: no state returned (timeout or missing).",
                self._name,
                instance_id,
            )
            return

        status = getattr(state.runtime_status, "name", str(state.runtime_status))
        if status == "COMPLETED":
            logger.info(
                "[%s] %s completed. Final Output=%s",
                self._name,
                instance_id,
                getattr(state, "serialized_output", None),
            )
            return

        fd = getattr(state, "failure_details", None)
        if fd:
            logger.error(
                "[%s] %s: FAILED. type=%s message=%s\n%s",
                self._name,
                instance_id,
                getattr(fd, "error_type", None),
                getattr(fd, "message", None),
                getattr(fd, "stack_trace", "") or "",
            )
        else:
            logger.error(
                "[%s] %s: finished with status=%s. custom_status=%s",
                self._name,
                instance_id,
                status,
                getattr(state, "serialized_custom_status", None),
            )

    # ----------------------- admin utilities ----------------------------

    def terminate_workflow(
        self,
        instance_id: str,
        *,
        output: Optional[Any] = None,
    ) -> None:
        """
        Terminate a running workflow.

        Args:
            instance_id: Workflow instance ID to terminate.
            output: Optional output to set for the terminated workflow.

        Returns:
            None
        """
        with self._client_lock:
            self._wf_client.terminate_workflow(instance_id=instance_id, output=output)
