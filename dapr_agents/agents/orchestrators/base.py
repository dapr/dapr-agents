from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, Dict, Optional, Callable

import dapr.ext.workflow as wf

from dapr_agents.agents.components import AgentComponents
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    WorkflowGrpcOptions,
    StateModelBundle,
)
from dapr_agents.agents.utils.text_printer import ColorTextFormatter
from dapr_agents.workflow.utils.grpc import apply_grpc_options

logger = logging.getLogger(__name__)


class OrchestratorBase(AgentComponents):
    """
    Workflow-native orchestrator base built on AgentComponents.

    Overview:
        Manages workflow runtime lifecycle (register/start/stop), optional
        self-registration in the agent registry (marked as orchestrator),
        console helpers for readable interactions, and small utilities like
        raising workflow events.
    """

    def __init__(
        self,
        *,
        name: str,
        pubsub: Optional[AgentPubSubConfig] = None,
        state: Optional[AgentStateConfig] = None,
        registry: Optional[AgentRegistryConfig] = None,
        execution: Optional[AgentExecutionConfig] = None,
        agent_metadata: Optional[Dict[str, Any]] = None,
        workflow_grpc: Optional[WorkflowGrpcOptions] = None,
        runtime: Optional[wf.WorkflowRuntime] = None,
        workflow_client: Optional[wf.DaprWorkflowClient] = None,
        default_bundle: Optional[StateModelBundle] = None,
        final_summary_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(
            name=name,
            pubsub=pubsub,
            state=state,
            registry=registry,
            workflow_grpc_options=workflow_grpc,
            default_bundle=default_bundle,
        )

        self._final_summary_callback = final_summary_callback

        self.execution: AgentExecutionConfig = execution or AgentExecutionConfig()
        try:
            self.execution.max_iterations = max(1, int(self.execution.max_iterations))
        except Exception:
            self.execution.max_iterations = 10

        # Ensure registry entry marks this as an orchestrator
        meta = dict(agent_metadata or {})
        meta.setdefault("orchestrator", True)
        if self.registry_state is not None:
            try:
                self.register_agentic_system(metadata=meta)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Could not register orchestrator in registry.", exc_info=True
                )

        # Runtime wiring
        apply_grpc_options(self.workflow_grpc_options)

        self._runtime: wf.WorkflowRuntime = runtime or wf.WorkflowRuntime()
        self._runtime_owned = runtime is None
        self._registered = False
        self._started = False
        self._workflow_client = workflow_client or wf.DaprWorkflowClient()

        # Presentation helper (console)
        self._text_formatter = ColorTextFormatter()

    # ------------------------------------------------------------------
    # Callback-safe helper method
    # ------------------------------------------------------------------
    def _invoke_final_summary_callback(self, summary: str) -> None:
        """
        Invoke the user-supplied final summary callback (if any).

        This MUST be called only during non-replay execution paths.
        """
        cb = getattr(self, "_final_summary_callback", None)
        if cb and callable(cb):
            try:
                cb(summary)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Final summary callback failed: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def runtime(self) -> wf.WorkflowRuntime:
        """Return the underlying workflow runtime."""
        return self._runtime

    @property
    def is_started(self) -> bool:
        """Return True if the runtime has been started by this orchestrator."""
        return self._started

    def register(self, runtime: wf.WorkflowRuntime) -> None:
        """
        Register workflows/activities onto an external runtime.

        Subclasses must implement `register_workflows(runtime)` to perform registrations.
        """
        self._runtime = runtime
        self._runtime_owned = False
        self.register_workflows(runtime)
        self._registered = True

    def start(
        self,
        runtime: Optional[wf.WorkflowRuntime] = None,
        *,
        auto_register: bool = True,
    ) -> None:
        """
        Start the workflow runtime and register workflows/activities if needed.

        Behavior:
        • If a runtime is provided, attach to it (we still consider it not owned).
        • Register workflows once (if not already).
        • Always attempt to start the runtime; treat start() as idempotent:
            - If it's already running, swallow/log the exception and continue.
        • We only call shutdown() later if we own the runtime.
        """
        if self._started:
            raise RuntimeError("Orchestrator has already been started.")

        if runtime is not None:
            self._runtime = runtime
            self._runtime_owned = False
            self._registered = False
            logger.info(
                "Attached injected WorkflowRuntime (owned=%s).", self._runtime_owned
            )

        if auto_register and not self._registered:
            self.register_workflows(self._runtime)
            self._registered = True
            logger.info("Registered workflows/activities on WorkflowRuntime.")

        try:
            self._runtime.start()
            logger.info("WorkflowRuntime started (owned=%s).", self._runtime_owned)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "WorkflowRuntime.start() raised (likely already running): %s",
                exc,
                exc_info=True,
            )

        self._started = True

    def stop(self) -> None:
        """Stop the workflow runtime if owned by this instance."""
        if not self._started:
            return

        if self._runtime_owned:
            try:
                self._runtime.shutdown()
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Error while shutting down orchestrator runtime", exc_info=True
                )

        self._started = False

    # ------------------------------------------------------------------
    # Registration hook
    # ------------------------------------------------------------------
    def register_workflows(
        self, runtime: wf.WorkflowRuntime
    ) -> None:  # pragma: no cover
        """
        Hook for subclasses to register workflows and activities.

        Example:
            runtime.register_workflow(self.my_workflow)
            runtime.register_activity(self.my_activity)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Presentation helpers (console)
    # ------------------------------------------------------------------
    @property
    def text_formatter(self) -> ColorTextFormatter:
        """Formatter used for human-friendly console output."""
        return self._text_formatter

    @text_formatter.setter
    def text_formatter(self, formatter: ColorTextFormatter) -> None:
        """Override the default text formatter."""
        self._text_formatter = formatter

    def print_interaction(
        self, source_agent_name: str, target_agent_name: str, message: str
    ) -> None:
        """
        Print a formatted interaction between two agents.

        Args:
            source_agent_name: Sender name.
            target_agent_name: Recipient name.
            message: Message content.
        """
        separator = "-" * 80
        parts = [
            (source_agent_name, "dapr_agents_pink"),
            (" -> ", "dapr_agents_teal"),
            (f"{target_agent_name}\n\n", "dapr_agents_pink"),
            (message + "\n\n", "dapr_agents_pink"),
            (separator + "\n", "dapr_agents_teal"),
        ]
        self._text_formatter.print_colored_text(parts)

    # ------------------------------------------------------------------
    # Team/registry convenience
    # ------------------------------------------------------------------
    def list_team_agents(
        self, *, team: Optional[str] = None, include_self: bool = True
    ) -> Dict[str, Any]:
        """
        Convenience wrapper over `get_agents_metadata`.

        Args:
            team: Team override.
            include_self: If True, include this orchestrator/agent in the results.

        Returns:
            Mapping of agent name to metadata.
        """
        return self.get_agents_metadata(
            exclude_self=not include_self,
            exclude_orchestrator=False,
            team=team,
        )

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------
    def raise_workflow_event(
        self, instance_id: str, event_name: str, *, data: Any | None = None
    ) -> None:
        """
        Raise an external event for a running workflow instance.

        Args:
            instance_id: Target workflow instance id.
            event_name: Name of the event to raise.
            data: Optional payload. If it is a Pydantic-like object with ``model_dump``,
                it will be serialized to a dict.

        Raises:
            RuntimeError: If raising the event fails.
        """
        try:
            payload = self._serialize_event_data(data)
            logger.info(
                "Raising workflow event '%s' for instance '%s'", event_name, instance_id
            )
            self._workflow_client.raise_workflow_event(
                instance_id=instance_id,
                event_name=event_name,
                data=payload,
            )
            logger.info(
                "Raised workflow event '%s' for instance '%s'", event_name, instance_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to raise workflow event '%s' for instance '%s'. Data=%s Error=%s",
                event_name,
                instance_id,
                data,
                exc,
            )
            raise RuntimeError(
                f"Failed to raise workflow event '{event_name}' for instance '{instance_id}': {exc}"
            ) from exc

    @staticmethod
    def _serialize_event_data(data: Any | None) -> Any:
        """
        Best-effort serialization for event data.

        Args:
            data: Arbitrary event payload.

        Returns:
            A JSON-serializable payload. Pydantic-like objects are converted via ``model_dump``.
        """
        if data is None:
            return None
        if hasattr(data, "model_dump"):
            try:
                return data.model_dump()
            except Exception:  # noqa: BLE001
                return data  # fallback; Dapr client will attempt serialization
        return data

    # ------------------------------------------------------------------
    # Small async helper (shared by orchestrators)
    # ------------------------------------------------------------------
    @staticmethod
    def _run_asyncio_task(coro: Coroutine[Any, Any, Any]) -> Any:
        """
        Execute an async coroutine from a sync context, creating a fresh loop if needed.

        Args:
            coro: The coroutine to execute.

        Returns:
            The result of the coroutine execution.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
