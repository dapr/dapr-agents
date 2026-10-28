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

"""One-shot lifecycle composition for agent-managed Drasi subscriptions."""

from __future__ import annotations

import logging
import math
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Callable, NoReturn

from dapr.clients import DaprClient
from dapr.clients.exceptions import DaprInternalError
from dapr.conf import settings
from dapr.ext.workflow import DaprWorkflowClient
from fastapi import FastAPI
from grpc import RpcError
from pydantic import ValidationError

from dapr_agents.agents.durable import DurableAgent
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners.agent import AgentRunner

from ._models import IntentDocument, ResolvedDrasiConfig, SubscriptionScope
from ._registration import (
    claim_dynamic_application,
    claim_dynamic_mode,
    validate_claimed_dynamic_mode,
    validate_dynamic_mode,
)
from .admission import DrasiAdmissionHandler
from .delivery import DrasiInbox, subscribe_drasi_inbox
from .intent_store import DaprIntentRepository
from .router_client import MCPRouterClient
from .subscription_manager import DrasiSubscriptionManager
from .subscription_tools import build_subscription_tools

logger = logging.getLogger(__name__)


class DrasiLifecycleError(RuntimeError):
    """Agent-managed Drasi lifecycle setup or cleanup failed."""


def _invalid(message: str) -> NoReturn:
    logger.error("%s", message)
    raise ValueError(message)


def _validate_supported_profile(agent: DurableAgent) -> None:
    if not isinstance(agent, DurableAgent):
        _invalid("Agent-managed Drasi subscriptions require a DurableAgent.")
    if agent.is_started:
        _invalid("Create the Drasi lifecycle before the agent is hosted or started.")
    # The core registration window stays closed after a runner stops the agent.
    if not agent._activation_window_open:
        _invalid("Create a fresh agent; this agent has already been hosted.")
    if agent.configuration is not None:
        _invalid(
            "Agent-managed Drasi subscriptions do not yet support "
            "RuntimeSubscriptionConfig. Construct the agent without configuration=."
        )
    if agent.activations:
        _invalid(
            "Agent-managed Drasi subscriptions do not yet support pre-existing "
            "activation callbacks on the same agent."
        )
    if agent.executor is not None or agent.orchestrator or agent.llm is None:
        _invalid(
            "Agent-managed Drasi subscriptions require the ordinary DurableAgent "
            "chat/tool loop, not executor= or orchestration mode."
        )


def _validate_new_agent(agent: DurableAgent) -> None:
    validate_dynamic_mode(agent)
    _validate_supported_profile(agent)


def _validate_claimed_agent(agent: DurableAgent) -> None:
    validate_claimed_dynamic_mode(agent)
    _validate_supported_profile(agent)


def _resolve_config(
    agent: DurableAgent,
    *,
    router_id: str,
    namespace: str,
    pubsub: str | None,
    dapr_http_port: int | None,
) -> tuple[ResolvedDrasiConfig, StateStoreService]:
    store = agent.state_store
    if not isinstance(store, StateStoreService):
        _invalid(
            "Drasi subscriptions require a configured AgentStateConfig.store; "
            "conversational memory is not durable subscription intent."
        )
    pubsub_name = agent.message_bus_name if pubsub is None else pubsub
    if pubsub_name is None:
        _invalid(
            "Drasi subscriptions require a Pub/Sub component. Configure the agent's "
            "bus or pass pubsub= explicitly."
        )

    port = dapr_http_port
    if port is None:
        try:
            port = int(settings.DAPR_HTTP_PORT)
        except (TypeError, ValueError):
            _invalid("DAPR_HTTP_PORT must be an integer TCP port.")
    try:
        config = ResolvedDrasiConfig(
            scope=SubscriptionScope(
                router_id=router_id,
                namespace=namespace,
                app_id=agent.appid,
                agent_name=agent.name,
            ),
            pubsub_name=pubsub_name,
            state_store_name=store.store_name,
            workflow_name=agent.agent_workflow_name,
            dapr_http_port=port,
        )
    except ValidationError:
        _invalid(
            "Invalid Drasi configuration. Supply router_id='<namespace>/<app-id>', "
            "the application's namespace, a valid Dapr application/agent identity, "
            "non-blank component names, and a TCP port from 1 to 65535."
        )

    if config.pubsub_name == agent.message_bus_name:
        framework_topics = {agent.topic_name, agent.broadcast_topic_name}
        if framework_topics.intersection(
            (config.scope.inbox_topic, config.scope.dead_letter_topic)
        ):
            _invalid(
                "The derived Drasi inbox and dead-letter topic must be distinct "
                "from the agent's framework topics on the same Pub/Sub component."
            )
    return config, store


def _check_sidecar(client: DaprClient, config: ResolvedDrasiConfig) -> None:
    try:
        metadata = client.get_metadata()
    except (RpcError, DaprInternalError, OSError) as error:
        logger.error(
            "Drasi sidecar metadata retrieval failed (%s).", type(error).__name__
        )
        raise DrasiLifecycleError(
            "Could not read Dapr metadata during Drasi preparation."
        ) from None
    if metadata.application_id != config.scope.app_id:
        _invalid("The Dapr sidecar application identity does not match the agent.")
    components = {
        component.name: component.type for component in metadata.registered_components
    }
    for name, kind in (
        (config.pubsub_name, "pubsub."),
        (config.state_store_name, "state."),
    ):
        if not components.get(name, "").startswith(kind):
            _invalid(f"Required Drasi {kind[:-1]} component {name!r} is not loaded.")


def _add_cleanup_notes(primary: BaseException, errors: list[Exception]) -> None:
    for error in errors:
        logger.error(
            "Drasi lifecycle cleanup failed (%s).",
            type(error).__name__,
            exc_info=(type(error), error, error.__traceback__),
        )
        primary.add_note(
            f"Drasi lifecycle cleanup failed: {type(error).__name__}: {error}"
        )


class DrasiSubscriptionLifecycle(AbstractAsyncContextManager[AgentRunner]):
    """Prepare, host, and close one dynamic Drasi-enabled agent lifecycle.

    The caller must supply a fresh ``DurableAgent`` constructed without
    ``runtime=`` and must not share or replace its workflow runtime. Current
    public Dapr Agents APIs do not expose runtime ownership for mechanical
    verification.
    """

    def __init__(
        self,
        agent: DurableAgent,
        *,
        router_id: str,
        namespace: str,
        app: FastAPI | None = None,
        workflow_client: DaprWorkflowClient | None = None,
        pubsub: str | None = None,
        dapr_http_port: int | None = None,
        router_timeout_seconds: float = 30.0,
        startup_timeout_seconds: float = 30.0,
    ) -> None:
        _validate_new_agent(agent)
        if not math.isfinite(startup_timeout_seconds) or startup_timeout_seconds <= 0:
            _invalid("startup_timeout_seconds must be finite and positive.")

        claim_dynamic_mode(agent)
        self._agent = agent
        self._router_id = router_id
        self._namespace = namespace
        self._app = app
        self._workflow_client = workflow_client
        self._owned_workflow_client: DaprWorkflowClient | None = None
        self._pubsub = pubsub
        self._dapr_http_port = dapr_http_port
        self._router_timeout_seconds = router_timeout_seconds
        self._startup_timeout_seconds = startup_timeout_seconds

        self._runner: AgentRunner | None = None
        self._runner_dapr_clients: list[DaprClient] = []
        self._router: MCPRouterClient | None = None
        self._inbox_client: DaprClient | None = None
        self._inbox: DrasiInbox | None = None
        self._release_application: Callable[[], None] | None = None
        self._hosting_attempted = False
        self._attempted = False
        self._runner_unwired = False
        self._runtime_drained = False
        self._cleanup_complete = False

    async def __aenter__(self) -> AgentRunner:
        if self._attempted:
            raise DrasiLifecycleError(
                "This Drasi lifecycle is one-shot; create fresh agent and runner objects."
            )
        if self._cleanup_complete:
            raise DrasiLifecycleError(
                "This Drasi lifecycle was closed before entry; create fresh agent "
                "and runner objects."
            )
        self._attempted = True
        try:
            _validate_claimed_agent(self._agent)
            await self._agent.connect_mcpservers()
            _validate_claimed_agent(self._agent)
            config, store = _resolve_config(
                self._agent,
                router_id=self._router_id,
                namespace=self._namespace,
                pubsub=self._pubsub,
                dapr_http_port=self._dapr_http_port,
            )
            self._release_application = claim_dynamic_application(self._agent)

            self._inbox_client = self._agent.client_factory()
            _check_sidecar(self._inbox_client, config)

            self._router = MCPRouterClient(
                config, timeout_seconds=self._router_timeout_seconds
            )
            catalog = self._router.list_queries()
            repository = DaprIntentRepository(scope=config.scope, store=store)
            manager = DrasiSubscriptionManager(
                repository=repository, router=self._router
            )
            snapshot = repository.load()
            tools = build_subscription_tools(
                catalog,
                manager,
                retained_query_ids=tuple(snapshot.document.intents)
                if snapshot is not None
                else (),
            )
            executor = self._agent.tool_executor
            for tool in tools:
                if executor.get_tool(tool.name) is not None:
                    _invalid(
                        f"Drasi tool {tool.name!r} conflicts with an existing tool. "
                        "Rename the existing tool before starting the lifecycle."
                    )

            _validate_claimed_agent(self._agent)
            current_config, current_store = _resolve_config(
                self._agent,
                router_id=self._router_id,
                namespace=self._namespace,
                pubsub=self._pubsub,
                dapr_http_port=self._dapr_http_port,
            )
            if current_config != config or current_store is not store:
                raise DrasiLifecycleError(
                    "Drasi identity or infrastructure changed during preparation."
                )

            if snapshot is None:
                repository.initialize(
                    IntentDocument(format_version=1, scope=config.scope, intents={})
                )
            manager.reconcile(catalog)

            for tool in tools:
                executor.register_tool(tool)
            if tools and self._agent.execution.tool_choice is None:
                self._agent.execution.tool_choice = "auto"

            def runner_client_factory() -> DaprClient:
                client = self._agent.client_factory()
                self._runner_dapr_clients.append(client)
                return client

            workflow_client = self._workflow_client
            if workflow_client is None:
                self._owned_workflow_client = DaprWorkflowClient()
                workflow_client = self._owned_workflow_client
            self._runner = AgentRunner(
                wf_client=workflow_client,
                client_factory=runner_client_factory,
            )
            self._hosting_attempted = True
            if self._app is None:
                self._runner.workflow(self._agent)
            else:
                self._runner.serve(self._agent, app=self._app)

            if not self._agent.runtime.wait_for_worker_ready(
                timeout=self._startup_timeout_seconds
            ):
                raise DrasiLifecycleError(
                    "The Dapr workflow worker did not become ready before the "
                    "Drasi startup deadline."
                )

            current_config, current_store = _resolve_config(
                self._agent,
                router_id=self._router_id,
                namespace=self._namespace,
                pubsub=self._pubsub,
                dapr_http_port=self._dapr_http_port,
            )
            if current_config != config or current_store is not store:
                raise DrasiLifecycleError(
                    "Drasi identity or infrastructure changed during startup."
                )
            if any(executor.get_tool(tool.name) is not tool for tool in tools):
                raise DrasiLifecycleError(
                    "The generated Drasi tool registrations changed during startup."
                )

            admission = DrasiAdmissionHandler(scope=config.scope, intents=repository)
            self._inbox = subscribe_drasi_inbox(
                config=config,
                admission=admission,
                dapr_client=self._inbox_client,
                workflow_client=self._runner.workflow_client(),
            )
            self._inbox_client = None
            return self._runner
        except BaseException as primary:
            _add_cleanup_notes(primary, self._cleanup())
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, traceback
        errors = self._cleanup()
        if exc is not None:
            _add_cleanup_notes(exc, errors)
            return False
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Drasi lifecycle cleanup failed.", errors)
        return False

    async def aclose(self) -> None:
        """Retry unresolved resource cleanup without rehosting the lifecycle."""
        errors = self._cleanup()
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Drasi lifecycle cleanup failed.", errors)

    def _cleanup(self) -> list[Exception]:
        if self._cleanup_complete:
            return []
        errors: list[Exception] = []

        if self._inbox is not None:
            try:
                self._inbox.close()
            except Exception as error:
                errors.append(error)
            if self._inbox.is_closed:
                self._inbox = None
            elif not self._inbox.is_stopped:
                return errors
        elif self._inbox_client is not None:
            try:
                self._inbox_client.close()
            except Exception as error:
                errors.append(error)
            else:
                self._inbox_client = None

        unresolved_runner_clients: list[DaprClient] = []
        for client in self._runner_dapr_clients:
            try:
                client.close()
            except Exception as error:
                errors.append(error)
                unresolved_runner_clients.append(client)
        self._runner_dapr_clients = unresolved_runner_clients

        if self._runner is not None and not self._runner_unwired:
            try:
                self._runner.unwire_pubsub()
            except Exception as error:
                errors.append(error)
                return errors
            else:
                self._runner_unwired = True
        if self._runner_dapr_clients:
            return errors

        if not self._hosting_attempted:
            self._runtime_drained = True
        elif not self._runtime_drained:
            try:
                self._agent.runtime.shutdown()
            except Exception as error:
                errors.append(error)
                return errors
            else:
                self._runtime_drained = True

        if self._runner is not None:
            try:
                self._runner.shutdown(self._agent if self._hosting_attempted else None)
            except Exception as error:
                errors.append(error)
                return errors
            else:
                self._runner = None

        if self._owned_workflow_client is not None:
            try:
                self._owned_workflow_client.close()
            except Exception as error:
                logger.error(
                    "Drasi workflow client close failed (%s).", type(error).__name__
                )
                errors.append(error)
                return errors
            else:
                self._owned_workflow_client = None

        if self._router is not None:
            try:
                self._router.close()
            except Exception as error:
                errors.append(error)
                return errors
            else:
                self._router = None

        if (
            self._inbox is not None
            or self._inbox_client is not None
            or self._runner_dapr_clients
            or self._runner is not None
            or self._owned_workflow_client is not None
            or self._router is not None
        ):
            return errors

        if self._release_application is not None:
            try:
                self._release_application()
            except Exception as error:
                errors.append(error)
                return errors
            else:
                self._release_application = None

        self._cleanup_complete = self._release_application is None
        return errors


def drasi_subscription_lifecycle(
    agent: DurableAgent,
    *,
    router_id: str,
    namespace: str,
    app: FastAPI | None = None,
    workflow_client: DaprWorkflowClient | None = None,
    pubsub: str | None = None,
    dapr_http_port: int | None = None,
    router_timeout_seconds: float = 30.0,
    startup_timeout_seconds: float = 30.0,
) -> DrasiSubscriptionLifecycle:
    """Create a one-shot context for agent-managed Drasi subscriptions."""
    return DrasiSubscriptionLifecycle(
        agent,
        router_id=router_id,
        namespace=namespace,
        app=app,
        workflow_client=workflow_client,
        pubsub=pubsub,
        dapr_http_port=dapr_http_port,
        router_timeout_seconds=router_timeout_seconds,
        startup_timeout_seconds=startup_timeout_seconds,
    )


__all__ = [
    "DrasiLifecycleError",
    "DrasiSubscriptionLifecycle",
    "drasi_subscription_lifecycle",
]
