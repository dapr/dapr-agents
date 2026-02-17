from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, Coroutine
from dapr_agents.agents.schemas import AgentWorkflowMessage, ConversationSummary

from dapr.clients import DaprClient
from dapr.clients.grpc._response import (
    GetMetadataResponse,
    RegisteredComponents,
    StateResponse,
    GetBulkSecretResponse,
    ConfigurationResponse,
)

from dapr_agents.agents.components import DaprInfra
from dapr_agents.agents.configs import (
    AgentConfigurationConfig,
    AgentLoggingExporter,
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    AgentExecutionConfig,
    AgentTracingExporter,
    WorkflowGrpcOptions,
    DEFAULT_AGENT_WORKFLOW_BUNDLE,
    AgentObservabilityConfig,
)
from dapr_agents.agents.prompting import AgentProfileConfig, PromptingAgentBase
from dapr_agents.agents.utils.text_printer import ColorTextFormatter
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.utils.defaults import get_default_llm
from dapr_agents.memory import ConversationDaprStateMemory, ConversationListMemory
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.storage.daprstores.stateservice import (
    StateStoreError,
    StateStoreService,
)
from dapr_agents.tool.base import AgentTool
from dapr_agents.tool.executor import AgentToolExecutor
from dapr_agents.types import (
    AgentError,
    AssistantMessage,
    ToolExecutionRecord,
    UserMessage,
)

from opentelemetry import trace
from opentelemetry import _logs
from opentelemetry.sdk._logs import LoggingHandler
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as OTLPGrpcSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter as OTLPGrpcLogExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as OTLPHTTPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter as OTLPHTTPLogExporter,
)
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogRecordExporter,
)
from dapr_agents.observability import DaprAgentsInstrumentor

logger = logging.getLogger(__name__)


class AgentBase:
    """
    Base class for agent behavior.

    Responsibilities:
    - Profile/prompt wiring (system prompt, instructions, style, template).
    - LLM client wiring.
    - Tool exposure and execution adapter.
    - Conversation memory management (configurable; defaults provided).

    Infrastructure (pub/sub, durable state, registry) is provided by `DaprInfra`.
    """

    def __init__(
        self,
        *,
        # Profile / prompt
        profile: Optional[AgentProfileConfig] = None,
        name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        instructions: Optional[Iterable[str]] = None,
        style_guidelines: Optional[Iterable[str]] = None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[PromptTemplateBase] = None,
        # Components (infrastructure)
        pubsub: Optional[AgentPubSubConfig] = None,
        state: Optional[AgentStateConfig] = None,
        registry: Optional[AgentRegistryConfig] = None,
        base_metadata: Optional[Dict[str, Any]] = None,
        max_etag_attempts: int = 10,
        # Memory / runtime
        memory: Optional[AgentMemoryConfig] = None,
        llm: Optional[ChatClientBase] = None,
        tools: Optional[Iterable[Any]] = None,
        # Metadata
        agent_metadata: Optional[Dict[str, Any]] = None,
        workflow_grpc: Optional[WorkflowGrpcOptions] = None,
        # Execution
        execution: Optional[AgentExecutionConfig] = None,
        agent_observability: Optional[AgentObservabilityConfig] = None,
        configuration: Optional[AgentConfigurationConfig] = None,
    ) -> None:
        """
        Initialize an agent with behavior + infrastructure.

        Args:
            profile: Base profile config (name/role/goal/prompts). Optional if
                individual fields are provided below.
            name: Agent name (required if `profile` is omitted).
            role: Agent role (e.g., "Assistant").
            goal: High-level agent objective.
            instructions: Additional instruction strings for the prompt.
            style_guidelines: Style directives for the prompt.
            system_prompt: System prompt override.
            prompt_template: Optional explicit prompt template instance.

            pubsub: Pub/Sub config used by `AgentComponents`.
            state: Durable state config used by `AgentComponents`.
            registry: Team registry config used by `AgentComponents`.
            execution: Execution dials for the agent run.
            base_metadata: Default Dapr state metadata used by `AgentComponents`.
            max_etag_attempts: Concurrency retry count for registry mutations.

            memory: Memory backend configuration. If omitted and a state store
                is configured, a Dapr-backed conversation memory is created by default.
            llm: Chat client. Defaults to `get_default_llm()`.
            tools: Optional tool callables or `AgentTool` instances.

            agent_metadata: Extra metadata to store in the registry.
            workflow_grpc: Optional gRPC overrides for the workflow runtime channel.
            execution: Execution dials for the agent run.
            agent_observability: Observability configuration for tracing/logging.
            configuration: Optional configuration store settings for hot-reloading.
        """
        # Resolve and validate profile (ensures non-empty name).
        resolved_profile = self._build_profile(
            base_profile=profile,
            name=name,
            role=role,
            goal=goal,
            instructions=instructions,
            style_guidelines=style_guidelines,
            system_prompt=system_prompt,
        )
        self.profile = resolved_profile
        self.name = resolved_profile.name  # type: ignore[assignment]

        self._runtime_secrets: Dict[str, str] = {}
        self._runtime_conf: Dict[str, str] = {}
        self._agent_observability = agent_observability
        self.configuration = configuration
        self._subscription_id: Optional[str] = None
        self.appid = (
            None  # We set the appid to None as standalone agents may not have one
        )

        try:
            with DaprClient(http_timeout_seconds=10) as _client:
                resp: GetMetadataResponse = _client.get_metadata()
                self.appid = resp.application_id
                components: Sequence[RegisteredComponents] = resp.registered_components
                for component in components:
                    if (
                        "state" in component.type
                        and component.name == "agent-memory"
                        and memory is None
                    ):
                        memory = AgentMemoryConfig(
                            store=ConversationDaprStateMemory(
                                store_name=component.name,
                                agent_name=self.name,
                            )
                        )
                    if "conversation" in component.type and llm is None:
                        # We got a default LLM component registered
                        logger.debug(f"LLM component found: {component.name}")
                        llm = get_default_llm()
                        if hasattr(llm, "component_name"):
                            llm.component_name = component.name  # type: ignore[attr-defined]

                    if (
                        "state" in component.type
                        and component.name == "agent-workflow"
                        and state is None
                    ):
                        state = AgentStateConfig(
                            store=StateStoreService(store_name=component.name),
                            state_key_prefix=f"{name.replace(' ', '-').lower() if name else 'default'}:_workflow",
                        )
                    if (
                        "state" in component.type
                        and component.name == "agent-registry"
                        and registry is None
                    ):
                        registry = AgentRegistryConfig(
                            store=StateStoreService(store_name=component.name),
                            team_name="default",
                        )
                    if (
                        "state" in component.type
                        and component.name == "agent-runtimestatestore"
                    ):
                        raw_runtime_conf: StateResponse = _client.get_state(
                            store_name=component.name,
                            key="agent_runtime",
                        )
                        try:
                            self._runtime_conf = (
                                json.loads(raw_runtime_conf.data)
                                if raw_runtime_conf.data
                                else {}
                            )
                            for key, value in self._runtime_conf.items():
                                logger.debug(f"Runtime configuration: {key}={value}")
                        except json.JSONDecodeError:
                            logger.warning(
                                "Failed to decode agent runtime configuration JSON. Using empty configuration."
                            )
                    if (
                        "pubsub" in component.type
                        and component.name == "agent-pubsub"
                        and pubsub is None
                    ):
                        logger.debug(f"topic: {name}.topic")
                        pubsub = AgentPubSubConfig(
                            pubsub_name=component.name,
                            agent_topic=f"{name.replace(' ', '-').lower()}.topic",
                            broadcast_topic="agents.broadcast",
                        )
                    if (
                        "secretstores" in component.type
                        and component.name == "agent-secretstore"
                    ):
                        try:
                            agent_secrets: GetBulkSecretResponse = (
                                _client.get_bulk_secret(store_name=component.name)
                            )
                            logger.debug(
                                f"Retrieved {len(agent_secrets.secrets.keys())} secrets from secret store."
                            )
                            for key, value in agent_secrets.secrets.items():
                                # Since dapr returns a nested dict we flatten it here
                                for _, v in value.items():
                                    self._runtime_secrets[key] = v
                        except Exception:
                            logger.warning(
                                "Failed to retrieve agent secrets. Skipping..."
                            )

        except TimeoutError:
            logger.warning(
                "Dapr sidecar not responding; proceeding without auto-configuration."
            )

        # Wire infrastructure via DaprInfra (composition).
        self._infra = DaprInfra(
            name=self.name,
            pubsub=pubsub,
            state=state,
            registry=registry,
            base_metadata=base_metadata,
            max_etag_attempts=max_etag_attempts,
            default_bundle=DEFAULT_AGENT_WORKFLOW_BUNDLE,
            workflow_grpc_options=workflow_grpc,
        )

        self.instrumentor: Optional[DaprAgentsInstrumentor] = None
        self._setup_agent_runtime_configuration()

        # -----------------------------
        # Memory wiring
        # -----------------------------
        self._memory = memory or AgentMemoryConfig()
        if self._memory.store and state is not None:
            # Auto-provision a Dapr-backed memory if we have a state store.
            self._memory.store = ConversationDaprStateMemory(  # type: ignore[union-attr]
                store_name=state.store.store_name,
                agent_name=self.name,
            )
        self.memory = self._memory.store or ConversationListMemory()
        if hasattr(self.memory, "agent_name"):
            self.memory.agent_name = self.name

        # -----------------------------
        # Prompting helper
        # -----------------------------
        self.prompting_helper = PromptingAgentBase(
            name=self.name,
            role=resolved_profile.role or "Assistant",
            goal=resolved_profile.goal or "Help users accomplish their tasks.",
            instructions=list(resolved_profile.instructions),
            style_guidelines=list(resolved_profile.style_guidelines),
            system_prompt=resolved_profile.system_prompt,
            template_format=resolved_profile.template_format,
            include_chat_history=True,
            prompt_template=prompt_template,
            profile=resolved_profile,
        )
        # Keep profile config synchronized with helper defaults.
        if self.profile.name is None:
            self.profile.name = self.prompting_helper.name
        if self.profile.role is None:
            self.profile.role = self.prompting_helper.role
        if self.profile.goal is None:
            self.profile.goal = self.prompting_helper.goal

        self.prompt_template = self.prompting_helper.prompt_template
        self._text_formatter = self.prompting_helper.text_formatter

        # -----------------------------
        # LLM wiring
        # -----------------------------
        self.llm: ChatClientBase = llm or get_default_llm()
        if self.llm:
            self.llm.prompt_template = self.prompt_template

        # -----------------------------
        # Tools
        # -----------------------------
        self.tools: List[Any] = list(tools or [])
        self.tool_executor = AgentToolExecutor(tools=list(self.tools))
        self.tool_history: List[ToolExecutionRecord] = []

        # -----------------------------
        # Execution config
        # -----------------------------
        self.execution = execution or AgentExecutionConfig()
        try:
            self.execution.max_iterations = max(1, int(self.execution.max_iterations))
        except Exception:
            self.execution.max_iterations = 10
        if not self.tools:
            if self.execution.tool_choice is not None:
                logger.debug(
                    "No tools configured for agent '%s'; ignoring tool_choice=%r.",
                    self.name,
                    self.execution.tool_choice,
                )
            self.execution.tool_choice = None
        elif self.execution.tool_choice is None:
            self.execution.tool_choice = "auto"

        # -----------------------------
        # Agent metadata & registry registration (from AgentComponents)
        # -----------------------------
        base_meta: Dict[str, Any] = {}
        base_meta["agent"] = {
            "appid": self.appid,
            "orchestrator": False,
            "role": self.profile.role,
            "goal": self.profile.goal,
            "name": self.profile.name,
            "instructions": list(self.profile.instructions),
        }

        if self.profile.system_prompt:
            base_meta["agent"]["system_prompt"] = self.profile.system_prompt

        if self.pubsub is not None:
            pubsub_meta: Dict[str, Any] = {}
            pubsub_meta["agent_name"] = self.agent_topic_name
            pubsub_meta["name"] = self.message_bus_name
            base_meta["pubsub"] = pubsub_meta

        if self.memory:
            memory_meta: Dict[str, Any] = {}
            memory_meta["type"] = type(self.memory).__name__
            if getattr(self.memory, "store_name", None) is not None:
                memory_meta["statestore"] = self.memory.store_name
            base_meta["memory"] = memory_meta

        if self.llm:
            llm_meta: Dict[str, Any] = {}
            llm_meta["client"] = type(self.llm).__name__
            llm_meta["provider"] = getattr(self.llm, "provider", "unknown")
            llm_meta["api"] = getattr(self.llm, "api", "unknown")
            llm_meta["model"] = getattr(self.llm, "model", "unknown")
            if hasattr(self.llm, "component_name") and self.llm.component_name:
                llm_meta["component_name"] = self.llm.component_name
            # Include endpoint info (non-sensitive)
            if hasattr(self.llm, "base_url") and self.llm.base_url:
                llm_meta["base_url"] = self.llm.base_url
            if hasattr(self.llm, "azure_endpoint") and self.llm.azure_endpoint:
                llm_meta["azure_endpoint"] = self.llm.azure_endpoint
            if hasattr(self.llm, "azure_deployment") and self.llm.azure_deployment:
                llm_meta["azure_deployment"] = self.llm.azure_deployment
            if self.llm.prompt_template is not None:
                llm_meta["prompt_template"] = type(self.llm.prompt_template).__name__
            if hasattr(self.llm, "prompty") and self.llm.prompty is not None:
                llm_meta["prompty"] = self.llm.prompty
            base_meta["llm"] = llm_meta

        if self.execution:
            if (
                hasattr(self.execution, "max_iterations")
                and self.execution.max_iterations is not None
            ):
                base_meta["max_iterations"] = self.execution.max_iterations
            if (
                hasattr(self.execution, "tool_choice")
                and self.execution.tool_choice is not None
            ):
                base_meta["tool_choice"] = self.execution.tool_choice

        if self.tools and len(self.tools) > 0:
            tools_list = [
                {
                    "tool_name": tool.name,
                    "tool_description": tool.description,
                    "tool_args": tool.args_schema,
                }
                for tool in self.tools
            ]
            base_meta["tools"] = tools_list

        merged_meta = {**base_meta, **(agent_metadata or {})}
        self.agent_metadata = merged_meta
        if self.registry_state is not None:
            try:
                self.register_agentic_system(metadata=merged_meta)
            except StateStoreError:
                logger.warning(
                    "Could not register agent metadata; registry unavailable."
                )
        else:
            logger.debug(
                "Registry configuration not provided; skipping agent registration."
            )

        if self.configuration:
            self._setup_configuration_subscription()

    def _setup_configuration_subscription(self) -> None:
        """Initialize the configuration subscription."""
        if not self.configuration:
            return

        try:
            # Determine the configuration name/keys to subscribe to
            config_name = self.configuration.config_name or self.name
            keys = self.configuration.keys or [config_name]

            # We use a dedicated client for the subscription to keep the connection alive.
            self._config_client = DaprClient()
            self._subscription_id = self._config_client.subscribe_configuration(
                store_name=self.configuration.store_name,
                keys=keys,
                handler=self._config_handler,
                config_metadata=self.configuration.metadata,
            )
            logger.info(
                f"Agent {self.name} subscribed to configuration store '{self.configuration.store_name}' for keys {keys} (ID: {self._subscription_id})"
            )
        except Exception as e:
            logger.error(
                f"Agent {self.name} failed to subscribe to configuration store '{self.configuration.store_name}': {e}"
            )

    def _config_handler(self, config_id: str, response: ConfigurationResponse) -> None:
        """Handler for configuration updates."""
        try:
            for key, item in response.items.items():
                # If the value is a JSON blob, try to parse it and apply updates
                logger.info(f"Received configuration update for key: {key}")
                try:
                    data = json.loads(item.value)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            self._apply_config_update(k, v)
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass

                # Otherwise treat as a single key-value pair
                self._apply_config_update(key, item.value)
        except Exception as e:
            logger.error(f"Error in configuration handler for agent {self.name}: {e}")

    _SENSITIVE_KEYS = frozenset({"llm_api_key", "openai_api_key"})

    def _apply_config_update(self, key: str, value: Any) -> None:
        """Apply a configuration update to the agent state."""
        normalized_key = key.lower().replace("-", "_")
        safe_value = "***" if normalized_key in self._SENSITIVE_KEYS else value
        logger.info(f"Agent {self.name} applying config update: {key}={safe_value}")

        # Profile updates
        if normalized_key in ["role", "agent_role"]:
            self.profile.role = str(value)
            self.prompting_helper.role = str(value)
        elif normalized_key in ["goal", "agent_goal"]:
            self.profile.goal = str(value)
            self.prompting_helper.goal = str(value)
        elif normalized_key in ["instructions", "agent_instructions"]:
            instructions = value if isinstance(value, list) else [str(value)]
            self.profile.instructions = instructions
            self.prompting_helper.instructions = instructions
        elif normalized_key in ["system_prompt", "agent_system_prompt"]:
            self.profile.system_prompt = str(value)
            self.prompting_helper.system_prompt = str(value)

        # LLM updates — some LLM clients expose read-only properties,
        # so we catch AttributeError when the field is not settable.
        elif normalized_key in ["llm_api_key", "openai_api_key"]:
            try:
                self.llm.api_key = str(value)  # type: ignore[union-attr]
            except (AttributeError, TypeError):
                logger.debug("LLM client does not support setting api_key")
        elif normalized_key == "llm_provider":
            try:
                self.llm.provider = str(value)  # type: ignore[union-attr]
            except (AttributeError, TypeError):
                logger.debug("LLM client does not support setting provider")
        elif normalized_key == "llm_model":
            try:
                self.llm.model = str(value)  # type: ignore[union-attr]
            except (AttributeError, TypeError):
                logger.debug("LLM client does not support setting model")

        # Component Reference Hot-Reloads
        elif normalized_key == "state_store":
            if self.state_store and hasattr(self.state_store, "store_name"):
                logger.info(f"Hot-reloading state store to: {value}")
                self.state_store.store_name = str(value)
        elif normalized_key == "registry_store":
            if self.registry_state and hasattr(self.registry_state, "store_name"):
                logger.info(f"Hot-reloading registry store to: {value}")
                self.registry_state.store_name = str(value)
        elif normalized_key == "memory_store":
            if hasattr(self.memory, "store_name"):
                logger.info(f"Hot-reloading memory store to: {value}")
                self.memory.store_name = str(value)  # type: ignore
        else:
            logger.debug(f"Agent {self.name} ignoring unrecognized config key: {key}")
            return

        # Re-register metadata only when profile or component fields changed
        if self.registry_state is not None:
            self.agent_metadata["agent"]["role"] = self.profile.role
            self.agent_metadata["agent"]["goal"] = self.profile.goal
            self.agent_metadata["agent"]["instructions"] = list(
                self.profile.instructions
            )
            if self.profile.system_prompt:
                self.agent_metadata["agent"]["system_prompt"] = (
                    self.profile.system_prompt
                )

            # Sync LLM metadata
            if "llm" in self.agent_metadata and self.llm:
                self.agent_metadata["llm"]["provider"] = getattr(
                    self.llm, "provider", "unknown"
                )
                self.agent_metadata["llm"]["model"] = getattr(
                    self.llm, "model", "unknown"
                )

            # Update component names in metadata if they changed
            if "statestore" in self.agent_metadata.get("agent", {}):
                self.agent_metadata["agent"]["statestore"] = self.state_store.store_name
            if "memory" in self.agent_metadata:
                self.agent_metadata["memory"]["statestore"] = getattr(
                    self.memory, "store_name", "unknown"
                )

            try:
                self.register_agentic_system(metadata=self.agent_metadata)
            except Exception as e:
                logger.warning(f"Failed to re-register agent after config update: {e}")

    def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        # Deregister from the team registry
        if self.registry_state is not None:
            try:
                self.deregister_agentic_system()
            except Exception as e:
                logger.debug(f"Error deregistering agent from registry: {e}")

        if hasattr(self, "_config_client") and getattr(self, "_config_client", None):
            if self._subscription_id and self.configuration:
                try:
                    self._config_client.unsubscribe_configuration(
                        store_name=self.configuration.store_name,
                        configuration_id=self._subscription_id,
                    )
                except Exception as e:
                    logger.debug(f"Error unsubscribing from configuration: {e}")
            try:
                self._config_client.close()
            except Exception:
                pass
            self._config_client = None

    # ------------------------------------------------------------------
    # DaprInfra delegation properties and methods
    # ------------------------------------------------------------------
    @property
    def pubsub(self):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.pubsub

    @property
    def registry_state(self):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.registry_state

    @property
    def agent_topic_name(self):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.agent_topic_name

    @property
    def message_bus_name(self):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.message_bus_name

    @property
    def broadcast_topic_name(self):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.broadcast_topic_name

    @property
    def workflow_grpc_options(self):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.workflow_grpc_options

    @property
    def state_store(self):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.state_store

    @property
    def _state_model(self):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra._state_model

    @property
    def state(self):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.state

    @property
    def workflow_state(self):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.workflow_state

    def save_state(self, workflow_instance_id: str) -> None:
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.save_state(workflow_instance_id)

    def get_state(self, instance_id: str) -> Optional[Any]:
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.get_state(instance_id)

    def mark_workflow_terminated(self, instance_id: str) -> None:
        """
        No-op for state store; terminated status comes from Dapr get_workflow runtime_status.
        """

    def register_agentic_system(self, *, metadata=None, team=None):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.register_agentic_system(metadata=metadata, team=team)

    def deregister_agentic_system(self, *, team=None):
        """Delegate to DaprInfra."""
        return self._infra.deregister_agentic_system(team=team)

    def get_agents_metadata(
        self, *, exclude_self=True, exclude_orchestrator=False, team=None
    ):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.get_agents_metadata(
                exclude_self=exclude_self,
                exclude_orchestrator=exclude_orchestrator,
                team=team,
            )

    def sync_system_messages(self, instance_id, all_messages):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra.sync_system_messages(
                instance_id=instance_id, all_messages=all_messages
            )

    def _message_dict_to_message_model(self, message):
        """Delegate to DaprInfra."""
        if hasattr(self, "_infra"):
            return self._infra._message_dict_to_message_model(message)

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------
    @property
    def text_formatter(self) -> ColorTextFormatter:
        """Formatter used for human-friendly console output."""
        return self._text_formatter

    @text_formatter.setter
    def text_formatter(self, formatter: ColorTextFormatter) -> None:
        """Override the default text formatter and keep the helper in sync."""
        self._text_formatter = formatter
        if hasattr(self, "prompting_helper"):
            self.prompting_helper._text_formatter = formatter

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
            (source_agent_name, "dapr_agents_mustard"),
            (" -> ", "dapr_agents_teal"),
            (f"{target_agent_name}\n\n", "dapr_agents_mustard"),
            (message + "\n\n", None),
            (separator + "\n", "dapr_agents_teal"),
        ]
        self._text_formatter.print_colored_text(parts)

    # ------------------------------------------------------------------
    # Prompting & memory utilities
    # ------------------------------------------------------------------
    def build_initial_messages(
        self,
        user_input: Optional[Union[str, Dict[str, Any]]] = None,
        workflow_instance_id: str = "default",
        **extra_variables: Any,
    ) -> List[Dict[str, Any]]:
        """
        Build the initial message list for an LLM call.

        Args:
            user_input: Optional user message or structured payload.
            workflow_instance_id: Workflow instance id for chat history; use "default" when not in a workflow.
            **extra_variables: Extra template variables for the prompt template.

        Returns:
            List of message dictionaries ready for an LLM chat API.
        """
        return self.prompting_helper.build_initial_messages(
            user_input,
            chat_history=self.get_chat_history(workflow_instance_id)
            if self.prompting_helper.include_chat_history
            else None,
            **extra_variables,
        )

    def get_chat_history(
        self, workflow_instance_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the conversation history from the configured memory backend.

        Args:
            workflow_instance_id: Workflow instance id to retrieve history for; use "default" when not in a workflow.

        Returns:
            A list of message-like dictionaries in normalized form.
        """
        try:
            history = self.memory.get_messages(workflow_instance_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory get_messages failed: %s", exc)
            return []

        normalized: List[Dict[str, Any]] = []
        for entry in history:
            if hasattr(entry, "model_dump"):
                normalized.append(entry.model_dump())
            elif isinstance(entry, dict):
                normalized.append(dict(entry))
        return normalized

    def reset_memory(self, workflow_instance_id: str = "default") -> None:
        """Clear all stored conversation messages for the given workflow instance."""
        if self.memory:
            self.memory.reset_memory(workflow_instance_id)

    def get_last_message(
        self, workflow_instance_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Return the last message stored in memory, if any."""
        history = self.get_chat_history(workflow_instance_id)
        return dict(history[-1]) if history else None

    def get_last_user_message(
        self, messages: Sequence[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Return the most recent message authored by the user from a sequence.

        Args:
            messages: Message sequence from which to extract the last user message.

        Returns:
            The last user message as a dict, or None if not present.
        """
        match = self._get_last_user_message(messages)
        if not match:
            return None
        result = dict(match)
        content = result.get("content")
        if isinstance(content, str):
            result["content"] = content.strip()
        return result

    def get_last_message_if_user(
        self, messages: Sequence[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Return the last message only if it is authored by the user.

        Args:
            messages: Message sequence.

        Returns:
            The last message as a dict if its role is 'user'; otherwise None.
        """
        if messages and messages[-1].get("role") == "user":
            msg = dict(messages[-1])
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = content.strip()
            return msg
        return None

    def _summarize_conversation(
        self,
        instance_id: str,
        messages_list: Sequence[Any],
        tool_history: Any,
    ) -> Dict[str, Any]:
        """
        Shared summarization: build summary from messages + tool_history via LLM,
        save to memory keyed by instance_id, optionally print. Used by durable and standalone.

        Args:
            instance_id: Workflow/run instance id for memory keying.
            messages_list: Sequence of message-like objects (entry.messages).
            tool_history: Tool call/result history (entry.tool_history or []).

        Returns:
            Dict with "content" key holding the summary text, or {} if nothing to summarize.

        Raises:
            AgentError: If memory disabled, LLM fails, or save fails.
        """
        if not self.memory:
            raise AgentError("Long-term conversation memory is not enabled.")

        if not messages_list:
            logger.debug("No messages to summarize for instance_id=%s", instance_id)
            return {}

        lines = []
        for m in messages_list:
            try:
                d = self._serialize_message(m)
            except TypeError:
                d = {"role": "unknown", "content": str(m)}
            lines.append(f"{d.get('role', 'unknown')}: {d.get('content', '')}")
        conversation_text = "\n".join(lines)
        tool_list = tool_history if tool_history is not None else []
        task = (
            "Summarize the following conversation and any tool usage concisely for long-term memory storage. "
            "Focus on key facts, decisions, and outcomes.\n\n"
            "Conversation:\n"
            f"{conversation_text}\n\n"
            "Tool calls/results:\n"
            f"{json.dumps(tool_list, default=str)}"
        )
        llm_messages: List[Dict[str, Any]] = [
            {"role": "user", "content": task},
        ]

        try:
            summary_model: ConversationSummary = self.llm.generate(
                messages=llm_messages,
                response_format=ConversationSummary,
            )
        except Exception as exc:  # noqa: BLE001
            raise AgentError(f"LLM summarize failed: {exc}") from exc

        summary_content = (summary_model.summary or "").strip()
        if not summary_content:
            raise AgentError("LLM returned an empty summary.")

        summary_message: Dict[str, Any] = {
            "role": "assistant",
            "content": summary_content,
            "name": self.name,
        }
        try:
            self.memory.add_message(summary_message, workflow_instance_id=instance_id)
        except Exception:
            raise AgentError(
                f"Failed to save summary to memory for instance_id={instance_id}"
            )
        logger.info("Saved summary to memory for instance_id=%s", instance_id)
        if getattr(self, "text_formatter", None):
            self.text_formatter.print_message(
                {**summary_message, "name": f"{self.name}"}
            )
        return {"content": summary_content}

    def get_llm_tools(self) -> List[Union[AgentTool, Dict[str, Any]]]:
        """
        Convert configured tools into LLM-friendly tool specs.

        Returns:
            List of `AgentTool` or tool-spec dicts.
        """
        llm_tools: List[Union[AgentTool, Dict[str, Any]]] = []
        for tool in self.tools:
            if isinstance(tool, AgentTool):
                llm_tools.append(tool)
            elif callable(tool):
                try:
                    llm_tools.append(AgentTool.from_func(tool))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to convert callable to AgentTool: %s", exc)
        return llm_tools

    def _build_profile(
        self,
        *,
        base_profile: Optional[AgentProfileConfig],
        name: Optional[str],
        role: Optional[str],
        goal: Optional[str],
        instructions: Optional[Iterable[str]],
        style_guidelines: Optional[Iterable[str]],
        system_prompt: Optional[str],
    ) -> AgentProfileConfig:
        """
        Construct a concrete AgentProfileConfig from a base profile and field overrides.

        Args:
            base_profile: Optional starting profile to clone (avoids mutating the caller’s).
            name: Name override.
            role: Role override.
            goal: Goal/mission override.
            instructions: Additional instruction strings.
            style_guidelines: Prompt style directives.
            system_prompt: System prompt override.

        Returns:
            A fully-populated AgentProfileConfig with a non-empty name.

        Raises:
            ValueError: If the resulting profile has an empty name.
        """
        # Clone the base profile to avoid external side effects.
        if base_profile is not None:
            profile = AgentProfileConfig(
                name=base_profile.name,
                role=base_profile.role,
                goal=base_profile.goal,
                instructions=list(base_profile.instructions),
                style_guidelines=list(base_profile.style_guidelines),
                system_prompt=base_profile.system_prompt,
                template_format=base_profile.template_format,
                modules=tuple(base_profile.modules),
                module_overrides=dict(base_profile.module_overrides),
            )
        else:
            profile = AgentProfileConfig()

        # Apply field-level overrides when provided.
        if name is not None:
            profile.name = name
        if role is not None:
            profile.role = role
        if goal is not None:
            profile.goal = goal
        if instructions is not None:
            profile.instructions = list(instructions)
        if style_guidelines is not None:
            profile.style_guidelines = list(style_guidelines)
        if system_prompt is not None:
            profile.system_prompt = system_prompt

        # Durable agents require a concrete name for state/memory/registry keys.
        if not profile.name or not profile.name.strip():
            raise ValueError(
                "Durable agents require a non-empty name "
                "(provide name= or profile.name)."
            )

        return profile

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _run_asyncio_task(coro: Coroutine[Any, Any, Any]) -> Any:
        """
        Execute an async coroutine from a synchronous context, creating a fresh loop if needed.

        Args:
            coro: The coroutine to execute.

        Returns:
            Any: The result of the coroutine execution.
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

    @staticmethod
    def _serialize_message(message: Any) -> Dict[str, Any]:
        """
        Convert a message-like object into a plain dict for history persistence.

        Args:
            message: Pydantic model, dict, or object exposing `model_dump`.

        Returns:
            Normalized dictionary representation.

        Raises:
            TypeError: When the input type is unsupported.
        """
        if hasattr(message, "model_dump"):
            return message.model_dump()
        if isinstance(message, dict):
            return dict(message)
        if hasattr(message, "__dict__"):
            return dict(message.__dict__)
        raise TypeError(
            f"Unsupported message type for serialization: {type(message)!r}"
        )

    def _get_last_user_message(
        self, messages: Sequence[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find the last user-role message from the given sequence."""
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        return None

    def _reconstruct_conversation_history(
        self, instance_id: str
    ) -> List[AgentWorkflowMessage]:
        """
        Build a conversation history combining persistent memory and per-instance messages.

        Args:
            instance_id: Workflow instance identifier.

        Returns:
            Combined message history excluding system messages from instance timeline.
        """
        try:
            entry = self._infra.get_state(instance_id)
        except Exception:
            logger.exception(
                f"Failed to reconstruct conversation workflow history for instance_id: {instance_id}"
            )
            raise

        return entry.messages

    def _sync_system_messages_with_state(
        self,
        instance_id: str,
        all_messages: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Persist the latest set of system messages into the instance state.

        Args:
            instance_id: Workflow instance id.
            all_messages: Complete message list to scan for system-role messages.
        """
        # Delegate to AgentComponents logic.
        self.sync_system_messages(instance_id=instance_id, all_messages=all_messages)

    def _process_user_message(
        self,
        instance_id: str,
        task: Optional[str],
        user_message_copy: Optional[Dict[str, Any]],
    ) -> None:
        """
        Append a user message into the instance timeline and memory, and persist state.

        Args:
            instance_id: Workflow instance id.
            task: Optional task string; if missing, no-op.
            user_message_copy: Message dict to append.
        """
        if not task or not user_message_copy:
            return

        try:
            entry = self._infra.get_state(instance_id)
        except Exception:
            logger.exception(
                f"Failed to get workflow state for instance_id to process user message: {instance_id}"
            )
            raise

        if entry is not None and hasattr(entry, "messages"):
            # Use configured coercer / message model
            message_model = (
                self._message_coercer(user_message_copy)  # type: ignore[attr-defined]
                if getattr(self, "_message_coercer", None)
                else self._message_dict_to_message_model(user_message_copy)
            )
            entry.messages.append(message_model)  # type: ignore[attr-defined]
            if hasattr(entry, "last_message"):
                entry.last_message = message_model  # type: ignore[attr-defined]

        self.save_state(instance_id)

    def _save_assistant_message(
        self, instance_id: str, assistant_message: Dict[str, Any]
    ) -> None:
        """
        Append an assistant message into the instance timeline and memory, and persist state.

        Args:
            instance_id: Workflow instance id.
            assistant_message: Assistant message dict (will be tagged with agent name).
        """
        assistant_message["name"] = self.name

        try:
            entry = self._infra.get_state(instance_id)
        except Exception:
            logger.exception(
                f"Failed to get workflow state for instance_id: {instance_id}"
            )
            raise

        if entry is not None and hasattr(entry, "messages"):
            message_id = assistant_message.get("id")
            if message_id and any(
                getattr(msg, "id", None) == message_id
                for msg in getattr(entry, "messages")
            ):
                # Duplicate in state - skip state update but still add to memory
                pass
            else:
                message_model = (
                    self._message_coercer(assistant_message)  # type: ignore[attr-defined]
                    if getattr(self, "_message_coercer", None)
                    else self._message_dict_to_message_model(assistant_message)
                )
                entry.messages.append(message_model)  # type: ignore[attr-defined]
                if hasattr(entry, "last_message"):
                    entry.last_message = message_model  # type: ignore[attr-defined]

        self.save_state(instance_id)

    # ------------------------------------------------------------------
    # Small convenience wrappers
    # ------------------------------------------------------------------
    def list_team_agents(
        self, *, team: Optional[str] = None, include_self: bool = True
    ) -> Dict[str, Any]:
        """
        Convenience wrapper over `get_agents_metadata`.

        Args:
            team: Team override.
            include_self: If True, include this agent in the results.

        Returns:
            Mapping of agent name to metadata.
        """
        return self.get_agents_metadata(
            exclude_self=not include_self,
            exclude_orchestrator=False,
            team=team,
        )

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_datetime(value: Optional[Any]) -> datetime:
        """Coerce strings/None to a timezone-aware UTC datetime."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def _resolve_observability_config(self) -> AgentObservabilityConfig:
        """
        Resolve the observability configuration for the agent in the following order:
        1. Passed through instantiation (highest priority)
        2. Environment variables
        3. Default statestore runtime config (lowest priority)

        Args:
            agent_observability: Optional observability config provided during initialization.
        Returns:
            Resolved AgentObservabilityConfig instance.
        """

        config = self._load_observability_from_statestore()
        logger.debug(f"Statestore observability config: {config}")

        env_config = AgentObservabilityConfig.from_env()
        logger.debug(f"Env observability config: {env_config}")

        config = self._merge_observability_configs(config, env_config)
        logger.debug(f"Merged observability config: {config}")

        if self._agent_observability:
            config = self._merge_observability_configs(
                config, self._agent_observability
            )
            logger.debug(f"Final observability config with override: {config}")
        return config

    def _load_observability_from_statestore(self) -> AgentObservabilityConfig:
        """
        Load observability configuration from the state store.

        Returns:
            AgentObservabilityConfig instance loaded from state store.
        """

        try:
            enabled = self._runtime_conf.get("OTEL_ENABLED", "false").lower() == "true"
            auth_token = (
                self._runtime_secrets.get("OTEL_TOKEN")
                or self._runtime_conf.get("OTEL_TOKEN")
                or None
            )
            endpoint = self._runtime_conf.get("OTEL_ENDPOINT") or None
            service_name = self._runtime_conf.get("OTEL_SERVICE_NAME") or None
            logging_enabled = (
                self._runtime_conf.get("OTEL_LOGGING_ENABLED", "false").lower()
                == "true"
            )
            tracing_enabled = (
                self._runtime_conf.get("OTEL_TRACING_ENABLED", "false").lower()
                == "true"
            )

            logging_exporter: Optional[AgentLoggingExporter] = None
            logging_exporter_str = self._runtime_conf.get(
                "OTEL_LOGGING_EXPORTER", "console"
            )
            if logging_exporter_str:
                try:
                    logging_exporter = AgentLoggingExporter(logging_exporter_str)
                except (ValueError, KeyError):
                    logging_exporter = AgentLoggingExporter.CONSOLE

            tracing_exporter: Optional[AgentTracingExporter] = None
            tracing_exporter_str = self._runtime_conf.get(
                "OTEL_TRACING_EXPORTER", "console"
            )
            if tracing_exporter_str:
                try:
                    tracing_exporter = AgentTracingExporter(tracing_exporter_str)
                except (ValueError, KeyError):
                    tracing_exporter = AgentTracingExporter.CONSOLE

            return AgentObservabilityConfig(
                enabled=enabled,
                auth_token=auth_token,
                endpoint=endpoint,
                service_name=service_name,
                logging_enabled=logging_enabled,
                logging_exporter=logging_exporter,
                tracing_enabled=tracing_enabled,
                tracing_exporter=tracing_exporter,
            )
        except Exception as e:
            logger.debug(f"Could not load observability config from statestore: {e}")
            return AgentObservabilityConfig()

    def _merge_observability_configs(
        self, base: AgentObservabilityConfig, override: AgentObservabilityConfig
    ) -> AgentObservabilityConfig:
        """
        Merge two observability configurations, with the override taking precedence.
        Only override if the override value is not None.

        Args:
            base: Base observability configuration.
            override: Override observability configuration.
        Returns:
            Merged AgentObservabilityConfig instance.
        """
        merged_headers = {**base.headers, **override.headers}

        enabled = override.enabled if override.enabled is not None else base.enabled
        logging_enabled = (
            override.logging_enabled
            if override.logging_enabled is not None
            else base.logging_enabled
        )
        tracing_enabled = (
            override.tracing_enabled
            if override.tracing_enabled is not None
            else base.tracing_enabled
        )

        merged_config = AgentObservabilityConfig(
            enabled=enabled,
            headers=merged_headers,
            auth_token=override.auth_token or base.auth_token,
            endpoint=override.endpoint or base.endpoint,
            service_name=override.service_name or base.service_name,
            logging_enabled=logging_enabled,
            logging_exporter=override.logging_exporter or base.logging_exporter,
            tracing_enabled=tracing_enabled,
            tracing_exporter=override.tracing_exporter or base.tracing_exporter,
        )
        return merged_config

    def _setup_agent_runtime_configuration(self) -> None:
        self._agent_observability = self._resolve_observability_config()
        self._setup_agent_observability(self._agent_observability)

    def _setup_agent_observability(self, config: AgentObservabilityConfig) -> None:
        """
        Setup agent runtime configuration.
        """

        # OTel setup
        if config.enabled:
            # Set resource name for tracing and logging
            resource = Resource(
                attributes={
                    "service.name": config.service_name
                    or self.name.replace(" ", "-").lower()
                }
            )

            otlp_headers: dict[str, str] = config.headers or {}

            otel_token = config.auth_token
            if otel_token != "":
                otlp_headers["authorization"] = f"Bearer {otel_token}"

            _endpoint = config.endpoint or ""

            logger_provider = None
            if config.logging_enabled:
                logger_provider = LoggerProvider(resource=resource)

                _exporter = config.logging_exporter
                if _endpoint == "" and _exporter != "console":
                    raise ValueError(
                        "OTEL_ENDPOINT must be set when OTEL_LOGGING_EXPORTER is not 'console'"
                    )

                match _exporter:
                    case AgentLoggingExporter.OTLP_GRPC:
                        log_processor = BatchLogRecordProcessor(
                            OTLPGrpcLogExporter(
                                endpoint=_endpoint, headers=otlp_headers
                            )
                        )
                    case AgentLoggingExporter.OTLP_HTTP:
                        log_processor = BatchLogRecordProcessor(
                            OTLPHTTPLogExporter(
                                endpoint=f"{_endpoint}/v1/logs"
                                if "/v1/logs" not in _endpoint
                                else _endpoint,
                                headers=otlp_headers,
                            )
                        )
                    case _:
                        log_processor = BatchLogRecordProcessor(
                            ConsoleLogRecordExporter()
                        )

                logger_provider.add_log_record_processor(log_processor)
                handler = LoggingHandler(
                    level=logging.NOTSET, logger_provider=logger_provider
                )
                logging.getLogger().addHandler(handler)
                _logs.set_logger_provider(logger_provider)

            tracer_provider = None
            if config.tracing_enabled:
                tracer_provider = TracerProvider(resource=resource)

                _exporter = config.tracing_exporter
                if _endpoint == "" and _exporter != "console":
                    raise ValueError(
                        "OTEL_ENDPOINT must be set when OTEL_TRACING_EXPORTER is not 'console'"
                    )

                match _exporter:
                    case AgentTracingExporter.OTLP_GRPC:
                        tracing_exporter = OTLPGrpcSpanExporter(
                            endpoint=_endpoint, headers=otlp_headers
                        )
                    case AgentTracingExporter.OTLP_HTTP:
                        tracing_exporter = OTLPHTTPSpanExporter(
                            endpoint=f"{_endpoint}/v1/traces"
                            if "/v1/traces" not in _endpoint
                            else _endpoint,
                            headers=otlp_headers,
                        )
                    case AgentTracingExporter.ZIPKIN:
                        tracing_exporter = ZipkinExporter(
                            endpoint=f"{_endpoint}/api/v2/spans"
                            if "/api/v2/spans" not in _endpoint
                            else _endpoint
                        )
                    case _:
                        tracing_exporter = ConsoleSpanExporter()

                span_processor = BatchSpanProcessor(tracing_exporter)
                tracer_provider.add_span_processor(span_processor)
                trace.set_tracer_provider(tracer_provider)

            self.instrumentor = DaprAgentsInstrumentor()
            self.instrumentor.instrument(
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
            )
