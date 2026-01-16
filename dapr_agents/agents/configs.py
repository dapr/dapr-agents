from __future__ import annotations

import re
from os import getenv
from enum import StrEnum
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Type,
    Union,
)

from pydantic import BaseModel

from dapr_agents.agents.schemas import AgentWorkflowMessage, AgentWorkflowState

from dapr_agents.memory import ConversationListMemory, MemoryBase
from dapr_agents.storage.daprstores.stateservice import StateStoreService

_JINJA_PLACEHOLDER_PATTERN = re.compile(r"(?<!\{)\{\s*(\w+)\s*\}(?!\})")


def _ensure_jinja_placeholders(text: str) -> str:
    return _JINJA_PLACEHOLDER_PATTERN.sub(r"{{\1}}", text)


def _empty_headers() -> Dict[str, str]:
    return {}


# Type hooks for state customization
EntryFactory = Callable[..., Any]
MessageCoercer = Callable[[Dict[str, Any]], Any]
EntryContainerGetter = Callable[[BaseModel], Optional[MutableMapping[str, Any]]]


@dataclass
class StateModelBundle:
    """
    Bundled state schema configuration for an agent/orchestrator type.

    Encapsulates the state and message model classes along with their
    associated factory/coercer hooks. This is an internal abstraction
    used by AgentStateConfig to ensure only vetted schemas are used.

    Attributes:
        state_model_cls: Root Pydantic model class for the state.
        message_model_cls: Pydantic model class for workflow/system messages.
        entry_factory: Optional factory to create workflow entry instances.
        message_coercer: Optional function to transform message dicts.
        entry_container_getter: Optional function to locate the instance container.
    """

    state_model_cls: Type[BaseModel]
    message_model_cls: Type[BaseModel]
    entry_factory: Optional[EntryFactory] = None
    message_coercer: Optional[MessageCoercer] = None
    entry_container_getter: Optional[EntryContainerGetter] = None


DEFAULT_AGENT_WORKFLOW_BUNDLE = StateModelBundle(
    state_model_cls=AgentWorkflowState,
    message_model_cls=AgentWorkflowMessage,
)


@dataclass
class WorkflowGrpcOptions:
    """
    Optional overrides for Durable Task gRPC channel limits.

    Allows agents/orchestrators to lift the default ~4 MB message size
    ceiling when sending or receiving large payloads through the workflow
    runtime channel.
    """

    max_send_message_length: Optional[int] = None
    max_receive_message_length: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            self.max_send_message_length is not None
            and self.max_send_message_length <= 0
        ):
            raise ValueError("max_send_message_length must be greater than 0")
        if (
            self.max_receive_message_length is not None
            and self.max_receive_message_length <= 0
        ):
            raise ValueError("max_receive_message_length must be greater than 0")


@dataclass
class AgentStateConfig:
    """
    State persistence configuration.

    Schema is auto-selected by agent/orchestrator type. Supply storage details
    and optional hooks; the framework injects the appropriate schema bundle.

    Examples:
        # Schema auto-selected by agent type
        config = AgentStateConfig(store=StateStoreService(...))
        agent = DurableAgent(state=config, ...)  # → AgentWorkflowState
        orch = LLMOrchestrator(state=config, ...)  # → LLMWorkflowState

        # With custom hooks
        config = AgentStateConfig(
            store=StateStoreService(...),
            entry_factory=custom_factory,
        )
    """

    store: "StateStoreService"
    default_state: Optional[Dict[str, Any] | BaseModel] = None
    state_key: Optional[str] = None

    # Hook overrides (optional - bundle provides defaults)
    entry_factory: Optional[EntryFactory] = None
    message_coercer: Optional[MessageCoercer] = None
    entry_container_getter: Optional[EntryContainerGetter] = None

    # Internal: schema bundle (injected by agent/orchestrator class)
    _state_model_bundle: Optional[StateModelBundle] = field(
        default=None, init=False, repr=False
    )

    def ensure_bundle(self, bundle: StateModelBundle) -> None:
        """
        Inject schema bundle (called by agent/orchestrator).

        Args:
            bundle: Schema bundle to use.

        Raises:
            RuntimeError: If different bundle already injected.
        """
        if self._state_model_bundle is not None:
            # Already set - verify it matches
            if (
                self._state_model_bundle.state_model_cls != bundle.state_model_cls
                or self._state_model_bundle.message_model_cls
                != bundle.message_model_cls
            ):
                raise RuntimeError(
                    f"State config already wired with "
                    f"{self._state_model_bundle.state_model_cls.__name__} schema. "
                    f"Cannot inject {bundle.state_model_cls.__name__} schema."
                )
            return  # Same bundle, no-op

        # Merge user hooks with bundle defaults
        self._state_model_bundle = StateModelBundle(
            state_model_cls=bundle.state_model_cls,
            message_model_cls=bundle.message_model_cls,
            entry_factory=self.entry_factory or bundle.entry_factory,
            message_coercer=self.message_coercer or bundle.message_coercer,
            entry_container_getter=self.entry_container_getter
            or bundle.entry_container_getter,
        )

        # Normalize default_state against the bundle's state model
        self._normalize_default_state()

    def get_state_model_bundle(self) -> StateModelBundle:
        """
        Get injected schema bundle.

        Returns:
            StateModelBundle with schema classes and hooks.

        Raises:
            RuntimeError: If bundle not injected yet.
        """
        if self._state_model_bundle is None:
            raise RuntimeError(
                "State config bundle not initialized. "
                "This should be injected by the agent/orchestrator class."
            )
        return self._state_model_bundle

    def _normalize_default_state(self) -> None:
        """Normalize default_state against bundle's schema."""
        if self._state_model_bundle is None:
            return  # Can't normalize without bundle

        Model = self._state_model_bundle.state_model_cls
        if self.default_state is None:
            self.default_state = Model().model_dump(mode="json")
        else:
            if isinstance(self.default_state, BaseModel):
                self.default_state = self.default_state.model_dump(mode="json")
            else:
                self.default_state = Model.model_validate(
                    self.default_state
                ).model_dump(mode="json")


@dataclass
class AgentRegistryConfig:
    """Configuration for agent registry storage."""

    store: StateStoreService
    team_name: Optional[str] = None


@dataclass
class AgentMemoryConfig:
    """Configuration wrapper for agent memory selection."""

    store: MemoryBase = field(default_factory=ConversationListMemory)


@dataclass
class AgentPubSubConfig:
    """Declarative pub/sub configuration for durable agents.

    Attributes:
        pubsub_name: Name of the Dapr pub/sub component to use for all agent traffic.
        agent_topic: Primary topic for direct messages to the agent. Defaults to ``name``.
        broadcast_topic: Optional topic shared by a team for broadcast messages.
    """

    pubsub_name: str
    agent_topic: Optional[str] = None
    broadcast_topic: Optional[str] = None


@dataclass
class PromptSection:
    """Reusable block for composing a structured system prompt."""

    title: str
    lines: List[str] = field(default_factory=list)

    def render(self, template_format: str) -> str:
        if not self.lines:
            return ""
        header = self.title.strip()
        body = "\n".join(f"- {line.strip()}" for line in self.lines if line.strip())
        section = f"{header}:\n{body}".strip()
        return (
            _ensure_jinja_placeholders(section)
            if template_format == "jinja2"
            else section
        )


@dataclass
class AgentProfileConfig:
    """
    High-level persona description for an agent.

    Mirrors common fields in OpenAI Agents SDK while remaining lightweight.
    """

    name: Optional[str] = None
    role: Optional[str] = None
    goal: Optional[str] = None
    instructions: List[str] = field(default_factory=list)
    style_guidelines: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    template_format: str = "jinja2"
    modules: Sequence[str] = field(default_factory=tuple)
    module_overrides: Dict[str, PromptSection] = field(default_factory=dict)


@dataclass
class AgentExecutionConfig:
    """
    Dials to configure the agent execution.
    """

    # TODO: add a forceFinalAnswer field in case maxIterations is near/reached. Or do we have a conclusion baked in by default? Do we want this to derive a conclusion by default?
    # TODO: add stop_at_tokens
    max_iterations: int = 10
    tool_choice: Optional[str] = "auto"


@dataclass
class WorkflowRetryPolicy:
    """
    Configuration for durable retry policies in workflows.

    Attributes:
        max_attempts: Maximum number of retry attempts.
        initial_backoff_seconds: Initial backoff interval in seconds.
        max_backoff_seconds: Maximum backoff interval in seconds.
        backoff_multiplier: Multiplier for exponential backoff.
        retry_timeout: Optional total timeout for all retries in seconds.
    """

    max_attempts: Optional[int] = 1
    initial_backoff_seconds: Optional[int] = 5
    max_backoff_seconds: Optional[int] = 30
    backoff_multiplier: Optional[float] = 1.5
    retry_timeout: Optional[Union[int, None]] = None


class AgentTracingExporter(StrEnum):
    """
    Supported tracing exporters for Dapr Agents observability.
    """

    OTLP_GRPC = "otlp_grpc"
    OTLP_HTTP = "otlp_http"
    ZIPKIN = "zipkin"
    CONSOLE = "console"


class AgentLoggingExporter(StrEnum):
    """
    Supported logging exporters for Dapr Agents observability.
    """

    CONSOLE = "console"
    OTLP_GRPC = "otlp_grpc"
    OTLP_HTTP = "otlp_http"


@dataclass
class AgentObservabilityConfig:
    """
    Configuration settings for Dapr Agents observability features.

    Attributes:
        enabled: Enable/Disable observability.
        headers: Optional headers for observability exporters.
        auth_token: Optional authentication token for exporters.
        endpoint: Optional endpoint URL for observability exporters.
        service_name: Optional service name for observability data.
        logging_enabled: Enable/disable logging observability.
        logging_exporter: Logging exporter type.
        tracing_enabled: Enable/disable tracing observability.
        tracing_exporter: Tracing exporter type.
    """

    enabled: Optional[bool] = None
    headers: Dict[str, str] = field(default_factory=_empty_headers)
    auth_token: Optional[str] = None
    endpoint: Optional[str] = None
    service_name: Optional[str] = None
    logging_enabled: Optional[bool] = None
    logging_exporter: Optional[AgentLoggingExporter] = None
    tracing_enabled: Optional[bool] = None
    tracing_exporter: Optional[AgentTracingExporter] = None

    @classmethod
    def from_env(cls) -> "AgentObservabilityConfig":
        """Create observability config from environment variables."""
        headers: Dict[str, str] = {}
        if token := getenv("OTEL_TOKEN"):
            headers["Authorization"] = token

        logging_exporter: Optional[AgentLoggingExporter] = None
        if logging_exporter_str := getenv("OTEL_LOGGING_EXPORTER"):
            try:
                logging_exporter = AgentLoggingExporter(logging_exporter_str)
            except (ValueError, KeyError):
                logging_exporter = AgentLoggingExporter.CONSOLE

        tracing_exporter: Optional[AgentTracingExporter] = None
        if tracing_exporter_str := getenv("OTEL_TRACING_EXPORTER"):
            try:
                tracing_exporter = AgentTracingExporter(tracing_exporter_str)
            except (ValueError, KeyError):
                tracing_exporter = AgentTracingExporter.CONSOLE

        enabled: Optional[bool] = None
        if getenv("OTEL_ENABLED") is not None:
            enabled = getenv("OTEL_ENABLED", "false").lower() == "true"

        logging_enabled: Optional[bool] = None
        if getenv("OTEL_LOGGING_ENABLED") is not None:
            logging_enabled = getenv("OTEL_LOGGING_ENABLED", "false").lower() == "true"

        tracing_enabled: Optional[bool] = None
        if getenv("OTEL_TRACING_ENABLED") is not None:
            tracing_enabled = getenv("OTEL_TRACING_ENABLED", "false").lower() == "true"

        return cls(
            enabled=enabled,
            headers=headers,
            endpoint=getenv("OTEL_ENDPOINT"),
            service_name=getenv("OTEL_SERVICE_NAME"),
            logging_enabled=logging_enabled,
            logging_exporter=logging_exporter,
            tracing_enabled=tracing_enabled,
            tracing_exporter=tracing_exporter,
        )
