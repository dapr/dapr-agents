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

import functools
import logging
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
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field

from dapr_agents.agents.utils.headers import parse_header_string
from dapr_agents.utils.config import ConfigFieldDescriptor, apply_config_map
from dapr_agents.utils.models import merge_models
from dapr_agents.types.agent import ToolChoice, ToolExecutionMode, OrchestrationMode
from dapr_agents.agents.constants import (
    AGENT_DEFAULT_MAX_ITERATIONS,
    AGENT_DEFAULT_TOOL_CHOICE,
    AGENT_DEFAULT_TOOL_EXECUTION_MODE,
)
from dapr_agents.agents.schemas import (
    AgentWorkflowEntry,
    AgentWorkflowMessage,
)

from dapr_agents.memory import ConversationListMemory, MemoryBase
from dapr_agents.storage.daprstores.stateservice import StateStoreService

_JINJA_PLACEHOLDER_PATTERN = re.compile(r"(?<!\{)\{\s*(\w+)\s*\}(?!\})")

# JSON Schema export constants
_JSON_SCHEMA_KEY = "$schema"
_JSON_SCHEMA_DRAFT_URL = "https://json-schema.org/draft/2020-12/schema"
_JSON_SCHEMA_VERSION_KEY = "version"


def _ensure_jinja_placeholders(text: str) -> str:
    return _JINJA_PLACEHOLDER_PATTERN.sub(r"{{\1}}", text)


def _empty_headers() -> Dict[str, str]:
    return {}


# Type hooks for state customization
EntryFactory = Callable[..., Any]
MessageCoercer = Callable[[Dict[str, Any]], Any]
EntryContainerGetter = Callable[[BaseModel], Optional[MutableMapping[str, Any]]]

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class StateModelBundle:
    """
    Bundled state schema configuration for an agent/orchestrator type.

    With one-key-per-workflow, each state store key holds a single workflow
    entry (entry_model_cls). This bundle identifies that type and related hooks.

    Attributes:
        entry_model_cls: Pydantic model class for one workflow's state (per key).
        message_model_cls: Pydantic model class for workflow/system messages.
        entry_factory: Optional factory to create workflow entry instances.
        message_coercer: Optional function to transform message dicts.
    """

    entry_model_cls: Type[BaseModel]
    message_model_cls: Type[BaseModel]
    entry_factory: Optional[EntryFactory] = None
    message_coercer: Optional[MessageCoercer] = None


AGENT_DEFAULT_WORKFLOW_BUNDLE = StateModelBundle(
    entry_model_cls=AgentWorkflowEntry,
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
    keepalive_time_ms: Optional[int] = None
    keepalive_timeout_ms: Optional[int] = None

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
        if self.keepalive_time_ms is not None and self.keepalive_time_ms <= 0:
            raise ValueError("keepalive_time_ms must be greater than 0")
        if self.keepalive_timeout_ms is not None and self.keepalive_timeout_ms <= 0:
            raise ValueError("keepalive_timeout_ms must be greater than 0")


@dataclass(frozen=True)
class RegistryIndexRetryConfig:
    """
    Retry/backoff policy for mutating the shared team-registry ``_index`` document.

    The index is a single document that every agent on a team contends on when it
    registers or deregisters (acutely so during a coordinated shutdown, where a
    whole team writes at once). Under that N-way optimistic-concurrency contention a
    single mutation can need many more attempts than an ordinary per-key state save,
    which is why this policy is intentionally separate from ``max_etag_attempts``
    (that knob governs single-writer workflow-state saves and is clamped low). A
    mutation is bounded by *both* an attempt count and a wall-clock timeout so it can
    never overrun the caller's shutdown grace period, and uses full-jitter
    exponential backoff to de-correlate concurrent writers so the team converges.

    Field names mirror :class:`WorkflowRetryPolicy` for consistency across the
    package's retry configuration.

    .. warning::
        **Alpha.** These fields are user-facing but not yet stable; the names may
        change in a future 0.x release.

    Attributes:
        max_attempts: Maximum optimistic-concurrency attempts before giving up.
        initial_backoff_seconds: Initial ceiling for the full-jitter backoff delay.
        max_backoff_seconds: Maximum ceiling for the full-jitter backoff delay.
        retry_timeout: Wall-clock budget in seconds across all attempts, including
            backoff.
    """

    max_attempts: int = 50
    initial_backoff_seconds: float = 0.05
    max_backoff_seconds: float = 1.0
    retry_timeout: float = 10.0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than 0")
        if self.initial_backoff_seconds <= 0:
            raise ValueError("initial_backoff_seconds must be greater than 0")
        if self.max_backoff_seconds <= 0:
            raise ValueError("max_backoff_seconds must be greater than 0")
        if self.retry_timeout <= 0:
            raise ValueError("retry_timeout must be greater than 0")


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
        orch = DurableAgent(state=config, orchestration_mode='agent', ...)  # → LLMWorkflowState

        # With custom hooks
        config = AgentStateConfig(
            store=StateStoreService(...),
            entry_factory=custom_factory,
        )
    """

    store: "StateStoreService"
    default_state: Optional[Dict[str, Any] | BaseModel] = None
    state_key_prefix: Optional[str] = None

    # Hook overrides (optional - bundle provides defaults)
    entry_factory: Optional[EntryFactory] = None
    message_coercer: Optional[MessageCoercer] = None

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
                self._state_model_bundle.entry_model_cls != bundle.entry_model_cls
                or self._state_model_bundle.message_model_cls
                != bundle.message_model_cls
            ):
                raise RuntimeError(
                    f"State config already wired with "
                    f"{self._state_model_bundle.entry_model_cls.__name__} schema. "
                    f"Cannot inject {bundle.entry_model_cls.__name__} schema."
                )
            return  # Same bundle, no-op

        # Merge user hooks with bundle defaults
        self._state_model_bundle = StateModelBundle(
            entry_model_cls=bundle.entry_model_cls,
            message_model_cls=bundle.message_model_cls,
            entry_factory=self.entry_factory or bundle.entry_factory,
            message_coercer=self.message_coercer or bundle.message_coercer,
        )

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


# ---------------------------------------------------------------------------
# Built-in config validators for agents
# ---------------------------------------------------------------------------

_config_logger = logging.getLogger(__name__)


def validate_non_empty_string(v: str) -> str:
    """Reject empty or whitespace-only strings."""
    if not v or not v.strip():
        raise ValueError("Value must not be empty")
    return v.strip()


def validate_max_iterations(v: int) -> int:
    """Ensure max_iterations is at least 1."""
    if v < 1:
        raise ValueError(f"max_iterations must be >= 1, got {v}")
    return v


def validate_tool_choice(v: str) -> str:
    """Warn if tool_choice is non-standard, but allow it."""
    try:
        ToolChoice(v.lower())
    except (ValueError, KeyError):
        _config_logger.warning(
            f"tool_choice {v} not in standard set {set([tc.value for tc in ToolChoice])}; allowing anyway."
        )

    return v


def validate_tool_execution_mode(v: str) -> str:
    """Validate that the tool execution mode is a known ToolExecutionMode value."""
    try:
        ToolExecutionMode(v.lower())
    except (ValueError, KeyError):
        raise ValueError(
            f"Unknown tool execution mode '{v}'. "
            f"Valid options: {[e.value for e in ToolExecutionMode]}"
        )

    return v


def validate_orchestration_mode(v: str) -> str:
    """Validate that the orchestration mode is a known OrchestrationMode value."""
    try:
        OrchestrationMode(v.lower())
    except (ValueError, KeyError):
        raise ValueError(
            f"Unknown orchestration mode '{v}'. "
            f"Valid options: {[e.value for e in OrchestrationMode]}"
        )

    return v


def validate_otel_exporter_tracing(v: str) -> str:
    """Validate that the tracing exporter is a known AgentTracingExporter value."""
    try:
        AgentTracingExporter(v)
    except (ValueError, KeyError):
        raise ValueError(
            f"Unknown tracing exporter '{v}'. "
            f"Valid options: {[e.value for e in AgentTracingExporter]}"
        )
    return v


def validate_otel_exporter_logging(v: str) -> str:
    """Validate that the logging exporter is a known AgentLoggingExporter value."""
    try:
        AgentLoggingExporter(v)
    except (ValueError, KeyError):
        raise ValueError(
            f"Unknown logging exporter '{v}'. "
            f"Valid options: {[e.value for e in AgentLoggingExporter]}"
        )
    return v


@dataclass
class RuntimeSubscriptionConfig:
    """Configuration for subscribing to a Dapr Configuration Store at runtime.

    Attributes:
        store_name: Name of the Dapr configuration store component.
        default_key: Fallback key used when ``keys`` is empty (defaults to agent name).
        keys: Optional list of keys to subscribe to.
        metadata: Optional metadata for the configuration subscription.
        on_config_change: Optional callback invoked after each successful config update.
            Receives the normalized key and coerced value.
    """

    store_name: str
    default_key: Optional[str] = None
    keys: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    on_config_change: Optional[Callable[[str, Any], None]] = None


@dataclass
class AgentRegistryConfig:
    """Configuration for agent registry storage.

    Attributes:
        store: Dapr state store backing the team registry.
        team_name: Optional team override; falls back to the configured default.
        index_retry: Retry/backoff policy for mutating the shared team-index
            document under multi-agent contention (e.g. coordinated shutdown).
    """

    store: StateStoreService
    team_name: Optional[str] = None
    index_retry: RegistryIndexRetryConfig = field(
        default_factory=RegistryIndexRetryConfig
    )


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
class AgentMCPConfig:
    """Configuration for MCPServer auto-discovery and tool loading.

    When a ``DurableAgent`` is created, the framework queries the Dapr sidecar
    metadata API for loaded ``MCPServer`` resources and automatically connects
    to each one via the built-in ``dapr.internal.mcp.<server>.ListTools``
    workflow.

    Attributes:
        timeout_in_seconds: Per-server timeout when waiting for the
            ``ListTools`` workflow to complete.
        allowed_tools: Optional allow-list of tool names.  Only tools whose
            name appears in this set are loaded.  ``None`` loads all tools.
        enabled: Set to ``False`` to disable MCP auto-discovery entirely.
    """

    timeout_in_seconds: int = 30
    allowed_tools: Optional[set] = None
    enabled: bool = True


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
class AgentApprovalConfig:
    """
    Infrastructure configuration for human-in-the-loop approval.

    This tells the agent how to deliver ApprovalRequiredEvent messages when a
    hook returns RequireApproval. The gate for whether approval runs is the hook
    itself — if no hook returns RequireApproval, this config is never used.

    Delivery modes:
        - pubsub_name set: publishes the event to the configured Dapr pub/sub topic.
          Use this when the agent is running in subscribe() or serve() mode and a
          pub/sub component is available (e.g. a Slack bot or dashboard is listening).
        - pubsub_name None (default): no pub/sub publish. The event is held in memory
          and exposed via GET /hitl/approvals when the agent is running in serve() mode.
          For workflow-only agents, submit responses directly via the Dapr sidecar:
          POST <sidecar>/v1.0-beta1/workflows/dapr/{instance_id}/raiseEvent/approval_response_{id}

    Attributes:
        pubsub_name: Optional Dapr pub/sub component for outbound approval events.
            Set to None (default) to disable pub/sub delivery and use HTTP polling instead.
        topic: Topic name used when pubsub_name is set.
        default_timeout_seconds: Seconds to wait before auto-denying when a
            RequireApproval decision does not specify its own timeout_seconds.
    """

    pubsub_name: Optional[str] = None
    topic: str = "agent-approval-requests"
    # None means wait indefinitely — the workflow suspends until a human responds, with no automatic denial. Use a positive int (seconds) to auto-deny after that window elapses when approvers may be unavailable.
    default_timeout_seconds: Optional[int] = 300


@dataclass
class AgentExecutionConfig:
    """
    Dials to configure the agent execution.

    Attributes:
        max_iterations: Maximum number of turns allowed for the agent to produce a final response.
        tool_choice: Tool choice strategy for the agent.
        tool_execution_mode: Tool execution mode for the agent.
        orchestration_mode: Orchestration strategy for the agent.
        max_grpc_inbound_message_size_bytes: Optional gRPC inbound message size
            limit in bytes. When set, takes precedence over
            ``DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES`` for this agent only —
            two agents in the same process can run with independent limits.
            The value is plumbed through a per-agent client factory shared by
            the agent's memory, state, registry, and LLM collaborators.
        approval: Human-in-the-loop configuration for the agent.
    """

    # TODO: add a forceFinalAnswer field in case max_iterations is near/reached. Or do we have a conclusion baked in by default? Do we want this to derive a conclusion by default?
    # TODO: add stop_at_tokens
    max_iterations: Optional[int] = AGENT_DEFAULT_MAX_ITERATIONS
    tool_choice: Optional[ToolChoice] = AGENT_DEFAULT_TOOL_CHOICE
    tool_execution_mode: Optional[ToolExecutionMode] = AGENT_DEFAULT_TOOL_EXECUTION_MODE
    orchestration_mode: Optional[OrchestrationMode] = None
    max_grpc_inbound_message_size_bytes: Optional[int] = None
    approval: Optional[AgentApprovalConfig] = field(default_factory=AgentApprovalConfig)

    @classmethod
    def from_env(cls) -> "AgentExecutionConfig":
        """
        Create execution configuration from environment variables.

        Returns:
            AgentExecutionConfig instance created from environment variables.
        """
        config_field_map = {
            EnvConfigKey.MAX_ITERATIONS: ConfigFieldDescriptor(
                target_type=Optional[int],
                setter=lambda obj, v: setattr(obj, "max_iterations", v),
                getter=lambda: getenv("MAX_ITERATIONS"),
                validator=validate_max_iterations,
                should_raise=False,
            ),
            EnvConfigKey.TOOL_CHOICE: ConfigFieldDescriptor(
                target_type=Optional[
                    str
                ],  # Allow any string as tool choices are permissive
                setter=lambda obj, v: setattr(obj, "tool_choice", v),
                getter=lambda: getenv("TOOL_CHOICE"),
                validator=validate_tool_choice,
            ),
            EnvConfigKey.TOOL_EXECUTION_MODE: ConfigFieldDescriptor(
                target_type=Optional[ToolExecutionMode],
                setter=lambda obj, v: setattr(obj, "tool_execution_mode", v),
                getter=lambda: getenv("TOOL_EXECUTION_MODE"),
                validator=validate_tool_execution_mode,
                should_raise=False,
            ),
            EnvConfigKey.ORCHESTRATION_MODE: ConfigFieldDescriptor(
                target_type=Optional[OrchestrationMode],
                setter=lambda obj, v: setattr(obj, "orchestration_mode", v),
                getter=lambda: getenv("ORCHESTRATION_MODE"),
                validator=validate_orchestration_mode,
                should_raise=False,
            ),
            EnvConfigKey.MAX_GRPC_INBOUND_MESSAGE_SIZE_BYTES: ConfigFieldDescriptor(
                target_type=Optional[int],
                setter=lambda obj, v: setattr(
                    obj, "max_grpc_inbound_message_size_bytes", v
                ),
                getter=lambda: getenv("MAX_GRPC_INBOUND_MESSAGE_SIZE_BYTES"),
                should_raise=False,
            ),
        }

        config = cls._template_config()
        apply_config_map(config, config_field_map)

        return config

    @classmethod
    def from_instantiation(
        cls, instantiated_config: Optional["AgentExecutionConfig"]
    ) -> "AgentExecutionConfig":
        """
        Create execution configuration from an instantiated configuration.

        Args:
            config: Optional user-instantiated configuration.

        Returns:
            AgentExecutionConfig instance created from the instantiated configuration.
        """
        instantiated_config = instantiated_config or cls._template_config()
        config_field_map = {
            "max_iterations": ConfigFieldDescriptor(
                target_type=Optional[int],
                setter=lambda obj, v: setattr(obj, "max_iterations", v),
                getter=lambda: instantiated_config.max_iterations,
                validator=validate_max_iterations,
            ),
            "tool_choice": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "tool_choice", v),
                getter=lambda: instantiated_config.tool_choice,
                validator=validate_tool_choice,
            ),
            "tool_execution_mode": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "tool_execution_mode", v),
                getter=lambda: instantiated_config.tool_execution_mode,
                validator=validate_tool_execution_mode,
            ),
            "orchestration_mode": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "orchestration_mode", v),
                getter=lambda: instantiated_config.orchestration_mode,
                validator=validate_orchestration_mode,
            ),
            "max_grpc_inbound_message_size_bytes": ConfigFieldDescriptor(
                target_type=Optional[int],
                setter=lambda obj, v: setattr(
                    obj, "max_grpc_inbound_message_size_bytes", v
                ),
                getter=lambda: instantiated_config.max_grpc_inbound_message_size_bytes,
            ),
            # TODO: validate approval config fields
            "approval": ConfigFieldDescriptor(
                target_type=Optional[AgentApprovalConfig],
                setter=lambda obj, v: setattr(obj, "approval", v),
                getter=lambda: instantiated_config.approval,
            ),
        }

        config = cls._template_config()
        apply_config_map(config, config_field_map)

        return config

    @classmethod
    def from_statestore(
        cls, runtime_config: Optional[Dict[str, Any]]
    ) -> "AgentExecutionConfig":
        """
        Validate and create execution configuration from state store runtime configuration.

        Args:
            runtime_config: Optional state store runtime configuration.

        Returns:
            AgentExecutionConfig instance created from state store runtime configuration.
        """
        runtime_config = runtime_config or {}
        config_field_map = {
            RuntimeConfigKey.MAX_ITERATIONS: ConfigFieldDescriptor(
                target_type=Optional[int],
                setter=lambda obj, v: setattr(obj, "max_iterations", v),
                getter=lambda: runtime_config.get("MAX_ITERATIONS"),
                validator=validate_max_iterations,
                should_raise=False,
            ),
            RuntimeConfigKey.TOOL_CHOICE: ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "tool_choice", v),
                getter=lambda: runtime_config.get("TOOL_CHOICE"),
                validator=validate_tool_choice,
            ),
            # TODO: support orchestration mode
        }

        config = cls._template_config()
        apply_config_map(config, config_field_map)

        return config

    @classmethod
    def resolve_config(
        cls,
        config: Optional["AgentExecutionConfig"],
        runtime_config: Optional[Dict[str, Any]],
    ) -> "AgentExecutionConfig":
        """
        Resolve the execution configuration for the agent in the following order:
        1. State store runtime configuration (highest priority)
        2. Passed through instantiation
        3. Environment variables (lowest priority)

        Args:
            config: Optional user-instantiated configuration.
            runtime_config: Optional state store runtime configuration.

        Returns:
            Resolved AgentExecutionConfig instance.
        """

        env_config = AgentExecutionConfig.from_env()
        logger.debug(f"Environment variable execution config: {env_config}")

        instantiated_config = AgentExecutionConfig.from_instantiation(config)
        logger.debug(f"Instantiated execution config: {instantiated_config}")

        statestore_config = AgentExecutionConfig.from_statestore(runtime_config)
        logger.debug(f"State store runtime execution config: {statestore_config}")

        resolved_config = functools.reduce(
            merge_models,
            [cls._base_config(), env_config, instantiated_config, statestore_config],
        )

        logger.debug(f"Final execution config: {resolved_config}")
        return resolved_config

    @classmethod
    def _template_config(cls) -> "AgentExecutionConfig":
        """Create an execution configuration with defaults cleared.
        Used by environment variable and state store resolution, and as a fallback for instantiated configuration
        so that unset fields with default values do not bleed into the merged configuration.
        """
        return cls(
            max_iterations=None,
            tool_choice=None,
            tool_execution_mode=None,
            orchestration_mode=None,
            approval=None,
            max_grpc_inbound_message_size_bytes=None,
        )

    @classmethod
    def _base_config(cls) -> "AgentExecutionConfig":
        """Create a base execution configuration for resolution."""
        return cls(
            max_iterations=AGENT_DEFAULT_MAX_ITERATIONS,
            tool_choice=AGENT_DEFAULT_TOOL_CHOICE,
            tool_execution_mode=AGENT_DEFAULT_TOOL_EXECUTION_MODE,
            orchestration_mode=None,
            approval=AgentApprovalConfig(),
            max_grpc_inbound_message_size_bytes=None,
        )


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

    max_attempts: Optional[int] = 3
    initial_backoff_seconds: Optional[int] = 5
    max_backoff_seconds: Optional[int] = 30
    backoff_multiplier: Optional[float] = 1.5
    retry_timeout: Optional[Union[int, None]] = None


class EnvConfigKey(StrEnum):
    """Supported keys for environment variable configuration resolution."""

    # Execution fields
    MAX_ITERATIONS = "max_iterations"
    TOOL_CHOICE = "tool_choice"
    TOOL_EXECUTION_MODE = "tool_execution_mode"
    ORCHESTRATION_MODE = "orchestration_mode"
    MAX_GRPC_INBOUND_MESSAGE_SIZE_BYTES = "max_grpc_inbound_message_size_bytes"

    # OTel fields — match standard env var names used throughout
    OTEL_SDK_DISABLED = "otel_sdk_disabled"
    OTEL_EXPORTER_OTLP_ENDPOINT = "otel_exporter_otlp_endpoint"
    OTEL_EXPORTER_OTLP_HEADERS = "otel_exporter_otlp_headers"
    OTEL_SERVICE_NAME = "otel_service_name"
    OTEL_TRACING_ENABLED = "otel_tracing_enabled"
    OTEL_TRACES_EXPORTER = "otel_traces_exporter"
    OTEL_LOGGING_ENABLED = "otel_logging_enabled"
    OTEL_LOGS_EXPORTER = "otel_logs_exporter"


class RuntimeConfigKey(StrEnum):
    """Supported keys for runtime configuration resolution and runtime configuration hot-reload.

    All profile keys use the ``agent_`` prefix to avoid ambiguity.
    Execution, LLM, and component keys are unprefixed.
    """

    # Profile fields
    AGENT_ROLE = "agent_role"
    AGENT_GOAL = "agent_goal"
    AGENT_INSTRUCTIONS = "agent_instructions"
    AGENT_SYSTEM_PROMPT = "agent_system_prompt"
    AGENT_STYLE_GUIDELINES = "agent_style_guidelines"

    # Execution fields
    MAX_ITERATIONS = "max_iterations"
    TOOL_CHOICE = "tool_choice"

    # LLM fields
    LLM_API_KEY = "llm_api_key"
    LLM_PROVIDER = "llm_provider"
    LLM_MODEL = "llm_model"

    # Component references
    AGENT_WORKFLOW = "agent_workflow"
    AGENT_REGISTRY = "agent_registry"
    AGENT_MEMORY = "agent_memory"

    # OTel fields — match standard env var names used throughout
    OTEL_SDK_DISABLED = "otel_sdk_disabled"
    OTEL_EXPORTER_OTLP_ENDPOINT = "otel_exporter_otlp_endpoint"
    OTEL_EXPORTER_OTLP_HEADERS = "otel_exporter_otlp_headers"
    OTEL_SERVICE_NAME = "otel_service_name"
    OTEL_TRACING_ENABLED = "otel_tracing_enabled"
    OTEL_TRACES_EXPORTER = "otel_traces_exporter"
    OTEL_LOGGING_ENABLED = "otel_logging_enabled"
    OTEL_LOGS_EXPORTER = "otel_logs_exporter"


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
        """Validate and create observability config from standard OTEL environment variables.

        Uses standard OpenTelemetry env var names where available:
        - OTEL_SDK_DISABLED (inverted: disabled != "true" means enabled)
        - OTEL_EXPORTER_OTLP_HEADERS (parses "Authorization=<token>" format)
        - OTEL_EXPORTER_OTLP_ENDPOINT
        - OTEL_SERVICE_NAME
        - OTEL_LOGGING_ENABLED (custom, no standard equivalent)
        - OTEL_LOGS_EXPORTER
        - OTEL_TRACING_ENABLED (custom, no standard equivalent)
        - OTEL_TRACES_EXPORTER

        Returns:
            AgentObservabilityConfig instance created from environment variables.
        """
        config_field_map = {
            EnvConfigKey.OTEL_SDK_DISABLED: ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "enabled", v),
                getter=lambda: getenv("OTEL_SDK_DISABLED"),
                validator=lambda v: (
                    v if v is None else not v
                ),  # Invert the disabled flag to set enabled
            ),
            EnvConfigKey.OTEL_EXPORTER_OTLP_HEADERS: ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "headers", v),
                getter=lambda: getenv("OTEL_EXPORTER_OTLP_HEADERS"),
                validator=parse_header_string,
            ),
            EnvConfigKey.OTEL_EXPORTER_OTLP_ENDPOINT: ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "endpoint", v),
                getter=lambda: getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                validator=validate_non_empty_string,
            ),
            EnvConfigKey.OTEL_SERVICE_NAME: ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "service_name", v),
                getter=lambda: getenv("OTEL_SERVICE_NAME"),
                validator=validate_non_empty_string,
            ),
            EnvConfigKey.OTEL_LOGGING_ENABLED: ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "logging_enabled", v),
                getter=lambda: getenv("OTEL_LOGGING_ENABLED"),
            ),
            EnvConfigKey.OTEL_LOGS_EXPORTER: ConfigFieldDescriptor(
                target_type=Optional[AgentLoggingExporter],
                setter=lambda obj, v: setattr(obj, "logging_exporter", v),
                getter=lambda: getenv("OTEL_LOGS_EXPORTER"),
                validator=validate_otel_exporter_logging,
                should_raise=False,
                fallback=AgentLoggingExporter.CONSOLE,
            ),
            EnvConfigKey.OTEL_TRACING_ENABLED: ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "tracing_enabled", v),
                getter=lambda: getenv("OTEL_TRACING_ENABLED"),
            ),
            EnvConfigKey.OTEL_TRACES_EXPORTER: ConfigFieldDescriptor(
                target_type=Optional[AgentTracingExporter],
                setter=lambda obj, v: setattr(obj, "tracing_exporter", v),
                getter=lambda: getenv("OTEL_TRACES_EXPORTER"),
                validator=validate_otel_exporter_tracing,
                should_raise=False,
                fallback=AgentTracingExporter.CONSOLE,
            ),
        }

        config = cls._template_config()
        apply_config_map(config, config_field_map)

        return config

    @classmethod
    def from_instantiation(
        cls, instantiated_config: Optional["AgentObservabilityConfig"]
    ) -> "AgentObservabilityConfig":
        """
        Validate and create observability configuration from an instantiated configuration.

        Args:
            config: Optional user-instantiated configuration.

        Returns:
            AgentObservabilityConfig instance created from the instantiated configuration.
        """
        instantiated_config = instantiated_config or cls._template_config()
        config_field_map = {
            "enabled": ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "enabled", v),
                getter=lambda: instantiated_config.enabled,
            ),
            "headers": ConfigFieldDescriptor(
                target_type=dict[str, str],
                setter=lambda obj, v: setattr(obj, "headers", v),
                getter=lambda: instantiated_config.headers,
            ),
            "auth_token": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "auth_token", v),
                getter=lambda: instantiated_config.auth_token,
                validator=validate_non_empty_string,
            ),
            "endpoint": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "endpoint", v),
                getter=lambda: instantiated_config.endpoint,
                validator=validate_non_empty_string,
            ),
            "service_name": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "service_name", v),
                getter=lambda: instantiated_config.service_name,
                validator=validate_non_empty_string,
            ),
            "logging_enabled": ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "logging_enabled", v),
                getter=lambda: instantiated_config.logging_enabled,
            ),
            "logging_exporter": ConfigFieldDescriptor(
                target_type=Optional[AgentLoggingExporter],
                setter=lambda obj, v: setattr(obj, "logging_exporter", v),
                getter=lambda: instantiated_config.logging_exporter,
                validator=validate_otel_exporter_logging,
            ),
            "tracing_enabled": ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "tracing_enabled", v),
                getter=lambda: instantiated_config.tracing_enabled,
            ),
            "tracing_exporter": ConfigFieldDescriptor(
                target_type=Optional[AgentTracingExporter],
                setter=lambda obj, v: setattr(obj, "tracing_exporter", v),
                getter=lambda: instantiated_config.tracing_exporter,
                validator=validate_otel_exporter_tracing,
            ),
        }

        config = cls._template_config()
        apply_config_map(config, config_field_map)

        return config

    @classmethod
    def from_statestore(
        cls, runtime_config: Optional[Dict[str, Any]]
    ) -> "AgentObservabilityConfig":
        """
        Validate and create observability configuration from state store runtime configuration.

        Args:
            runtime_config: Optional state store runtime configuration.

        Returns:
            AgentObservabilityConfig instance created from state store runtime configuration.
        """
        runtime_config = runtime_config or {}
        config_field_map = {
            RuntimeConfigKey.OTEL_SDK_DISABLED: ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "enabled", v),
                getter=lambda: runtime_config.get("OTEL_SDK_DISABLED"),
                validator=lambda v: (
                    v if v is None else not v
                ),  # Invert the disabled flag to set enabled
            ),
            RuntimeConfigKey.OTEL_EXPORTER_OTLP_HEADERS: ConfigFieldDescriptor(
                target_type=Optional[str],
                # Target the auth_token field as runtime secrets may contain an access token
                setter=lambda obj, v: setattr(obj, "auth_token", v),
                getter=lambda: runtime_config.get("OTEL_EXPORTER_OTLP_HEADERS"),
                validator=validate_non_empty_string,
            ),
            RuntimeConfigKey.OTEL_EXPORTER_OTLP_ENDPOINT: ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "endpoint", v),
                getter=lambda: runtime_config.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
                validator=validate_non_empty_string,
            ),
            RuntimeConfigKey.OTEL_SERVICE_NAME: ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "service_name", v),
                getter=lambda: runtime_config.get("OTEL_SERVICE_NAME"),
                validator=validate_non_empty_string,
            ),
            RuntimeConfigKey.OTEL_LOGGING_ENABLED: ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "logging_enabled", v),
                getter=lambda: runtime_config.get("OTEL_LOGGING_ENABLED"),
            ),
            RuntimeConfigKey.OTEL_LOGS_EXPORTER: ConfigFieldDescriptor(
                target_type=Optional[AgentLoggingExporter],
                setter=lambda obj, v: setattr(obj, "logging_exporter", v),
                getter=lambda: runtime_config.get("OTEL_LOGS_EXPORTER"),
                validator=validate_otel_exporter_logging,
                should_raise=False,
                fallback=AgentLoggingExporter.CONSOLE,
            ),
            RuntimeConfigKey.OTEL_TRACING_ENABLED: ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "tracing_enabled", v),
                getter=lambda: runtime_config.get("OTEL_TRACING_ENABLED"),
            ),
            RuntimeConfigKey.OTEL_TRACES_EXPORTER: ConfigFieldDescriptor(
                target_type=Optional[AgentTracingExporter],
                setter=lambda obj, v: setattr(obj, "tracing_exporter", v),
                getter=lambda: runtime_config.get("OTEL_TRACES_EXPORTER"),
                validator=validate_otel_exporter_tracing,
                should_raise=False,
                fallback=AgentTracingExporter.CONSOLE,
            ),
        }

        config = cls._template_config()
        apply_config_map(config, config_field_map)

        return config

    @classmethod
    def resolve_config(
        cls,
        config: Optional["AgentObservabilityConfig"],
        runtime_config: Optional[Dict[str, Any]],
    ) -> "AgentObservabilityConfig":
        """
        Resolve the observability configuration for the agent in the following order:
        1. Passed through instantiation (highest priority)
        2. Environment variables
        3. State store runtime configuration (lowest priority)

        Args:
            config: Optional user-instantiated configuration.
            runtime_config: Optional state store runtime configuration.

        Returns:
            Resolved AgentObservabilityConfig instance.
        """
        statestore_config = AgentObservabilityConfig.from_statestore(runtime_config)
        logger.debug(f"State store runtime observability config: {statestore_config}")

        env_config = AgentObservabilityConfig.from_env()
        logger.debug(f"Environment variable observability config: {env_config}")

        instantiated_config = AgentObservabilityConfig.from_instantiation(config)
        logger.debug(f"Instantiated observability config: {instantiated_config}")

        resolved_config = functools.reduce(
            merge_models,
            [cls._base_config(), statestore_config, env_config, instantiated_config],
        )

        logger.debug(f"Final observability config: {resolved_config}")
        return resolved_config

    @classmethod
    def _template_config(cls) -> "AgentObservabilityConfig":
        """Create an observability configuration with defaults cleared.
        Used by environment variable and state store resolution, and as a fallback for instantiated configuration
        so that unset fields with default values do not bleed into the merged configuration.
        """
        return cls(
            enabled=None,
            headers={},
            auth_token=None,
            endpoint=None,
            service_name=None,
            logging_enabled=None,
            logging_exporter=None,
            tracing_enabled=None,
            tracing_exporter=None,
        )

    @classmethod
    def _base_config(cls) -> "AgentObservabilityConfig":
        """Create a base observability configuration for resolution."""
        return cls(
            enabled=False,
            headers={},
            auth_token=None,
            endpoint=None,
            service_name=None,
            logging_enabled=False,
            logging_exporter=AgentLoggingExporter.CONSOLE,
            tracing_enabled=False,
            tracing_exporter=AgentTracingExporter.CONSOLE,
        )


class AgentMetadata(BaseModel):
    """Metadata about an agent's configuration and capabilities."""

    appid: str = Field(
        ...,
        description="Dapr application ID (APP_ID) of the sidecar; may differ from the agent name",
    )
    type: str = Field(..., description="Type of the agent (e.g., standalone, durable)")
    orchestrator: bool = Field(
        False, description="Indicates if the agent is an orchestrator"
    )
    role: Optional[str] = Field(default=None, description="Role of the agent")
    goal: Optional[str] = Field(
        default=None, description="High-level objective of the agent"
    )
    instructions: Optional[List[str]] = Field(
        default=None, description="Instructions for the agent"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt guiding the agent's behavior"
    )
    framework: Optional[str] = Field(
        default=None, description="Framework or library the agent is built with"
    )
    max_iterations: Optional[int] = Field(
        default=None, description="Maximum iterations for agent execution"
    )
    tool_choice: Optional[str] = Field(default=None, description="Tool choice strategy")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional user-supplied metadata about the agent"
    )


class PubSubMetadata(BaseModel):
    """Pub/Sub configuration information."""

    resource_name: str = Field(..., description="Pub/Sub component name")
    broadcast_topic: Optional[str] = Field(
        default=None, description="Pub/Sub topic for broadcasting messages"
    )
    agent_topic: Optional[str] = Field(
        default=None, description="Pub/Sub topic for direct agent messages"
    )


class MemoryStoreMetadata(BaseModel):
    """Metadata about a single memory backing store."""

    type: str = Field(..., description="Implementation class name")
    resource_name: Optional[str] = Field(
        default=None, description="Dapr resource name for this store"
    )


class MemoryMetadata(BaseModel):
    """Memory configuration information."""

    short_term: Optional[MemoryStoreMetadata] = Field(
        default=None, description="Short-term workflow state store"
    )
    long_term: Optional[MemoryStoreMetadata] = Field(
        default=None, description="Long-term conversation memory store"
    )


class LLMMetadata(BaseModel):
    """LLM configuration information."""

    client: str = Field(..., description="LLM client used by the agent")
    provider: str = Field(..., description="LLM provider used by the agent")
    api: str = Field(default="unknown", description="API type used by the LLM client")
    model: str = Field(default="unknown", description="Model name or identifier")
    resource_name: Optional[str] = Field(
        default=None, description="Dapr resource name for the LLM client"
    )
    base_url: Optional[str] = Field(
        default=None, description="Base URL for the LLM API if applicable"
    )
    azure_endpoint: Optional[str] = Field(
        default=None, description="Azure endpoint if using Azure OpenAI"
    )
    azure_deployment: Optional[str] = Field(
        default=None, description="Azure deployment name if using Azure OpenAI"
    )
    prompt_template: Optional[str] = Field(
        default=None, description="Prompt template used by the agent"
    )


class ToolMetadata(BaseModel):
    """Metadata about a tool available to the agent."""

    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of the tool's functionality")
    args: str = Field(..., description="Arguments for the tool")


class RegistryMetadata(BaseModel):
    """Registry configuration information."""

    resource_name: Optional[str] = Field(
        None,
        description="Dapr resource name backing the registry (e.g. state store component)",
    )
    name: Optional[str] = Field(
        default=None, description="Logical team name the agent is registered under"
    )


class AgentMetadataSchema(BaseModel):
    """Schema for agent metadata including schema version."""

    version: str = Field(
        ...,
        description="Version of the schema used for the agent metadata.",
    )
    agent: AgentMetadata = Field(
        ..., description="Agent configuration and capabilities"
    )
    name: str = Field(
        ...,
        description="Logical agent name used as the registry key; distinct from agent.appid",
    )
    registered_at: str = Field(..., description="ISO 8601 timestamp of registration")
    pubsub: Optional[PubSubMetadata] = Field(
        None, description="Pub/sub configuration if enabled"
    )
    memory: Optional[MemoryMetadata] = Field(
        None, description="Memory configuration if enabled"
    )
    llm: Optional[LLMMetadata] = Field(None, description="LLM configuration")
    registry: Optional[RegistryMetadata] = Field(
        None, description="Registry configuration"
    )
    tools: Optional[List[ToolMetadata]] = Field(None, description="Available tools")

    @classmethod
    def export_json_schema(cls, version: str) -> Dict[str, Any]:
        """
        Export the JSON schema with version information.

        Args:
            version: The dapr-agents version for this schema

        Returns:
            JSON schema dictionary with metadata
        """
        schema = cls.model_json_schema()
        schema[_JSON_SCHEMA_KEY] = _JSON_SCHEMA_DRAFT_URL
        schema[_JSON_SCHEMA_VERSION_KEY] = version
        return schema
