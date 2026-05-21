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

import json
import logging
import re
from os import getenv
from enum import StrEnum
from dataclasses import dataclass, field, is_dataclass
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

from dapr_agents.agents.utils.models import (
    get_model_factory,
    get_model_fields,
    is_supported_config_model,
)
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


@dataclass(frozen=True)
class ConfigFieldDescriptor:
    """Describes how a configuration key maps to an agent attribute.

    Attributes:
        target_type: Expected Python type for the coerced value.
        setter: Callable ``(agent, value) -> None`` that applies the value.
        getter: Optional callable ``() -> Any`` that retrieves the value.
        sensitive: If ``True``, the value is redacted in log output.
        validator: Optional idempotent callable ``(value) -> Any`` to validate/transform the coerced value.
        rebuilds_prompt: If ``True``, the prompt template is rebuilt after update.
    """

    target_type: Type
    setter: Callable[..., None]
    getter: Optional[Callable[[], Any]] = None
    sensitive: bool = False
    validator: Optional[Callable[..., Any]] = None
    rebuilds_prompt: bool = False
    triggers_otel_reload: bool = False


# ---------------------------------------------------------------------------
# Built-in validators for ConfigFieldDescriptor
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


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def apply_config_update(
    target_obj: Any,
    key: str,
    value: Any,
    descriptor: ConfigFieldDescriptor,
) -> Any:
    """
    Process and apply a configuration update to an object.
    This function is guaranteed to be idempotent if the processing logic is idempotent.

    Args:
        target_obj: The object to be updated.
        key: The configuration key.
        value: Optional value to process and apply.
            Falls back to the descriptor's getter if not provided (may not be idempotent).
        descriptor: An object describing how to process a value for a particular key.

    Returns:
        The final applied value.

    Raises:
        ValueError: If no value can be retrieved or processing fails.
        RuntimeError: If the value cannot be applied.
    """

    processed_value = process_config_update(key, value, descriptor)

    # Apply via setter callback
    try:
        descriptor.setter(target_obj, processed_value)
    except (AttributeError, TypeError):
        raise RuntimeError(
            f"Could not apply setter for key '{key}' (likely read-only)."
        )

    return processed_value


def process_config_update(
    key: str,
    value: Any,
    descriptor: ConfigFieldDescriptor,
) -> Any:
    """
    Process a configuration update by coercing, validating, and transforming a value.
    This function is guaranteed to be idempotent if the processing logic is idempotent.

    Args:
        key: The configuration key.
        value: Optional value to process.
            Falls back to the descriptor's getter if not provided (may not be idempotent).
        descriptor: An object describing how to process a value for a particular key.

    Returns:
        The processed value.

    Raises:
        ValueError: If no value can be retrieved or processing fails.
    """

    if not descriptor:
        raise ValueError(f"Unrecognized config key: {key}.")

    # Retrieve value using getter callback as a fallback
    if not value and descriptor.getter:
        try:
            value = descriptor.getter()
        except Exception as e:
            raise ValueError(f"Unable to retrieve value for key '{key}': {e}.")

    # Type coercion
    try:
        processed_value = coerce_config_value(value, descriptor.target_type)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid value for key '{key}': {e}.")

    # Validation/transformation
    if descriptor.validator is not None:
        try:
            processed_value = descriptor.validator(processed_value)
        except Exception as e:
            raise ValueError(f"Validation failed for key '{key}': {e}.")

    return processed_value


def coerce_config_value(value: Any, target_type: Type) -> Any:
    """Coerce a configuration value (usually a string) to the target Python type."""
    if isinstance(value, target_type):
        return value

    if target_type is str:
        return str(value)

    if target_type is int:
        return int(float(value))

    if target_type is float:
        return float(value)

    if target_type is bool:
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no"):
                return False
        raise ValueError(f"Cannot coerce {value!r} to bool")

    if target_type is list:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    if target_type is dict:
        if isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError(f"JSON parsed to {type(parsed).__name__}, expected dict")
        if isinstance(value, dict):
            return value
        raise ValueError(f"Cannot coerce {type(value).__name__} to dict")

    raise ValueError(f"Unsupported target type: {target_type}")


def merge_configs(base: T, override: T) -> T:
    """
    Merge two configuration models of the same type, with override taking precedence.
    Only override if the override value is not None.

    Args:
        base: The original configuration model.
        override: The new configuration model with potential override values.

    Returns:
        The merged configuration model.

    Raises:
        TypeError: If models are of incompatible types or unsupported type.
        ValueError: If merging fails.
    """
    # NOTE: this implementation doesn't handle override values that are explicitly None

    if not is_supported_config_model(type(base)):
        raise TypeError(f"Unsupported model type: {base!r}")

    if not is_supported_config_model(type(override)):
        raise TypeError(f"Unsupported model type: {override!r}")

    if base.__class__ != override.__class__:
        raise TypeError(
            f"Cannot merge models of different types: {base!r} and {override!r}"
        )

    # Infer model type from the base model
    model_fields = get_model_fields(base)
    model_factory = get_model_factory(base)

    config: Dict[str, Any] = {}

    for model_field in model_fields:
        base_field = getattr(base, model_field)
        override_field = getattr(override, model_field)

        if isinstance(base_field, dict) and isinstance(override_field, dict):
            # Shallow merge dicts
            config[model_field] = {**base_field, **override_field}
        else:
            config[model_field] = (
                override_field if override_field is not None else base_field
            )

    return model_factory(config)


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
        app_health_check_enabled: Enable/disable Kubernetes liveness probes.
        approval: Human-in-the-loop configuration for the agent.
        app_ready_check_enabled: Enable/disable Dapr health/Kubernetes readiness probes.
    """

    # TODO: add a forceFinalAnswer field in case max_iterations is near/reached. Or do we have a conclusion baked in by default? Do we want this to derive a conclusion by default?
    # TODO: add stop_at_tokens
    max_iterations: int = AGENT_DEFAULT_MAX_ITERATIONS
    tool_choice: Optional[ToolChoice] = AGENT_DEFAULT_TOOL_CHOICE
    tool_execution_mode: ToolExecutionMode = AGENT_DEFAULT_TOOL_EXECUTION_MODE
    orchestration_mode: Optional[OrchestrationMode] = None
    approval: AgentApprovalConfig = field(default_factory=AgentApprovalConfig)

    app_health_check_enabled: Optional[bool] = None
    app_ready_check_enabled: Optional[bool] = None

    @classmethod
    def from_env(cls) -> "AgentExecutionConfig":
        """Create execution config from environment variables."""

        max_iterations: Optional[int] = None
        if max_iterations_str := getenv("MAX_ITERATIONS"):
            try:
                max_iterations = max(1, int(max_iterations_str))
            except ValueError:
                max_iterations = AGENT_DEFAULT_MAX_ITERATIONS

        tool_choice: Optional[ToolChoice] = None
        if tool_choice_str := getenv("TOOL_CHOICE"):
            try:
                tool_choice = ToolChoice(tool_choice_str)
            except (ValueError, KeyError):
                tool_choice = AGENT_DEFAULT_TOOL_CHOICE

        tool_execution_mode: Optional[ToolExecutionMode] = None
        if tool_execution_mode_str := getenv("TOOL_EXECUTION_MODE"):
            try:
                tool_execution_mode = ToolExecutionMode(tool_execution_mode_str)
            except (ValueError, KeyError):
                tool_execution_mode = AGENT_DEFAULT_TOOL_EXECUTION_MODE

        orchestration_mode: Optional[OrchestrationMode] = None
        if orchestration_mode_str := getenv("ORCHESTRATION_MODE"):
            try:
                orchestration_mode = OrchestrationMode(orchestration_mode_str)
            except (ValueError, KeyError):
                orchestration_mode = None

        app_health_check_enabled: Optional[bool] = None
        if getenv("ENABLE_APP_HEALTH_CHECK") is not None:
            app_health_check_enabled = (
                getenv("ENABLE_APP_HEALTH_CHECK", "false").lower() == "true"
            )

        app_ready_check_enabled: Optional[bool] = None
        if getenv("ENABLE_APP_READY_CHECK") is not None:
            app_ready_check_enabled = (
                getenv("ENABLE_APP_READY_CHECK", "false").lower() == "true"
            )

        return cls(
            max_iterations=max_iterations,
            tool_choice=tool_choice,
            tool_execution_mode=tool_execution_mode,
            orchestration_mode=orchestration_mode,
            app_health_check_enabled=app_health_check_enabled,
            app_ready_check_enabled=app_ready_check_enabled,
        )

    @classmethod
    def from_statestore(cls, config: Dict[str, Any]) -> "AgentExecutionConfig":
        """
        Load execution configuration from the state store.

        Returns:
            AgentExecutionConfig instance loaded from state store.
        """

        try:
            max_iterations: Optional[int] = None
            if max_iterations_str := config.get("MAX_ITERATIONS"):
                try:
                    max_iterations = max(1, int(max_iterations_str))
                except ValueError:
                    max_iterations = AGENT_DEFAULT_MAX_ITERATIONS

            tool_choice: Optional[ToolChoice] = None
            if tool_choice_str := config.get("TOOL_CHOICE"):
                try:
                    tool_choice = ToolChoice(tool_choice_str)
                except (ValueError, KeyError):
                    tool_choice = AGENT_DEFAULT_TOOL_CHOICE

            tool_execution_mode: Optional[ToolExecutionMode] = None
            orchestration_mode: Optional[OrchestrationMode] = None
            approval: Optional[AgentApprovalConfig] = None
            app_health_check_enabled: Optional[bool] = None
            app_ready_check_enabled: Optional[bool] = None

            return AgentExecutionConfig(
                max_iterations=max_iterations,
                tool_choice=tool_choice,
                tool_execution_mode=tool_execution_mode,
                orchestration_mode=orchestration_mode,
                approval=approval,
                app_health_check_enabled=app_health_check_enabled,
                app_ready_check_enabled=app_ready_check_enabled,
            )
        except Exception:
            return AgentExecutionConfig()

    def resolve_config(self, runtime_config: Dict[str, Any]) -> "AgentExecutionConfig":
        """
        Resolve the execution configuration for the agent in the following order:
        1. Statestore runtime config (highest priority)
        2. Passed through instantiation
        3. Environment variables (lowest priority)

        Args:
            runtime_config: Runtime configuration.

        Returns:
            Resolved AgentExecutionConfig instance for fluent chaining.
        """

        config = AgentExecutionConfig.from_env()
        logger.debug(f"Env execution config: {config}")

        try:
            config = merge_configs(config, self)
        except Exception as e:
            logger.warning(f"Failed to merge execution config with env execution config: {e}")
            config = self

        logger.debug(f"Merged execution config: {config}")

        statestore_config = AgentExecutionConfig.from_statestore(runtime_config)
        logger.debug(f"Statestore execution config: {statestore_config}")

        try:
            config = merge_configs(config, statestore_config)
        except Exception as e:
            logger.warning(f"Failed to merge execution config with statestore execution config: {e}")
            config = self

        logger.debug(f"Final execution config: {config}")

        for k, v in config.__dict__.items():
            setattr(self, k, v)

        return self


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


class RuntimeConfigKey(StrEnum):
    """Supported keys for runtime configuration hot-reload.

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
        """Create observability config from standard OTEL environment variables.

        Uses standard OpenTelemetry env var names where available:
        - OTEL_SDK_DISABLED (inverted: disabled != "true" means enabled)
        - OTEL_EXPORTER_OTLP_ENDPOINT
        - OTEL_EXPORTER_OTLP_HEADERS (parses "Authorization=<token>" format)
        - OTEL_SERVICE_NAME
        - OTEL_TRACES_EXPORTER
        - OTEL_LOGS_EXPORTER
        - OTEL_TRACING_ENABLED (custom, no standard equivalent)
        - OTEL_LOGGING_ENABLED (custom, no standard equivalent)
        """
        headers: Dict[str, str] = {}
        raw_headers = getenv("OTEL_EXPORTER_OTLP_HEADERS")
        if raw_headers:
            for pair in raw_headers.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    headers[k.strip()] = v.strip()

        logging_exporter: Optional[AgentLoggingExporter] = None
        if logging_exporter_str := getenv("OTEL_LOGS_EXPORTER"):
            try:
                logging_exporter = AgentLoggingExporter(logging_exporter_str)
            except (ValueError, KeyError):
                logging_exporter = AgentLoggingExporter.CONSOLE

        tracing_exporter: Optional[AgentTracingExporter] = None
        if tracing_exporter_str := getenv("OTEL_TRACES_EXPORTER"):
            try:
                tracing_exporter = AgentTracingExporter(tracing_exporter_str)
            except (ValueError, KeyError):
                tracing_exporter = AgentTracingExporter.CONSOLE

        enabled: Optional[bool] = None
        if getenv("OTEL_SDK_DISABLED") is not None:
            enabled = getenv("OTEL_SDK_DISABLED", "false").lower() != "true"

        logging_enabled: Optional[bool] = None
        if getenv("OTEL_LOGGING_ENABLED") is not None:
            logging_enabled = getenv("OTEL_LOGGING_ENABLED", "false").lower() == "true"

        tracing_enabled: Optional[bool] = None
        if getenv("OTEL_TRACING_ENABLED") is not None:
            tracing_enabled = getenv("OTEL_TRACING_ENABLED", "false").lower() == "true"

        return cls(
            enabled=enabled,
            headers=headers,
            endpoint=getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            service_name=getenv("OTEL_SERVICE_NAME"),
            logging_enabled=logging_enabled,
            logging_exporter=logging_exporter,
            tracing_enabled=tracing_enabled,
            tracing_exporter=tracing_exporter,
        )

    @classmethod
    def from_statestore(cls, config: Dict[str, Any]) -> "AgentObservabilityConfig":
        """
        Load observability configuration from the state store.

        Returns:
            AgentObservabilityConfig instance loaded from state store.
        """

        try:
            # Use standard OTEL env var names in statestore config
            sdk_disabled = config.get("OTEL_SDK_DISABLED", "true").lower()
            enabled = sdk_disabled != "true"
            auth_token = (
                config.get("OTEL_EXPORTER_OTLP_HEADERS")
                or config.get("OTEL_EXPORTER_OTLP_HEADERS")
                or None
            )
            endpoint = config.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None
            service_name = config.get("OTEL_SERVICE_NAME") or None
            logging_enabled = (
                config.get("OTEL_LOGGING_ENABLED", "false").lower() == "true"
            )
            tracing_enabled = (
                config.get("OTEL_TRACING_ENABLED", "false").lower() == "true"
            )

            logging_exporter: Optional[AgentLoggingExporter] = None
            logging_exporter_str = config.get("OTEL_LOGS_EXPORTER", "console")
            if logging_exporter_str:
                try:
                    logging_exporter = AgentLoggingExporter(logging_exporter_str)
                except (ValueError, KeyError):
                    logging_exporter = AgentLoggingExporter.CONSOLE

            tracing_exporter: Optional[AgentTracingExporter] = None
            tracing_exporter_str = config.get("OTEL_TRACES_EXPORTER", "console")
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
        except Exception:
            return AgentObservabilityConfig()

    def resolve_config(
        self, runtime_config: Dict[str, Any]
    ) -> "AgentObservabilityConfig":
        """
        Resolve the observability configuration for the agent in the following order:
        1. Passed through instantiation (highest priority)
        2. Environment variables
        3. Default statestore runtime config (lowest priority)

        Args:
            runtime_config: Runtime configuration.

        Returns:
            Resolved AgentObservabilityConfig instance for fluent chaining.
        """

        config = AgentObservabilityConfig.from_statestore(runtime_config)
        logger.debug(f"Statestore observability config: {config}")

        env_config = AgentObservabilityConfig.from_env()
        logger.debug(f"Env observability config: {env_config}")

        try:
            config = merge_configs(config, env_config)
        except Exception as e:
            logger.warning(f"Failed to merge observability config with env observability config: {e}")
            config = self

        logger.debug(f"Merged observability config: {config}")

        try:
            config = merge_configs(config, self)
        except Exception as e:
            logger.warning(f"Failed to merge observability config with statestore observability config: {e}")
            config = self

        logger.debug(f"Final observability config: {config}")

        for k, v in config.__dict__.items():
            setattr(self, k, v)

        return self


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
