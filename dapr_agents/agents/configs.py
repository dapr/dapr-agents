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
import json
import logging
import re
from os import getenv
from enum import Enum, StrEnum
from dataclasses import dataclass, field
from types import UnionType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field

from dapr_agents.agents.utils.headers import parse_header_string
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

# Sentinel value for unset values (instead of None)
UNSET = object()

# Sentinel value for unsupported config keys
_UNSUPPORTED = object()


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
    """Describes how a configuration key maps to a configuration attribute.

    Attributes:
        target_type: Expected Python type for the coerced value.
        setter: Callable ``(obj, value) -> None`` that applies the value after coercion and validation.
        getter: Optional callable ``() -> Any`` that retrieves the value before coercion.
        validator: Optional idempotent callable ``(value) -> Any`` to validate/transform the coerced value.
        allow_unset: If ``True``, coercion will pass through ``UNSET`` values (to disambiguate between an explicitly set ``None`` and an unset value that defaults to ``None``).
            If enabled and a validator is provided, it must be able to handle ``UNSET`` values.
        sensitive: If ``True``, the value is redacted in log output.
        rebuilds_prompt: If ``True``, the prompt template is rebuilt after update.
        triggers_otel_reload: If ``True``, triggers an OpenTelemetry configuration reload after update.
    """

    target_type: Type
    setter: Callable[..., None]
    getter: Optional[Callable[[], Any]] = None
    validator: Optional[Callable[..., Any]] = None
    allow_unset: bool = False
    sensitive: bool = False
    rebuilds_prompt: bool = False
    triggers_otel_reload: bool = False


# ---------------------------------------------------------------------------
# Built-in ConfigFieldDescriptor validators for agents
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


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def normalize_config_key(key: str) -> str:
    """Default normalization of configuration keys to attribute names."""
    return key.lower().replace("-", "_")


def apply_config_map(target_obj: Any, config_field_map: Dict[str, Any]) -> None:
    """
    Apply a map of configuration field names to field descriptors onto a target object.
    If a field is ``UNSUPPORTED``, its name must be normalizable to the corresponding attribute on the target object, whose value is set to None.
    """
    for key, descriptor in config_field_map.items():
        if descriptor == _UNSUPPORTED:
            setattr(target_obj, normalize_config_key(key), None)
            continue

        try:
            apply_config_update(target_obj=target_obj, key=key, descriptor=descriptor)
        except (ValueError, RuntimeError) as e:
            logger.debug(f"Failed to apply config update for {key}: {e}")


def apply_config_update(
    target_obj: Any,
    *,
    key: str,
    descriptor: ConfigFieldDescriptor,
    value: Any = None,
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
    processed_value = process_config_update(key=key, value=value, descriptor=descriptor)

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
    descriptor: ConfigFieldDescriptor,
    value: Any = None,
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
    if value is None and descriptor.getter:
        try:
            value = descriptor.getter()
        except Exception as e:
            raise ValueError(f"Unable to retrieve value for key '{key}': {e}.")

    # Type coercion
    try:
        if descriptor.allow_unset and value is UNSET:
            # Pass through `UNSET` unless we do not have a validator, in which case we coerce to `None` and return immediately
            if not descriptor.validator:
                return None
            processed_value = UNSET
        else:
            processed_value = coerce_config_value(value, descriptor.target_type)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid value for key '{key}': {e}.")

    # Validation/transformation
    if descriptor.validator:
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

    # Handle types that are not classes
    origin = get_origin(target_type)
    if origin is not None:
        # Support union and bar syntax
        if origin in (Union, UnionType):
            for arg in get_args(target_type):
                try:
                    return coerce_config_value(value, arg)
                except ValueError:
                    continue
            raise ValueError(f"Cannot coerce {value!r} to any type in {target_type}")

    raise ValueError(f"Unsupported target type: {target_type}")


def merge_models(base: T, override: T) -> T:
    """
    Merge two models of the same type, with override taking precedence.
    Only override if the override value is not None.
    If merging fails, falls back to the base model if it is valid, otherwise falls back to the override model.

    Args:
        base: The base model.
        override: The new model with potential override values.

    Returns:
        The merged model.
    """
    if not is_supported_config_model(type(base)):
        logger.warning(f"Unsupported model type: {base!r}")
        return override

    if not is_supported_config_model(type(override)):
        logger.warning(f"Unsupported model type: {override!r}")
        return base

    if base.__class__ != override.__class__:
        logger.warning(
            f"Cannot merge models of different types: {base!r} and {override!r}"
        )
        return base

    try:
        # Infer model type from the base model
        model_fields = get_model_fields(base)
        model_factory = get_model_factory(base)

        logger.debug(
            (f"Merging models:\nBase model: {base!r}\nOverride model: {override!r}")
        )

        merged_fields: Dict[str, Any] = {}

        for model_field in model_fields:
            base_field = getattr(base, model_field)
            override_field = getattr(override, model_field)

            if isinstance(base_field, dict) and isinstance(override_field, dict):
                # Shallow merge dicts
                merged_fields[model_field] = {**base_field, **override_field}
            else:
                merged_fields[model_field] = (
                    override_field if override_field is not None else base_field
                )

        model = model_factory(merged_fields)

        logger.debug(f"Merged model: {model!r}")
        return model
    except Exception as e:
        logger.warning(f"Failed to merge models: {e}")
        return base


def _validate_with_fallback(
    f: Callable[..., Any], value: Any, fallback: Any = UNSET
) -> Callable[..., Any]:
    """
    Calls a validation function with a fallback return value if the function raises an exception.
    If the value or fallback is `UNSET`, returns `None` so that merging works properly.
    """
    if value is UNSET:
        return None
    try:
        return f(value)
    except Exception as e:
        if fallback is UNSET:
            fallback = None
        # Avoid noisy logs if validation errors are frequent
        logger.debug(
            f"Validation with function {f.__name__} failed: {e}. Using fallback value {fallback!r}."
        )
        return fallback


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
        app_health_check_enabled: Enable/disable Kubernetes liveness probes.
        approval: Human-in-the-loop configuration for the agent.
        app_ready_check_enabled: Enable/disable Dapr health/Kubernetes readiness probes.
    """

    # TODO: add a forceFinalAnswer field in case max_iterations is near/reached. Or do we have a conclusion baked in by default? Do we want this to derive a conclusion by default?
    # TODO: add stop_at_tokens
    max_iterations: Optional[int] = AGENT_DEFAULT_MAX_ITERATIONS
    tool_choice: Optional[ToolChoice] = AGENT_DEFAULT_TOOL_CHOICE
    tool_execution_mode: Optional[ToolExecutionMode] = AGENT_DEFAULT_TOOL_EXECUTION_MODE
    orchestration_mode: Optional[OrchestrationMode] = None
    approval: Optional[AgentApprovalConfig] = field(default_factory=AgentApprovalConfig)
    max_grpc_inbound_message_size_bytes: Optional[int] = None
    app_health_check_enabled: Optional[bool] = None
    app_ready_check_enabled: Optional[bool] = None

    @classmethod
    def from_env(cls) -> "AgentExecutionConfig":
        """Create execution config from environment variables."""
        config_field_map = {
            "max_iterations": ConfigFieldDescriptor(
                target_type=Optional[int],
                setter=lambda obj, v: setattr(obj, "max_iterations", v),
                getter=lambda: getenv("MAX_ITERATIONS", AGENT_DEFAULT_MAX_ITERATIONS),
                validator=lambda v: _validate_with_fallback(
                    validate_max_iterations, v, AGENT_DEFAULT_MAX_ITERATIONS
                ),
            ),
            "tool_choice": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "tool_choice", v),
                getter=lambda: getenv("TOOL_CHOICE", AGENT_DEFAULT_TOOL_CHOICE),
                validator=lambda v: _validate_with_fallback(
                    validate_tool_choice, v, AGENT_DEFAULT_TOOL_CHOICE
                ),
            ),
            "tool_execution_mode": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "tool_execution_mode", v),
                getter=lambda: getenv(
                    "TOOL_EXECUTION_MODE", AGENT_DEFAULT_TOOL_EXECUTION_MODE
                ),
                validator=lambda v: _validate_with_fallback(
                    validate_tool_execution_mode, v, AGENT_DEFAULT_TOOL_EXECUTION_MODE
                ),
            ),
            "orchestration_mode": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "orchestration_mode", v),
                getter=lambda: getenv("ORCHESTRATION_MODE", None),
                validator=lambda v: _validate_with_fallback(
                    validate_orchestration_mode, v, None
                ),
            ),
            "max_grpc_inbound_message_size_bytes": ConfigFieldDescriptor(
                target_type=Optional[int],
                setter=lambda obj, v: setattr(
                    obj, "max_grpc_inbound_message_size_bytes", v
                ),
                getter=lambda: getenv("MAX_GRPC_INBOUND_MESSAGE_SIZE_BYTES", UNSET),
                allow_unset=True,
            ),
            "app_health_check_enabled": ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "app_health_check_enabled", v),
                getter=lambda: getenv("ENABLE_APP_HEALTH_CHECK", "false"),
            ),
            "app_ready_check_enabled": ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "app_ready_check_enabled", v),
                getter=lambda: getenv("ENABLE_APP_READY_CHECK", "false"),
            ),
            "approval": _UNSUPPORTED,
        }

        config = cls()
        apply_config_map(config, config_field_map)

        return config

    @classmethod
    def from_statestore(cls, runtime_config: Dict[str, Any]) -> "AgentExecutionConfig":
        """
        Load execution configuration from the state store.

        Returns:
            AgentExecutionConfig instance loaded from state store.
        """
        config_field_map = {
            "max_iterations": ConfigFieldDescriptor(
                target_type=Optional[int],
                setter=lambda obj, v: setattr(obj, "max_iterations", v),
                getter=lambda: runtime_config.get("MAX_ITERATIONS", UNSET),
                validator=lambda v: _validate_with_fallback(
                    validate_max_iterations, v, AGENT_DEFAULT_MAX_ITERATIONS
                ),
                allow_unset=True,
            ),
            "tool_choice": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "tool_choice", v),
                getter=lambda: runtime_config.get("TOOL_CHOICE", UNSET),
                validator=lambda v: _validate_with_fallback(
                    validate_tool_choice, v, AGENT_DEFAULT_TOOL_CHOICE
                ),
                allow_unset=True,
            ),
            "tool_execution_mode": _UNSUPPORTED,
            "orchestration_mode": _UNSUPPORTED,
            "approval": _UNSUPPORTED,
            "max_grpc_inbound_message_size_bytes": _UNSUPPORTED,
            "app_health_check_enabled": _UNSUPPORTED,
            "app_ready_check_enabled": _UNSUPPORTED,
        }

        config = cls()
        apply_config_map(config, config_field_map)

        return config

    @classmethod
    def resolve_config(
        cls, config: "AgentExecutionConfig", runtime_config: Dict[str, Any]
    ) -> "AgentExecutionConfig":
        """
        Resolve the execution configuration for the agent in the following order:
        1. Statestore runtime config (highest priority)
        2. Passed through instantiation
        3. Environment variables (lowest priority)

        Args:
            config: User-instantiated configuration.
            runtime_config: Runtime configuration.

        Returns:
            Resolved AgentExecutionConfig instance.
        """

        env_config = AgentExecutionConfig.from_env()
        logger.debug(f"Env execution config: {env_config}")

        logger.debug(f"Instantiated execution config: {config}")

        statestore_config = AgentExecutionConfig.from_statestore(runtime_config)
        logger.debug(f"Statestore execution config: {statestore_config}")

        resolved_config = functools.reduce(
            merge_models,
            [env_config, config, statestore_config],
        )

        logger.debug(f"Final execution config: {resolved_config}")
        return resolved_config


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
        - OTEL_EXPORTER_OTLP_HEADERS (parses "Authorization=<token>" format)
        - OTEL_EXPORTER_OTLP_ENDPOINT
        - OTEL_SERVICE_NAME
        - OTEL_LOGGING_ENABLED (custom, no standard equivalent)
        - OTEL_LOGS_EXPORTER
        - OTEL_TRACING_ENABLED (custom, no standard equivalent)
        - OTEL_TRACES_EXPORTER
        """
        config_field_map = {
            "enabled": ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "enabled", v),
                getter=lambda: getenv("OTEL_SDK_DISABLED", UNSET),
                validator=lambda v: _validate_with_fallback(
                    lambda v: not v, v, UNSET
                ),  # Invert the disabled flag to set enabled
                allow_unset=True,
            ),
            "headers": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "headers", v),
                getter=lambda: getenv("OTEL_EXPORTER_OTLP_HEADERS", UNSET),
                validator=lambda v: _validate_with_fallback(parse_header_string, v, {}),
                allow_unset=True,
            ),
            "endpoint": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "endpoint", v),
                getter=lambda: getenv("OTEL_EXPORTER_OTLP_ENDPOINT", UNSET),
                validator=lambda v: _validate_with_fallback(
                    validate_non_empty_string, v, UNSET
                ),
                allow_unset=True,
            ),
            "service_name": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "service_name", v),
                getter=lambda: getenv("OTEL_SERVICE_NAME", UNSET),
                validator=lambda v: _validate_with_fallback(
                    validate_non_empty_string, v, UNSET
                ),
                allow_unset=True,
            ),
            "logging_enabled": ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "logging_enabled", v),
                getter=lambda: getenv("OTEL_LOGGING_ENABLED", UNSET),
                allow_unset=True,
            ),
            "logging_exporter": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "logging_exporter", v),
                getter=lambda: getenv("OTEL_LOGS_EXPORTER", UNSET),
                validator=lambda v: _validate_with_fallback(
                    validate_otel_exporter_logging, v, AgentLoggingExporter.CONSOLE
                ),
                allow_unset=True,
            ),
            "tracing_enabled": ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "tracing_enabled", v),
                getter=lambda: getenv("OTEL_TRACING_ENABLED", UNSET),
                allow_unset=True,
            ),
            "tracing_exporter": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "tracing_exporter", v),
                getter=lambda: getenv("OTEL_TRACES_EXPORTER", UNSET),
                validator=lambda v: _validate_with_fallback(
                    validate_otel_exporter_tracing, v, AgentTracingExporter.CONSOLE
                ),
                allow_unset=True,
            ),
        }

        config = cls()
        apply_config_map(config, config_field_map)

        return config

    @classmethod
    def from_statestore(
        cls, runtime_secrets: Dict[str, Any], runtime_config: Dict[str, Any]
    ) -> "AgentObservabilityConfig":
        """
        Load observability configuration from the state store.

        Returns:
            AgentObservabilityConfig instance loaded from state store.
        """
        config_field_map = {
            "enabled": ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "enabled", v),
                getter=lambda: runtime_config.get("OTEL_SDK_DISABLED", "true"),
                validator=lambda v: _validate_with_fallback(
                    lambda v: not v, v, UNSET
                ),  # Invert the disabled flag to set enabled
            ),
            "auth_token": ConfigFieldDescriptor(
                target_type=Optional[str],
                # State store may have auth credentials so we target the "auth_token" field
                setter=lambda obj, v: setattr(obj, "auth_token", v),
                getter=lambda: (
                    runtime_secrets.get("OTEL_EXPORTER_OTLP_HEADERS")
                    or runtime_config.get("OTEL_EXPORTER_OTLP_HEADERS")
                    or UNSET
                ),
                allow_unset=True,
            ),
            "endpoint": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "endpoint", v),
                getter=lambda: runtime_config.get("OTEL_EXPORTER_OTLP_ENDPOINT", UNSET),
                validator=lambda v: _validate_with_fallback(
                    validate_non_empty_string, v, UNSET
                ),
                allow_unset=True,
            ),
            "service_name": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "service_name", v),
                getter=lambda: runtime_config.get("OTEL_SERVICE_NAME", UNSET),
                validator=lambda v: _validate_with_fallback(
                    validate_non_empty_string, v, UNSET
                ),
                allow_unset=True,
            ),
            "logging_enabled": ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "logging_enabled", v),
                getter=lambda: runtime_config.get("OTEL_LOGGING_ENABLED", "false"),
            ),
            "logging_exporter": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "logging_exporter", v),
                getter=lambda: runtime_config.get(
                    "OTEL_LOGS_EXPORTER", AgentLoggingExporter.CONSOLE
                ),
                validator=lambda v: _validate_with_fallback(
                    validate_otel_exporter_logging, v, AgentLoggingExporter.CONSOLE
                ),
            ),
            "tracing_enabled": ConfigFieldDescriptor(
                target_type=Optional[bool],
                setter=lambda obj, v: setattr(obj, "tracing_enabled", v),
                getter=lambda: runtime_config.get("OTEL_TRACING_ENABLED", "false"),
            ),
            "tracing_exporter": ConfigFieldDescriptor(
                target_type=Optional[str],
                setter=lambda obj, v: setattr(obj, "tracing_exporter", v),
                getter=lambda: runtime_config.get(
                    "OTEL_TRACES_EXPORTER", AgentTracingExporter.CONSOLE
                ),
                validator=lambda v: _validate_with_fallback(
                    validate_otel_exporter_tracing, v, AgentTracingExporter.CONSOLE
                ),
            ),
        }

        config = cls()
        apply_config_map(config, config_field_map)

        return config

    @classmethod
    def resolve_config(
        cls,
        config: "AgentObservabilityConfig",
        runtime_secrets: Dict[str, Any],
        runtime_config: Dict[str, Any],
    ) -> "AgentObservabilityConfig":
        """
        Resolve the observability configuration for the agent in the following order:
        1. Passed through instantiation (highest priority)
        2. Environment variables
        3. Default statestore runtime config (lowest priority)

        Args:
            config: User-instantiated configuration.
            runtime_secrets: Runtime secrets.
            runtime_config: Runtime configuration.

        Returns:
            Resolved AgentObservabilityConfig instance.
        """
        statestore_config = AgentObservabilityConfig.from_statestore(
            runtime_secrets, runtime_config
        )
        logger.debug(f"Statestore observability config: {statestore_config}")

        env_config = AgentObservabilityConfig.from_env()
        logger.debug(f"Env observability config: {env_config}")

        logger.debug(f"Instantiated observability config: {config}")

        resolved_config = functools.reduce(
            merge_models,
            [statestore_config, env_config, config],
        )

        logger.debug(f"Final observability config: {resolved_config}")
        return resolved_config


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
