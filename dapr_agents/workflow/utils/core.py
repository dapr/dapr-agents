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

import asyncio
import inspect
import logging
import signal
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Optional, Type
import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext
from pydantic import BaseModel

from dapr_agents.tool.workflow.agent_tool import agent_workflow_id
from dapr_agents.tool.utils.function_calling import sanitize_openai_tool_name

logger = logging.getLogger(__name__)


async def wait_for_shutdown() -> None:
    """Block until Ctrl+C or SIGTERM is received."""
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _set_stop(*_: object) -> None:
        stop.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _set_stop)
        loop.add_signal_handler(signal.SIGTERM, _set_stop)
    except (NotImplementedError, RuntimeError):
        # Windows fallback where add_signal_handler is unavailable
        signal.signal(signal.SIGINT, lambda *_: _set_stop())
        signal.signal(signal.SIGTERM, lambda *_: _set_stop())

    await stop.wait()


def _registry_lookup(agent_name: str, registry: Any) -> Dict[str, Any]:
    """Return the agent's metadata dict from the registry, or {} if not found."""
    try:
        agents = registry.get_agents_metadata(exclude_self=False)
        meta = agents.get(agent_name, {})
        return meta.get("agent", {}) if isinstance(meta, dict) else {}
    except Exception:
        return {}


class _TriggerWorkflow:
    """
    Callable workflow that triggers a target agent as a child workflow.
    """

    def __init__(
        self,
        trigger_name: str,
        workflow_name: str,
        input: Any,
        app_id: str,
    ) -> None:
        self.__name__ = trigger_name
        self._workflow_name = workflow_name
        self._input = input
        self._app_id = app_id

    def __call__(self, ctx: DaprWorkflowContext) -> Any:
        result = yield ctx.call_child_workflow(
            workflow=self._workflow_name,
            input=self._input,
            app_id=self._app_id,
        )
        return result


def _resolve_registry(
    name: str,
    app_id: Optional[str],
    framework: Optional[str],
    registry: Optional[Any],
) -> tuple[Optional[str], Optional[str]]:
    """Look up missing app_id / framework from the registry."""
    if (app_id is None or framework is None) and registry is not None:
        agent_info = _registry_lookup(name, registry)
        if app_id is None:
            app_id = agent_info.get("appid")
        if framework is None:
            framework = agent_info.get("framework")
    return app_id, framework


def call_agent(
    ctx: DaprWorkflowContext,
    name: str,
    input: Any,
    *,
    app_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    framework: Optional[str] = None,
    registry: Optional[Any] = None,
) -> Any:
    """
    Call a DurableAgent's workflow as a child workflow from within another Dapr workflow.

    Must be yielded by the calling workflow:
        result = yield call_agent(ctx, "WeatherAgent", input={...}, app_id="weather-agent")

    If app_id or framework are not provided, they are looked up from the agent registry
    (requires registry to be passed, e.g. an agent's DaprInfra instance).

    Args:
        ctx: The DaprWorkflowContext passed into the calling workflow function.
        name: Name of the target agent (e.g. "WeatherAgent", "weather agent").
        input: Payload to send to the agent.
        app_id: Dapr app ID of the app running the target agent. Looked up from
            registry if not provided.
        instance_id: Optional explicit child workflow instance ID.
        framework: Optional framework name override. Looked up from registry if not
            provided (defaults to "agents" if registry also has no entry).
        registry: Optional registry object (e.g. DaprInfra) used to look up app_id
            and framework when they are not supplied explicitly.

    Returns:
        A Task that must be yielded by the calling workflow.
    """
    app_id, framework = _resolve_registry(name, app_id, framework, registry)

    if app_id is None:
        raise ValueError(
            f"app_id is required for call_agent('{name}') — "
            "provide it directly or pass a registry to look it up."
        )

    workflow_name = agent_workflow_id(name, framework=framework)
    kwargs: Dict[str, Any] = {
        "workflow": workflow_name,
        "input": input,
        "app_id": app_id,
    }
    if instance_id is not None:
        kwargs["instance_id"] = instance_id
    return ctx.call_child_workflow(**kwargs)


def trigger_agent(
    name: str,
    input: Any,
    *,
    app_id: Optional[str] = None,
    timeout_in_seconds: int = 120,
    framework: Optional[str] = None,
    registry: Optional[Any] = None,
) -> Any:
    """
    Trigger a DurableAgent and block until it completes. No workflow boilerplate needed.

    Handles WorkflowRuntime lifecycle, workflow registration, scheduling, and waiting
    internally. The caller only needs the agent name and the Dapr app ID.

        result = trigger_agent("WeatherAgent", input={...}, app_id="weather-agent")

    If app_id or framework are not provided, they are looked up from the agent registry
    (requires registry to be passed, e.g. an agent's DaprInfra instance).

    Args:
        name: Name of the target agent (e.g. "WeatherAgent").
        input: Payload to send to the agent.
        app_id: Dapr app ID of the app running the target agent. Looked up from
            registry if not provided.
        timeout_in_seconds: How long to wait for completion (default 120s).
        framework: Optional framework name override. Looked up from registry if not
            provided (defaults to "agents" if registry also has no entry).
        registry: Optional registry object (e.g. DaprInfra) used to look up app_id
            and framework when they are not supplied explicitly.

    Returns:
        The serialized output from the agent workflow.
    """
    app_id, framework = _resolve_registry(name, app_id, framework, registry)

    if app_id is None:
        raise ValueError(
            f"app_id is required for trigger_agent('{name}') — "
            "provide it directly or pass a registry to look it up."
        )

    workflow_name = agent_workflow_id(name, framework=framework)

    # Name follows the dapr.agents.<name>.<suffix> convention:
    #   dapr.agents.<name>.workflow      — primary agent workflow
    #   dapr.agents.<name>.broadcast     — fanout broadcast handler
    #   dapr.agents.<name>.orchestration — multi-agent coordination
    #   dapr.agents.<name>.trigger       — caller-side trigger wrapper (this one)
    sanitized = sanitize_openai_tool_name(name)
    framework_prefix = framework or "agents"
    trigger_name = f"dapr.{framework_prefix}.{sanitized}.trigger"

    trigger = _TriggerWorkflow(trigger_name, workflow_name, input, app_id)

    wfr = wf.WorkflowRuntime()
    wfr.register_workflow(trigger)
    wfr.start()

    try:
        with wf.DaprWorkflowClient() as client:
            instance_id = client.schedule_new_workflow(workflow=trigger)
            try:
                state = client.wait_for_workflow_completion(
                    instance_id=instance_id,
                    timeout_in_seconds=timeout_in_seconds,
                    fetch_payloads=True,
                )
            except TimeoutError:
                logger.warning(
                    "trigger_agent('%s') timed out after %ds (instance_id=%s). "
                    "The workflow may still be running.",
                    name,
                    timeout_in_seconds,
                    instance_id,
                )
                return None
        return state.serialized_output if state is not None else None
    finally:
        wfr.shutdown()


def is_pydantic_model(obj: Any) -> bool:
    """Check if the given type is a subclass of Pydantic's BaseModel."""
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def is_supported_model(cls: Any) -> bool:
    """Checks if a class is a supported message schema (Pydantic, dataclass, or dict)."""
    return cls is dict or is_dataclass(cls) or is_pydantic_model(cls)


def is_valid_routable_model(cls: Any) -> bool:
    return is_dataclass(cls) or is_pydantic_model(cls)


def get_decorated_methods(instance: Any, attribute_name: str) -> Dict[str, Callable]:
    """
    Find all **public** bound methods on `instance` that carry a given decorator attribute.

    This will:
      1. Inspect the class for functions or methods.
      2. Bind them to the instance (so `self` is applied).
      3. Filter in only those where `hasattr(method, attribute_name) is True`.

    Args:
        instance:  Any object whose methods you want to inspect.
        attribute_name:
            The name of the attribute set by your decorator
            (e.g. "_is_task" or "_is_workflow").

    Returns:
        A dict mapping `method_name` → `bound method`.

    Example:
        >>> class A:
        ...     @task
        ...     def foo(self): ...
        ...
        >>> get_decorated_methods(A(), "_is_task")
        {"foo": <bound method A.foo of <A object ...>>}
    """
    discovered: Dict[str, Callable] = {}

    cls = type(instance)
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        # skip private/protected
        if name.startswith("_"):
            continue

        # bind to instance so that signature(self, ...) works
        try:
            bound = getattr(instance, name)
        except Exception as e:
            logger.warning(f"Could not bind method '{name}': {e}")
            continue

        # pick up only those with our decorator flag
        if hasattr(bound, attribute_name):
            discovered[name] = bound
            logger.debug(f"Discovered decorated method: {name}")

    return discovered


def to_payload(model: Any) -> Dict[str, Any]:
    """
    Convert supported message instances to a JSON-serializable dict.

    Supports:
    - dict: returned as a shallow copy
    - dataclass: converted via asdict()
    - Pydantic v2: model_dump(exclude_none=True)
    - Pydantic v1: dict(exclude_none=True)

    Falls back to dict() if possible, otherwise raises.
    """
    if isinstance(model, dict):
        return dict(model)

    if is_dataclass(model):
        return asdict(model)

    # Pydantic v2
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)

    # Pydantic v1
    if hasattr(model, "dict"):
        return model.dict(exclude_none=True)

    try:
        return dict(model)
    except Exception as e:
        raise TypeError(
            f"Unsupported message payload type for serialization: {type(model)!r}"
        ) from e


def coerce_to_model(model_type: Type[Any], value: Any) -> Any:
    """
    Best-effort coercion of `value` into `model_type` where possible.

    Args:
        model_type: Expected model class/type.
        value: Incoming value.

    Returns:
        Any: Value coerced to `model_type` when feasible; original value otherwise.
    """
    if model_type is Any or model_type is dict:
        return value

    try:
        if isinstance(value, model_type):
            return value
    except TypeError:
        # model_type may be typing constructs (e.g., typing.Dict) that break isinstance
        pass

    if hasattr(model_type, "model_validate"):
        return model_type.model_validate(value)

    if hasattr(model_type, "parse_obj"):
        return model_type.parse_obj(value)

    if is_dataclass(model_type):
        try:
            if isinstance(value, model_type):
                return value
        except TypeError:
            pass
        if isinstance(value, dict):
            return model_type(**value)

    return value
