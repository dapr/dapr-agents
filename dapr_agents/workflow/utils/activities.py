from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Tuple

from dapr.ext.workflow import WorkflowActivityContext  # type: ignore
from pydantic import BaseModel

from dapr_agents.llm.utils import StructureHandler
from dapr_agents.types import BaseMessage, LLMChatResponse, UserMessage

# Public alias for normalized/filtered inputs passed to activities.
NormalizedInput = Dict[str, Any]


def normalize_input(
    signature: inspect.Signature,
    raw_input: Any,
    *,
    strict: bool = True,
) -> NormalizedInput:
    """Normalize a workflow payload to kwargs that match an activity signature.

    Prefer structured forms (dataclass, ``SimpleNamespace``, Pydantic, mapping).
    Fall back to assigning the first positional parameter if the payload is a
    scalar or otherwise not mapping-like.

    When ``strict`` is True, raise ``TypeError`` if the payload includes keys
    not present in the activity signature. This detects mismatches early.

    Args:
        signature: Activity signature (minus context parameter).
        raw_input: Original payload provided by the workflow.
        strict: Whether to raise on unexpected keys (default: True).

    Returns:
        Mapping aligned to parameter names for the activity.

    Raises:
        TypeError: If ``strict`` is True and unknown keys are detected.
    """
    if raw_input is None:
        return {}

    # Coerce common structured inputs to dicts.
    if is_dataclass(raw_input):
        data = asdict(raw_input)
    elif isinstance(raw_input, SimpleNamespace):
        data = vars(raw_input)
    elif isinstance(raw_input, BaseModel):
        data = raw_input.model_dump()
    elif isinstance(raw_input, Mapping):
        data = dict(raw_input)
    else:
        # Scalar / positional fallback: bind to the first parameter if present.
        if not signature.parameters:
            return {}
        first_param = next(iter(signature.parameters))
        return {first_param: raw_input}

    param_names = set(signature.parameters.keys())
    unknown = set(data.keys()) - param_names
    if strict and unknown:
        raise TypeError(f"Unexpected input keys for activity: {sorted(unknown)}")

    return {k: v for k, v in data.items() if k in param_names}


def format_prompt(
    signature: inspect.Signature, template: str, data: Dict[str, Any]
) -> str:
    """Render a prompt template using defaults from the signature and provided data.

    The provided ``data`` should already be normalized via :func:`normalize_input`,
    which filters unknown keys. This function binds partial arguments against the
    signature to apply parameter defaults before formatting the template.

    Args:
        signature: Activity signature used for default resolution.
        template: Prompt template (may refer to argument names).
        data: Normalized payload for formatting.

    Returns:
        Rendered prompt string. Returns an empty string if ``template`` is falsy.

    Raises:
        ValueError: If binding or template formatting fails due to missing keys.
    """
    if not template:
        return ""

    # Bind partially to compute defaults for any omitted optional parameters.
    try:
        bound = signature.bind_partial(**data)
    except TypeError as exc:
        raise ValueError(
            f"Failed to bind prompt arguments to signature: {exc}"
        ) from exc

    bound.apply_defaults()

    # Format with the bound arguments. Report the first missing key clearly.
    try:
        return template.format(**bound.arguments)
    except KeyError as exc:
        missing_key = exc.args[0]
        raise ValueError(
            f"Prompt template expects missing key: '{missing_key}'"
        ) from exc


def format_agent_input(payload: Any, data: Dict[str, Any]) -> str:
    """Create a simple natural-language string for agent input from payload/data.

    This is a best-effort formatter for agent-facing inputs when no explicit
    prompt template is used.

    Args:
        payload: Original input object.
        data: Normalized mapping (post-filtering) of payload fields.

    Returns:
        A concise, readable string for agent consumption.
    """
    if payload is None:
        return ""

    # Fast-path scalars.
    if isinstance(payload, (str, int, float, bool)):
        return str(payload)

    # If there's a single normalized field, return its string form.
    if data and len(data) == 1:
        value = next(iter(data.values()))
        return "" if value is None else str(value)

    # Otherwise, render a multi-line key: value list.
    if data:
        parts = [f"{key}: {value}" for key, value in data.items() if value is not None]
        return "\n".join(parts)

    # Last-resort stringification of the raw payload.
    return str(payload)


def build_llm_params(
    signature: inspect.Signature,
    prompt: str,
    structured_mode: str,
) -> Dict[str, Any]:
    """Build keyword arguments for ``ChatClientBase.generate``.

    If the activity return type is annotated, add a ``response_format`` so
    structured output can be requested from providers that support it.

    Args:
        signature: Activity signature; its return annotation drives structure.
        prompt: Final prompt string (already formatted).
        structured_mode: Provider-specific structured mode (e.g., ``"json"``).

    Returns:
        Dict of parameters for the chat client, including messages and, when
        applicable, structured output hints.
    """
    messages: List[BaseMessage] = []
    if prompt:
        messages.append(UserMessage(prompt))

    params: Dict[str, Any] = {"messages": messages}

    # If a concrete return annotation is present, try to resolve a model.
    if signature.return_annotation is not inspect.Signature.empty:
        model_cls = StructureHandler.resolve_response_model(signature.return_annotation)
        if model_cls:
            params["response_format"] = model_cls
            params["structured_mode"] = structured_mode

    return params


def convert_result(result: Any) -> Any:
    """Convert LLM/agent results to plain Python containers.

    Normalization rules:
    - ``LLMChatResponse`` → extract underlying message content.
    - Pydantic model → ``dict`` via ``model_dump()``.
    - List of Pydantic models → ``list[dict]``.
    - Otherwise → pass through unchanged.

    Args:
        result: Raw result returned by the chat client or agent.

    Returns:
        A Python container suitable for validation/serialization.
    """
    if isinstance(result, LLMChatResponse):
        message = result.get_message()
        return getattr(message, "content", None)

    if isinstance(result, BaseMessage):
        return result.model_dump()

    if isinstance(result, BaseModel):
        return result.model_dump()

    if isinstance(result, list) and all(isinstance(item, BaseModel) for item in result):
        return [item.model_dump() for item in result]

    return result


async def validate_result(result: Any, signature: inspect.Signature) -> Any:
    """Validate/transform a result against an activity's return annotation.

    If the result is awaitable, it is awaited first. If no return annotation is
    present, the value is returned as-is. Otherwise, the value is validated and
    coerced according to the annotation via :class:`StructureHandler`.

    Args:
        result: Raw (possibly awaitable) result.
        signature: Activity signature with return annotation.

    Returns:
        Validated/converted value compatible with the annotation.
    """
    if inspect.isawaitable(result):
        result = await result

    if (
        not signature.return_annotation
        or signature.return_annotation is inspect.Signature.empty
    ):
        return result

    return StructureHandler.validate_against_signature(
        result, signature.return_annotation
    )


def strip_context_parameter(signature: inspect.Signature) -> inspect.Signature:
    """Remove the leading workflow context parameter, when present.

    Activities often accept a Dapr workflow context (``ctx``) as the first
    argument; removing it simplifies payload binding/formatting, which should
    only consider user-provided inputs.

    Args:
        signature: Original callable signature.

    Returns:
        A new signature without the context parameter (if detected).
    """
    params = list(signature.parameters.values())
    if not params:
        return signature

    first = params[0]
    if (
        first.annotation is WorkflowActivityContext  # explicit type annotation
        or first.name in {"ctx", "context", "workflow_ctx"}  # common names
    ):
        params = params[1:]

    return inspect.Signature(
        parameters=params, return_annotation=signature.return_annotation
    )


def extract_ctx_and_payload(
    args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Any, Any]:
    """Extract ``(ctx, payload)`` from wrapper ``args``/``kwargs``.

    Accepts either positional ``(ctx, payload)`` or keyword forms
    (``ctx=...``, and one of ``input=``/``payload=``). Any remaining keyword
    arguments are interpreted as the payload mapping.

    Args:
        args: Positional args from the activity invocation.
        kwargs: Keyword args from the activity invocation (may be consumed).

    Returns:
        ``(ctx, payload)`` pair suitable for execution.

    Raises:
        ValueError: If the workflow context is missing.
    """
    ctx = None
    payload: Any = None

    # Positional: (ctx[, payload])
    if args:
        ctx = args[0]
        if len(args) > 1:
            payload = args[1]

    # Keyword fallback for ctx
    if ctx is None:
        ctx = kwargs.pop("ctx", kwargs.pop("context", None))

    # Keyword payload (prefer 'payload', then 'input')
    if payload is None:
        payload = kwargs.pop("payload", kwargs.pop("input", None))

    # If there are still kwargs left, treat them as a payload mapping.
    if payload is None and kwargs:
        payload = kwargs  # remaining named fields

    if ctx is None:
        raise ValueError("Workflow context is required for activity execution.")

    return ctx, payload
