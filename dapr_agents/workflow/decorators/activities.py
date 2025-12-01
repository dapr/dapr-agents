from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable, Literal, Optional, TypeVar

from dapr.ext.workflow import WorkflowActivityContext  # type: ignore

from dapr_agents.agents.base import AgentBase
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.workflow.utils.activities import (
    build_llm_params,
    convert_result,
    extract_ctx_and_payload,
    format_agent_input,
    format_prompt,
    normalize_input,
    strip_context_parameter,
    validate_result,
)

logger = logging.getLogger(__name__)

R = TypeVar("R")


def workflow_entry(func: Callable[..., R]) -> Callable[..., R]:
    """
    Mark a method/function as the workflow entrypoint for an Agent.

    This decorator does not wrap the function; it simply annotates the callable
    with `_is_workflow_entry = True` so AgentRunner can discover it on the agent
    instance via reflection.

    Usage:
        class MyAgent:
            @workflow_entry
            def my_workflow(self, ctx: DaprWorkflowContext, wf_input: dict) -> str:
                ...

    Returns:
        The same callable (unmodified), with an identifying attribute.
    """
    setattr(func, "_is_workflow_entry", True)  # type: ignore[attr-defined]
    return func


def llm_activity(
    *,
    prompt: str,
    llm: ChatClientBase,
    structured_mode: Literal["json", "function_call"] = "json",
    **task_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Delegate an activity's implementation to an LLM.

    The decorated function's body is not executed directly. Instead:
    1) Build a prompt from the activity's signature + `prompt`
    2) Call the provided LLM client
    3) Validate the result against the activity's return annotation

    Args:
        prompt: Prompt template (e.g., "Summarize {text} in 3 bullets.")
        llm: Chat client capable of `generate(**params)`.
        structured_mode: Provider structured output mode ("json" or "function_call").
        **task_kwargs: Reserved for future routing/provider knobs.

    Returns:
        A wrapper suitable to register as a Dapr activity.

    Raises:
        ValueError: If `prompt` is empty or `llm` is missing.
    """
    if not prompt:
        raise ValueError("@llm_activity requires a prompt template.")
    if llm is None:
        raise ValueError("@llm_activity requires an explicit `llm` client instance.")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(func):
            raise ValueError("@llm_activity must decorate a callable activity.")

        original_sig = inspect.signature(func)
        activity_sig = strip_context_parameter(original_sig)
        effective_structured_mode = task_kwargs.get("structured_mode", structured_mode)

        async def _execute(ctx: WorkflowActivityContext, payload: Any = None) -> Any:
            """Run the LLM pipeline inside the worker."""
            normalized = (
                normalize_input(activity_sig, payload) if payload is not None else {}
            )

            formatted_prompt = format_prompt(activity_sig, prompt, normalized)
            params = build_llm_params(
                activity_sig, formatted_prompt, effective_structured_mode
            )

            raw = llm.generate(**params)
            if inspect.isawaitable(raw):
                raw = await raw

            converted = convert_result(raw)
            validated = await validate_result(converted, activity_sig)
            return validated

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Sync activity wrapper: execute async pipeline to completion."""
            ctx, payload = extract_ctx_and_payload(args, dict(kwargs))
            result = _execute(ctx, payload)  # coroutine

            # If we're in a thread with an active loop, run thread-safely
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(result, loop)
                return fut.result()

            # Otherwise create and run a fresh loop
            return asyncio.run(result)

        # Useful metadata for debugging/inspection
        wrapper._is_llm_activity = True  # noqa: SLF001
        wrapper._llm_activity_config = {  # noqa: SLF001
            "prompt": prompt,
            "structured_mode": effective_structured_mode,
            "task_kwargs": task_kwargs,
        }
        wrapper._original_activity = func  # noqa: SLF001
        return wrapper

    return decorator


def agent_activity(
    *,
    agent: AgentBase,
    prompt: Optional[str] = None,
    **task_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Route an activity through an `AgentBase`.

    The agent receives either a formatted `prompt` or a natural-language
    rendering of the payload. The result is validated against the activity's return
    annotation.

    Args:
        agent: Agent to run the activity through.
        prompt: Optional prompt template for the agent.
        **task_kwargs: Reserved for future routing/provider knobs.

    Returns:
        A wrapper suitable to register as a Dapr activity.

    Raises:
        ValueError: If `agent` is missing.
    """
    if agent is None:
        raise ValueError("@agent_activity requires an AgentBase instance.")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(func):
            raise ValueError("@agent_activity must decorate a callable activity.")

        original_sig = inspect.signature(func)
        activity_sig = strip_context_parameter(original_sig)
        prompt_template = prompt or ""

        async def _execute(ctx: WorkflowActivityContext, payload: Any = None) -> Any:
            normalized = (
                normalize_input(activity_sig, payload) if payload is not None else {}
            )

            if prompt_template:
                formatted_prompt = format_prompt(
                    activity_sig, prompt_template, normalized
                )
            else:
                formatted_prompt = format_agent_input(payload, normalized)

            raw = await agent.run(formatted_prompt)
            converted = convert_result(raw)
            validated = await validate_result(converted, activity_sig)
            return validated

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Sync activity wrapper: execute async pipeline to completion."""
            ctx, payload = extract_ctx_and_payload(args, dict(kwargs))
            result = _execute(ctx, payload)  # coroutine

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(result, loop)
                return fut.result()

            return asyncio.run(result)

        wrapper._is_agent_activity = True  # noqa: SLF001
        wrapper._agent_activity_config = {  # noqa: SLF001
            "prompt": prompt,
            "task_kwargs": task_kwargs,
        }
        wrapper._original_activity = func  # noqa: SLF001
        return wrapper

    return decorator
