import functools
from typing import Any, Callable, Optional
import logging


def task(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    agent: Optional[Any] = None,
    llm: Optional[Any] = None,
    include_chat_history: bool = False,
    **task_kwargs,
) -> Callable:
    """
    Decorator to register a function as a Dapr workflow task.

    This allows configuring a task with an LLM, agent, chat history, and other options.
    All additional keyword arguments are stored and forwarded to the WorkflowTask constructor.

    Args:
        func (Optional[Callable]): The function to wrap. Can also be used as `@task(...)`.
        name (Optional[str]): Optional custom task name. Defaults to the function name.
        description (Optional[str]): Optional prompt template for LLM-based execution.
        agent (Optional[Any]): Optional agent to handle the task instead of an LLM or function.
        llm (Optional[Any]): Optional LLM client used to execute the task.
        include_chat_history (bool): Whether to include prior messages in LLM calls.
        **task_kwargs: Additional keyword arguments to forward to `WorkflowTask`.

    Returns:
        Callable: The decorated function with attached task metadata.
    """

    if isinstance(func, str):
        # Allow syntax: @task("some description")
        description = func
        func = None

    def decorator(f: Callable) -> Callable:
        if not callable(f):
            raise ValueError(f"@task must be applied to a function, got {type(f)}.")

        # Attach task metadata
        f._is_task = True
        f._task_name = name or f.__name__
        f._task_description = description
        f._task_agent = agent
        f._task_llm = llm
        f._task_include_chat_history = include_chat_history
        f._explicit_llm = llm is not None or bool(description)
        f._task_kwargs = task_kwargs

        # wrap it so we can log, validate, etc., without losing signature/docs
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            logging.getLogger(__name__).debug(f"Calling task '{f._task_name}'")
            return f(*args, **kwargs)

        # copy our metadata onto the wrapper so discovery still sees it
        for attr in (
            "_is_task",
            "_task_name",
            "_task_description",
            "_task_agent",
            "_task_llm",
            "_task_include_chat_history",
            "_explicit_llm",
            "_task_kwargs",
        ):
            setattr(wrapper, attr, getattr(f, attr))

        return wrapper

    return (
        decorator(func) if func else decorator
    )  # Supports both @task and @task(name="custom")


def workflow(
    func: Optional[Callable] = None, *, name: Optional[str] = None
) -> Callable:
    """
    Decorator to register a function as a Dapr workflow.

    - Attaches workflow metadata for discovery and registration.
    - Works seamlessly with standalone functions, instance methods, and class methods.

    Args:
        func (Callable, optional): Function to be decorated as a workflow.
        name (Optional[str]): The name to register the workflow with.

    Returns:
        Callable: The decorated function with workflow metadata for Dapr compatibility.
    """

    def decorator(f: Callable) -> Callable:
        """
        Minimal workflow decorator for Dapr workflows.
        - Sets workflow metadata for discovery and registration.
        """
        if not callable(f):
            raise ValueError(f"@workflow must be applied to a function, got {type(f)}.")

        f._is_workflow = True
        f._workflow_name = name or f.__name__

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper._is_workflow = True
        wrapper._workflow_name = f._workflow_name
        return wrapper

    return (
        decorator(func) if func else decorator
    )  # Supports both `@workflow` and `@workflow(name="custom")`
