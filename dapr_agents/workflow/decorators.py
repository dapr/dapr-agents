import functools
import inspect
from typing import Any, Callable, Optional
import logging

from pydantic import BaseModel, ValidationError

from dapr.ext.workflow import DaprWorkflowContext


def route(path: str, method: str = "GET", **kwargs):
    """
    Decorator to mark an instance method as a FastAPI route.

    Args:
        path (str): The URL path to bind this route to.
        method (str): The HTTP method to use (e.g., 'GET', 'POST'). Defaults to 'GET'.
        **kwargs: Additional arguments passed to FastAPI's `add_api_route`.

    Example:
        @route("/status", method="GET", summary="Show status", tags=["monitoring"])
        def health(self):
            return {"ok": True}
    """

    def decorator(func):
        func._is_fastapi_route = True
        func._route_path = path
        func._route_method = method.upper()
        func._route_kwargs = kwargs
        return func

    return decorator


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


def is_pydantic_model(obj: Any) -> bool:
    """Check if the given type is a subclass of Pydantic's BaseModel."""
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def workflow(
    func: Optional[Callable] = None, *, name: Optional[str] = None
) -> Callable:
    """
    Decorator to register a function as a Dapr workflow with optional Pydantic validation.

    - Ensures the correct placement of `ctx: DaprWorkflowContext`
    - If an input parameter is a Pydantic model, validates and serializes it
    - Works seamlessly with standalone functions, instance methods, and class methods.

    Args:
        func (Callable, optional): Function to be decorated as a workflow.
        name (Optional[str]): The name to register the workflow with.

    Returns:
        Callable: The decorated function with input validation.
    """

    def decorator(f: Callable) -> Callable:
        if not callable(f):
            raise ValueError(f"@workflow must be applied to a function, got {type(f)}.")

        # Assign workflow metadata attributes
        f._is_workflow = True
        f._workflow_name = name or f.__name__

        sig = inspect.signature(f)
        params = list(sig.parameters.values())

        # Determine if function is an instance method, class method, or standalone function
        is_instance_method = False
        is_class_method = False

        if isinstance(f, classmethod):  # If already wrapped as a class method
            is_class_method = True
            f = f.__func__  # Extract the underlying function
            sig = inspect.signature(f)  # Recompute signature after unwrapping
            params = list(sig.parameters.values())  # Update parameter list

        elif params and params[0].name == "self":  # Instance method
            is_instance_method = True

        elif (
            params and params[0].name == "cls"
        ):  # Class method without @classmethod decorator
            is_class_method = True

        # Compute the expected index for `ctx`
        ctx_index = 1 if is_instance_method or is_class_method else 0

        # Ensure `ctx` is correctly positioned
        if (
            len(params) <= ctx_index
            or params[ctx_index].annotation is not DaprWorkflowContext
        ):
            raise TypeError(
                f"Workflow '{f.__name__}' must have 'ctx: DaprWorkflowContext' as the {'second' if ctx_index == 1 else 'first'} parameter."
            )

        # Identify the input parameter (third argument for methods, second otherwise)
        input_param_index = ctx_index + 1
        input_param = (
            params[input_param_index] if len(params) > input_param_index else None
        )
        input_type = input_param.annotation if input_param else None
        is_pydantic = is_pydantic_model(input_type)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            """Wrapper for handling input validation and execution."""

            logging.getLogger(__name__).info(f"Starting workflow '{f._workflow_name}'")

            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            # Extract `ctx` (first parameter for functions, second for methods)
            ctx = bound_args.arguments.get(params[ctx_index].name, None)
            if not isinstance(ctx, DaprWorkflowContext):
                raise TypeError(
                    f"Expected '{params[ctx_index].name}' to be a DaprWorkflowContext instance."
                )

            # Extract `input` (third parameter for methods, second otherwise)
            input_value = (
                bound_args.arguments.get(input_param.name, None)
                if input_param
                else None
            )

            # Ensure metadata is extracted without modifying input_value unnecessarily
            metadata = None
            if isinstance(input_value, dict) and "_message_metadata" in input_value:
                metadata = input_value.pop("_message_metadata")

            # Validate input if it's a Pydantic model
            if is_pydantic and input_value:
                try:
                    validated_input: BaseModel = input_type(
                        **input_value
                    )  # Validate with Pydantic
                except ValidationError as e:
                    raise ValueError(
                        f"Invalid input for workflow '{f._workflow_name}': {e.errors()}"
                    ) from e

                # Convert back to dict and reattach metadata
                validated_dict = validated_input.model_dump()
                if metadata is not None:
                    validated_dict[
                        "_message_metadata"
                    ] = metadata  # Ensure metadata is not lost

                # Overwrite the function argument with the modified dictionary
                bound_args.arguments[input_param.name] = validated_dict

            return f(*bound_args.args, **bound_args.kwargs)

        wrapper._is_workflow = True
        wrapper._workflow_name = f._workflow_name
        return wrapper

    return (
        decorator(func) if func else decorator
    )  # Supports both `@workflow` and `@workflow(name="custom")`
