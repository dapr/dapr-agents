import asyncio
import functools
import inspect
import json
import logging
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from dapr.ext.workflow import (
    DaprWorkflowClient,
    WorkflowActivityContext,
    WorkflowRuntime,
)
from dapr.ext.workflow.workflow_state import WorkflowState
from durabletask import task as dtask
from pydantic import BaseModel, ConfigDict, Field
from typing import ClassVar

from dapr_agents.agents.base import ChatClientBase
from dapr_agents.types.workflow import DaprWorkflowStatus
from dapr_agents.workflow.task import WorkflowTask
from dapr_agents.workflow.utils.core import get_decorated_methods, is_pydantic_model

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WorkflowApp(BaseModel):
    """
    A Pydantic-based class to encapsulate a Dapr Workflow runtime and manage workflows and tasks.
    """

    # Class-level constant for instrumentation callback
    # This is used to register the workflow instrumentation callback with the instrumentor
    # This is needed because the instrumentor is not available immediately when the WorkflowApp is initialized
    INSTRUMENTATION_CALLBACK_ATTR: ClassVar[str] = "_workflow_instrumentation_callback"

    llm: Optional[ChatClientBase] = Field(
        default=None,
        description="The default LLM client for tasks that explicitly require an LLM but don't specify one (optional).",
    )
    # TODO: I think this should be within the wf client or wf runtime...?
    timeout: int = Field(
        default=300,
        description="Default timeout duration in seconds for workflow tasks.",
    )

    # Initialized in model_post_init
    wf_runtime: Optional[WorkflowRuntime] = Field(
        default=None, init=False, description="Workflow runtime instance."
    )
    wf_runtime_is_running: Optional[bool] = Field(
        default=None, init=False, description="Is the Workflow runtime running?"
    )
    wf_client: Optional[DaprWorkflowClient] = Field(
        default=None, init=False, description="Workflow client instance."
    )
    tasks: Dict[str, Callable] = Field(
        default_factory=dict, init=False, description="Dictionary of registered tasks."
    )
    workflows: Dict[str, Callable] = Field(
        default_factory=dict,
        init=False,
        description="Dictionary of registered workflows.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize the Dapr workflow runtime and register tasks & workflows.
        """
        # Initialize clients and runtime
        self.wf_runtime = WorkflowRuntime()
        self.wf_runtime_is_running = False
        self.wf_client = DaprWorkflowClient()
        logger.info("WorkflowApp initialized; discovering tasks and workflows.")

        # Discover and register tasks and workflows
        discovered_tasks = self._discover_tasks()
        self._register_tasks(discovered_tasks)
        discovered_wfs = self._discover_workflows()
        self._register_workflows(discovered_wfs)

        super().model_post_init(__context)

    def _choose_llm_for(self, method: Callable) -> Optional[ChatClientBase]:
        """
        Encapsulate LLM selection logic.
          1. Use per-task override if provided on decorator.
          2. Else if marked as explicitly requiring an LLM, fall back to default app LLM (if available).
          3. Otherwise, returns None.
        """
        per_task = getattr(method, "_task_llm", None)
        if per_task:
            return per_task
        if getattr(method, "_explicit_llm", False):
            if self.llm is None:
                logger.warning(
                    f"Task '{getattr(method, '_task_name', getattr(method, '__name__', str(method)))}' requires an LLM "
                    "but no default LLM is configured in WorkflowApp and no explicit LLM was provided."
                )
            return self.llm
        return None

    def _discover_tasks(self) -> Dict[str, Callable]:
        """Gather all @task-decorated functions and methods."""
        module = sys.modules["__main__"]
        tasks: Dict[str, Callable] = {}
        # Free functions in __main__
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if getattr(fn, "_is_task", False) and fn.__module__ == module.__name__:
                tasks[getattr(fn, "_task_name", name)] = fn
        # Bound methods (if any) discovered via helper
        for name, method in get_decorated_methods(self, "_is_task").items():
            tasks[getattr(method, "_task_name", name)] = method
        logger.debug(f"Discovered tasks: {list(tasks)}")
        return tasks

    def register_task(self, task_func: Callable, name: Optional[str] = None) -> None:
        """
        Manually register a @task-decorated function, similar to native Dapr pattern.

        This allows explicit registration of tasks from other modules or packages.
        The task will be registered immediately with the Dapr runtime.

        Args:
            task_func: The @task-decorated function to register
            name: Optional custom name (defaults to function name or _task_name)

        Example:
            from tasks.my_tasks import generate_queries
            wfapp.register_task(generate_queries)
            wfapp.register_task(generate_queries, name="custom_name")
        """
        # Validate input
        if not callable(task_func):
            raise ValueError(
                f"task_func must be callable, got {type(task_func)}: {task_func}"
            )

        if not getattr(task_func, "_is_task", False):
            raise ValueError(
                f"Function {getattr(task_func, '__name__', str(task_func))} is not decorated with @task"
            )

        task_name = name or getattr(
            task_func, "_task_name", getattr(task_func, "__name__", "unknown_task")
        )

        # Check if already registered
        if task_name in self.tasks:
            logger.warning(f"Task '{task_name}' is already registered, skipping")
            return

        # Register immediately with Dapr runtime using existing registration logic
        llm = self._choose_llm_for(task_func)
        logger.debug(
            f"Manually registering task '{task_name}' with llm={getattr(llm, '__class__', None)}"
        )

        kwargs = getattr(task_func, "_task_kwargs", {})
        task_instance = WorkflowTask(
            func=task_func,
            description=getattr(task_func, "_task_description", None),
            agent=getattr(task_func, "_task_agent", None),
            llm=llm,
            include_chat_history=getattr(
                task_func, "_task_include_chat_history", False
            ),
            workflow_app=self,
            **kwargs,
        )

        # Wrap for Dapr invocation
        wrapped = self._make_task_wrapper(task_name, task_func, task_instance)
        self.wf_runtime.register_activity(wrapped)
        self.tasks[task_name] = wrapped

    def register_tasks_from_module(
        self, module_name_or_object: Union[str, Any]
    ) -> None:
        """
        Register all @task-decorated functions from a specific module.

        Args:
            module_name_or_object: Module name string (e.g., "tasks.queries") or imported module object

        Example:
            # Using string name
            wfapp.register_tasks_from_module("tasks.queries")

            # Using imported module
            import tasks.queries
            wfapp.register_tasks_from_module(tasks.queries)

            # Using from import
            from tasks import queries
            wfapp.register_tasks_from_module(queries)
        """
        try:
            # Handle both string names and module objects
            if isinstance(module_name_or_object, str):
                import importlib

                module = importlib.import_module(module_name_or_object)
                module_name = module_name_or_object
            else:
                # Assume it's a module object
                module = module_name_or_object
                module_name = getattr(module, "__name__", str(module))

            registered_count = 0
            for name, fn in inspect.getmembers(module, inspect.isfunction):
                if getattr(fn, "_is_task", False):
                    task_name = getattr(fn, "_task_name", name)
                    if task_name not in self.tasks:  # Skip if already registered
                        self.register_task(fn)
                        registered_count += 1

            logger.info(
                f"Registered {registered_count} tasks from module '{module_name}'"
            )

        except ImportError as e:
            raise ImportError(f"Could not import module '{module_name_or_object}': {e}")
        except Exception as e:
            raise RuntimeError(
                f"Error registering tasks from module '{module_name_or_object}': {e}"
            )

    def register_tasks_from_package(self, package_name: str) -> None:
        """
        Register all @task-decorated functions from all modules in a package.

        Args:
            package_name: Name of package to scan (e.g., "tasks")

        Example:
            wfapp.register_tasks_from_package("tasks")  # Scans tasks/*.py
        """
        try:
            import importlib
            import pkgutil

            package = importlib.import_module(package_name)

            # Collect all tasks first, then register using the original _register_tasks method
            discovered_tasks: Dict[str, Callable] = {}

            total_tasks = 0
            for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
                if not ispkg:  # Only scan modules, not sub-packages
                    full_module_name = f"{package_name}.{modname}"
                    try:
                        module = importlib.import_module(full_module_name)

                        for name, fn in inspect.getmembers(module, inspect.isfunction):
                            if getattr(fn, "_is_task", False):
                                task_name = getattr(fn, "_task_name", name)
                                if (
                                    task_name not in self.tasks
                                ):  # Skip if already registered
                                    discovered_tasks[task_name] = fn
                                    total_tasks += 1

                    except Exception as e:
                        logger.warning(f"Failed to scan module {full_module_name}: {e}")

            # Now register all discovered tasks using the original _register_tasks method
            if discovered_tasks:
                self._register_tasks(discovered_tasks)

            logger.info(f"Registered {total_tasks} tasks from package '{package_name}'")
        except ImportError as e:
            raise ImportError(f"Could not import package '{package_name}': {e}")
        except Exception as e:
            raise RuntimeError(
                f"Error registering tasks from package '{package_name}': {e}"
            )

    def _register_tasks(self, tasks: Dict[str, Callable]) -> None:
        """Register each discovered task with the Dapr runtime using direct registration."""
        for task_name, method in tasks.items():
            llm = self._choose_llm_for(method)
            logger.debug(
                f"Registering task '{task_name}' with llm={getattr(llm, '__class__', None)}"
            )
            kwargs = getattr(method, "_task_kwargs", {})
            task_instance = WorkflowTask(
                func=method,
                description=getattr(method, "_task_description", None),
                agent=getattr(method, "_task_agent", None),
                llm=llm,
                include_chat_history=getattr(
                    method, "_task_include_chat_history", False
                ),
                workflow_app=self,
                **kwargs,
            )
            # Wrap for Dapr invocation
            wrapped = self._make_task_wrapper(task_name, method, task_instance)

            # Use direct registration like official Dapr examples
            self.wf_runtime.register_activity(wrapped)
            self.tasks[task_name] = wrapped

    def _make_task_wrapper(
        self, task_name: str, method: Callable, task_instance: WorkflowTask
    ) -> Callable:
        """Produce the function that Dapr will invoke for each activity."""

        def run_sync(coro):
            # Try to get the running event loop and run until complete
            try:
                loop = asyncio.get_running_loop()
                return loop.run_until_complete(coro)
            except RuntimeError:
                # If no running loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)

        @functools.wraps(method)
        def wrapper(ctx: WorkflowActivityContext, *args, **kwargs):
            wf_ctx = WorkflowActivityContext(ctx)
            try:
                call = task_instance(wf_ctx, *args, **kwargs)
                if asyncio.iscoroutine(call):
                    return run_sync(call)
                return call
            except Exception:
                logger.exception(f"Task '{task_name}' failed")
                raise

        return wrapper

    # TODO: workflow discovery can also come from dapr runtime
    # Python workflows can be registered in a variety of ways, and we need to support all of them.
    # This supports decorator-based registration;
    # however, there is also manual registration approach.
    # See example below:
    #     def setup_workflow_runtime():
    #       wf_runtime = WorkflowRuntime()
    #       wf_runtime.register_workflow(order_processing_workflow)
    #       wf_runtime.register_workflow(fulfillment_workflow)
    #       wf_runtime.register_activity(process_payment)
    #       wf_runtime.register_activity(send_notification)
    #     return wf_runtime

    # runtime = setup_workflow_runtime()
    # runtime.start()
    def _discover_workflows(self) -> Dict[str, Callable]:
        """Gather all @workflow-decorated functions and methods."""
        module = sys.modules["__main__"]
        wfs: Dict[str, Callable] = {}
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if getattr(fn, "_is_workflow", False) and fn.__module__ == module.__name__:
                wfs[getattr(fn, "_workflow_name", name)] = fn
        for name, method in get_decorated_methods(self, "_is_workflow").items():
            wfs[getattr(method, "_workflow_name", name)] = method
        logger.info(f"Discovered workflows: {list(wfs)}")
        return wfs

    def _register_workflows(self, wfs: Dict[str, Callable]) -> None:
        """Register each discovered workflow with the Dapr runtime."""
        for wf_name, method in wfs.items():
            # Use a closure helper to avoid late-binding capture issues.
            def make_wrapped(meth: Callable) -> Callable:
                @functools.wraps(meth)
                def wrapped(*args, **kwargs):
                    return meth(*args, **kwargs)

                return wrapped

            decorator = self.wf_runtime.workflow(name=wf_name)
            self.workflows[wf_name] = decorator(make_wrapped(method))

    def resolve_task(self, task: Union[str, Callable]) -> Callable:
        """
        Resolves a registered task function by its name or decorated function.

        Args:
            task (Union[str, Callable]): The task name or callable function.

        Returns:
            Callable: The resolved task function.

        Raises:
            AttributeError: If the task is not found.
        """
        if isinstance(task, str):
            task_name = task
        elif callable(task):
            task_name = getattr(
                task, "_task_name", getattr(task, "__name__", "unknown_task")
            )
        else:
            raise ValueError(f"Invalid task reference: {task}")

        task_func = self.tasks.get(task_name)
        if not task_func:
            raise AttributeError(f"Task '{task_name}' not found.")

        return task_func

    def resolve_workflow(self, workflow: Union[str, Callable]) -> Callable:
        """
        Resolves a registered workflow function by its name or decorated function.

        Args:
            workflow (Union[str, Callable]): The workflow name or callable function.

        Returns:
            Callable: The resolved workflow function.

        Raises:
            AttributeError: If the workflow is not found.
        """
        if isinstance(workflow, str):
            workflow_name = workflow  # Direct lookup by string name
        elif callable(workflow):
            workflow_name = getattr(
                workflow,
                "_workflow_name",
                getattr(workflow, "__name__", "unknown_workflow"),
            )
        else:
            raise ValueError(f"Invalid workflow reference: {workflow}")

        workflow_func = self.workflows.get(workflow_name)
        if not workflow_func:
            raise AttributeError(f"Workflow '{workflow_name}' not found.")

        return workflow_func

    def start_runtime(self):
        """Idempotently start the Dapr workflow runtime."""
        if not self.wf_runtime_is_running:
            logger.info("Starting workflow runtime.")
            self.wf_runtime.start()
            self.wf_runtime_is_running = True

            # Apply workflow instrumentation if callback was registered
            if hasattr(self.__class__, self.INSTRUMENTATION_CALLBACK_ATTR):
                getattr(self.__class__, self.INSTRUMENTATION_CALLBACK_ATTR)()
        else:
            logger.debug("Workflow runtime already running; skipping.")

    def stop_runtime(self):
        """Idempotently stop the Dapr workflow runtime."""
        if self.wf_runtime_is_running:
            logger.info("Stopping workflow runtime.")
            self.wf_runtime.shutdown()
            self.wf_runtime_is_running = False
        else:
            logger.debug("Workflow runtime already stopped; skipping.")

    def run_workflow(
        self, workflow: Union[str, Callable], input: Union[str, Dict[str, Any]] = None
    ) -> str:
        """
        Starts a workflow execution.

        Args:
            workflow (Union[str, Callable]): The workflow name or callable.
            input (Union[str, Dict[str, Any]], optional): Input data for the workflow.

        Returns:
            str: The instance ID of the started workflow.

        Raises:
            Exception: If workflow execution fails.
        """
        try:
            # Generate unique instance ID
            instance_id = uuid.uuid4().hex

            # Resolve the workflow function
            workflow_func = self.resolve_workflow(workflow)

            # Schedule workflow execution
            instance_id = self.wf_client.schedule_new_workflow(
                workflow=workflow_func, input=input, instance_id=instance_id
            )

            logger.info(f"Started workflow with instance ID {instance_id}.")
            return instance_id
        except Exception as e:
            logger.error(f"Failed to start workflow {workflow}: {e}")
            raise

    async def monitor_workflow_state(self, instance_id: str) -> Optional[WorkflowState]:
        """
        Monitors and retrieves the final state of a workflow instance.

        Args:
            instance_id (str): The workflow instance ID.

        Returns:
            Optional[WorkflowState]: The final state of the workflow or None if not found.
        """
        try:
            state: WorkflowState = await asyncio.to_thread(
                self.wait_for_workflow_completion,
                instance_id,
                fetch_payloads=True,
                timeout_in_seconds=self.timeout,
            )

            if not state:
                logger.error(f"Workflow '{instance_id}' not found.")
                return None

            return state
        except TimeoutError:
            logger.error(f"Workflow '{instance_id}' monitoring timed out.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving workflow state for '{instance_id}': {e}")
            return None

    async def monitor_workflow_completion(self, instance_id: str) -> None:
        """
        Monitors the execution of a workflow and logs its final state.

        Args:
            instance_id (str): The workflow instance ID.
        """
        try:
            logger.info(f"Monitoring workflow '{instance_id}'...")

            # Retrieve workflow state
            state: WorkflowState = await self.monitor_workflow_state(instance_id)
            if not state:
                return  # Error already logged in monitor_workflow_state

            # Extract relevant details
            workflow_status = state.runtime_status.name
            failure_details = (
                state.failure_details
            )  # This is an object, not a dictionary

            if workflow_status == "COMPLETED":
                logger.info(
                    f"Workflow '{instance_id}' completed successfully. Status: {workflow_status}."
                )

                if state.serialized_output:
                    logger.debug(
                        f"Output: {json.dumps(state.serialized_output, indent=2)}"
                    )

            elif workflow_status in ("FAILED", "ABORTED"):
                # Ensure `failure_details` exists before accessing attributes
                error_type = getattr(failure_details, "error_type", "Unknown")
                message = getattr(failure_details, "message", "No message provided")
                stack_trace = getattr(
                    failure_details, "stack_trace", "No stack trace available"
                )

                logger.error(
                    f"Workflow '{instance_id}' failed.\n"
                    f"Error Type: {error_type}\n"
                    f"Message: {message}\n"
                    f"Stack Trace:\n{stack_trace}\n"
                    f"Input: {json.dumps(state.serialized_input, indent=2)}"
                )

                self.terminate_workflow(instance_id)

            else:
                logger.warning(
                    f"Workflow '{instance_id}' ended with status '{workflow_status}'.\n"
                    f"Input: {json.dumps(state.serialized_input, indent=2)}"
                )

            logger.debug(
                f"Workflow Details: Instance ID={state.instance_id}, Name={state.name}, "
                f"Created At={state.created_at}, Last Updated At={state.last_updated_at}"
            )

        except Exception as e:
            logger.error(
                f"Error monitoring workflow '{instance_id}': {e}", exc_info=True
            )
        finally:
            logger.info(f"Finished monitoring workflow '{instance_id}'.")

    async def run_and_monitor_workflow_async(
        self,
        workflow: Union[str, Callable],
        input: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Runs a workflow asynchronously and monitors its completion.

        Args:
            workflow (Union[str, Callable]): The workflow name or callable.
            input (Optional[Union[str, Dict[str, Any]]]): The workflow input payload.

        Returns:
            Optional[str]: The serialized output of the workflow.
        """
        instance_id = None
        try:
            # Off-load the potentially blocking run_workflow call to a thread.
            instance_id = await asyncio.to_thread(self.run_workflow, workflow, input)

            # Await the asynchronous monitoring of the workflow state.
            state = await self.monitor_workflow_state(instance_id)

            if not state:
                raise RuntimeError(f"Workflow '{instance_id}' not found.")

            workflow_status = (
                DaprWorkflowStatus[state.runtime_status.name]
                if state.runtime_status.name in DaprWorkflowStatus.__members__
                else DaprWorkflowStatus.UNKNOWN
            )

            if workflow_status == DaprWorkflowStatus.COMPLETED:
                logger.info(f"Workflow '{instance_id}' completed successfully!")
                logger.debug(f"Output: {state.serialized_output}")
            else:
                logger.error(
                    f"Workflow '{instance_id}' ended with status '{workflow_status.value}'."
                )

            # Return the final state output
            return state.serialized_output

        except Exception as e:
            logger.error(f"Error during workflow '{instance_id}': {e}")
            raise
        finally:
            logger.info(f"Finished workflow with Instance ID: {instance_id}.")

    def run_and_monitor_workflow_sync(
        self,
        workflow: Union[str, Callable],
        input: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Synchronous wrapper for running and monitoring a workflow.
        This allows calling code that is not async to still run the workflow.

        Args:
            workflow (Union[str, Callable]): The workflow name or callable.
            input (Optional[Union[str, Dict[str, Any]]]): The workflow input payload.

        Returns:
            Optional[str]: The serialized output of the workflow.
        """
        return asyncio.run(self.run_and_monitor_workflow_async(workflow, input))

    def terminate_workflow(
        self, instance_id: str, *, output: Optional[Any] = None
    ) -> None:
        """
        Terminates a running workflow.

        Args:
            instance_id (str): The workflow instance ID.
            output (Optional[Any]): Optional output to set for the terminated workflow.

        Raises:
            Exception: If the termination fails.
        """
        try:
            self.wf_client.terminate_workflow(instance_id=instance_id, output=output)
            logger.info(
                f"Successfully terminated workflow '{instance_id}' with output: {output}"
            )
        except Exception as e:
            logger.error(f"Failed to terminate workflow '{instance_id}'. Error: {e}")
            raise Exception(f"Error terminating workflow '{instance_id}': {e}")

    def get_workflow_state(self, instance_id: str) -> Optional[Any]:
        """
        Retrieves the state of a workflow instance.

        Args:
            instance_id (str): The workflow instance ID.

        Returns:
            Optional[Any]: The workflow state if found.

        Raises:
            RuntimeError: If retrieving the state fails.
        """
        try:
            state = self.wf_client.get_workflow_state(instance_id)
            logger.info(
                f"Retrieved state for workflow {instance_id}: {state.runtime_status}."
            )
            return state
        except Exception as e:
            logger.error(f"Failed to retrieve workflow state for {instance_id}: {e}")
            return None

    def wait_for_workflow_completion(
        self,
        instance_id: str,
        fetch_payloads: bool = True,
        timeout_in_seconds: int = 120,
    ) -> Optional[WorkflowState]:
        """
        Waits for a workflow to complete and retrieves its state.

        Args:
            instance_id (str): The workflow instance ID.
            fetch_payloads (bool): Whether to fetch input/output payloads.
            timeout_in_seconds (int): Maximum wait time in seconds.

        Returns:
            Optional[WorkflowState]: The final state or None if it times out.

        Raises:
            RuntimeError: If waiting for completion fails.
        """
        try:
            state = self.wf_client.wait_for_workflow_completion(
                instance_id,
                fetch_payloads=fetch_payloads,
                timeout_in_seconds=timeout_in_seconds,
            )
            if state:
                logger.info(
                    f"Workflow {instance_id} completed with status: {state.runtime_status}."
                )
            else:
                logger.warning(
                    f"Workflow {instance_id} did not complete within the timeout period."
                )
            return state
        except Exception as e:
            logger.error(
                f"Error while waiting for workflow {instance_id} completion: {e}"
            )
            return None

    def raise_workflow_event(
        self, instance_id: str, event_name: str, *, data: Any | None = None
    ) -> None:
        """
        Raises an event for a running workflow instance.

        Args:
            instance_id (str): The workflow instance ID.
            event_name (str): The name of the event to raise.
            data (Any | None): Optional event data.

        Raises:
            Exception: If raising the event fails.
        """
        try:
            logger.info(
                f"Raising workflow event '{event_name}' for instance '{instance_id}'"
            )
            # Ensure data is in a serializable format
            if is_pydantic_model(type(data)):
                # Convert Pydantic model to dict
                data = data.model_dump()
            # Raise the event using the Dapr workflow client with serialized data
            self.wf_client.raise_workflow_event(
                instance_id=instance_id, event_name=event_name, data=data
            )
            logger.info(
                f"Successfully raised workflow event '{event_name}' for instance '{instance_id}'!"
            )
        except Exception as e:
            logger.error(
                f"Error raising workflow event '{event_name}' for instance '{instance_id}'. "
                f"Data: {data}, Error: {e}"
            )
            raise Exception(
                f"Failed to raise workflow event '{event_name}' for instance '{instance_id}': {str(e)}"
            )

    def when_all(self, tasks: List[dtask.Task[T]]) -> dtask.WhenAllTask[T]:
        """
        Waits for all given tasks to complete.

        Args:
            tasks (List[dtask.Task[T]]): The tasks to wait for.

        Returns:
            dtask.WhenAllTask[T]: A task that completes when all tasks finish.
        """
        return dtask.when_all(tasks)

    def when_any(self, tasks: List[dtask.Task[T]]) -> dtask.WhenAnyTask:
        """
        Waits for any one of the given tasks to complete.

        Args:
            tasks (List[dtask.Task[T]]): The tasks to monitor.

        Returns:
            dtask.WhenAnyTask: A task that completes when the first task finishes.
        """
        return dtask.when_any(tasks)
