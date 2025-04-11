import asyncio
import functools
import inspect
import json
import logging
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

from durabletask import task as dtask

from dapr.clients import DaprClient
from dapr.clients.grpc._request import TransactionOperationType, TransactionalStateOperation
from dapr.clients.grpc._response import StateResponse
from dapr.clients.grpc._state import Concurrency, Consistency, StateOptions
from dapr.ext.workflow import DaprWorkflowClient, WorkflowActivityContext, WorkflowRuntime
from dapr.ext.workflow.workflow_state import WorkflowState

from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.types.workflow import DaprWorkflowStatus
from dapr_agents.workflow.task import WorkflowTask
from dapr_agents.workflow.utils import get_callable_decorated_methods

logger = logging.getLogger(__name__)

T = TypeVar('T')

class WorkflowApp(BaseModel):
    """
    A Pydantic-based class to encapsulate a Dapr Workflow runtime and manage workflows and tasks.
    """

    llm: Optional[ChatClientBase] = Field(default=None, description="The default LLM client for all LLM-based tasks.")
    timeout: int = Field(default=300, description="Default timeout duration in seconds for workflow tasks.")

    # Initialized in model_post_init
    wf_runtime: Optional[WorkflowRuntime] = Field(default=None, init=False, description="Workflow runtime instance.")
    wf_runtime_is_running: Optional[bool] = Field(default=None, init=False, description="Is the Workflow runtime running?.")
    wf_client: Optional[DaprWorkflowClient] = Field(default=None, init=False, description="Workflow client instance.")
    client: Optional[DaprClient] = Field(default=None, init=False, description="Dapr client instance.")
    tasks: Dict[str, Callable] = Field(default_factory=dict, init=False, description="Dictionary of registered tasks.")
    workflows: Dict[str, Callable] = Field(default_factory=dict, init=False, description="Dictionary of registered workflows.")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization configuration for the WorkflowApp.

        Initializes the Dapr Workflow runtime, client, and state store, and ensures
        that workflows and tasks are registered.
        """

        # Initialize WorkflowRuntime and DaprWorkflowClient
        self.wf_runtime = WorkflowRuntime()
        self.wf_runtime_is_running = False
        self.wf_client = DaprWorkflowClient()
        self.client = DaprClient()

        logger.info(f"Initialized WorkflowApp.")

        # Register workflows and tasks after the instance is created
        self.register_all_workflows()
        self.register_all_tasks()

        # Proceed with base model setup
        super().model_post_init(__context)
    
    def register_agent(self, store_name: str, store_key: str, agent_name: str, agent_metadata: dict) -> None:
        """
        Merges the existing data with the new data and updates the store.

        Args:
            store_name (str): The name of the Dapr state store component.
            key (str): The key to update.
            data (dict): The data to update the store with.
        """
        # retry the entire operation up to ten times sleeping 1 second between each attempt
        for attempt in range(1, 11):
            try:
                response: StateResponse = self.client.get_state(store_name=store_name, key=store_key)
                if not response.etag:
                    # if there is no etag the following transaction won't work as expected
                    # so we need to save an empty object with a strong consistency to force the etag to be created
                    self.client.save_state(
                        store_name=store_name,
                        key=store_key,
                        value=json.dumps({}),
                        state_metadata={"contentType": "application/json"},
                        options=StateOptions(concurrency=Concurrency.first_write, consistency=Consistency.strong)
                    )
                    # raise an exception to retry the entire operation
                    raise Exception(f"No etag found for key: {store_key}")
                existing_data = json.loads(response.data) if response.data else {}
                if (agent_name, agent_metadata) in existing_data.items():
                    logger.debug(f"agent {agent_name} already registered.")
                    return None
                agent_data = {agent_name: agent_metadata}
                merged_data = {**existing_data, **agent_data}
                logger.debug(f"merged data: {merged_data} etag: {response.etag}")
                try:
                    # using the transactional API to be able to later support the Dapr outbox pattern
                    self.client.execute_state_transaction(
                        store_name=store_name,
                        operations=[
                            TransactionalStateOperation(
                                key=store_key,
                                data=json.dumps(merged_data),
                                etag=response.etag,
                                operation_type=TransactionOperationType.upsert
                            )
                        ],
                        transactional_metadata={"contentType": "application/json"}
                    )
                except Exception as e:
                    raise e
                return None
            except Exception as e:
                logger.debug(f"Error on transaction attempt: {attempt}: {e}")
                logger.debug(f"Sleeping for 1 second before retrying transaction...")
                time.sleep(1)
        raise Exception(f"Failed to update state store key: {store_key} after 10 attempts.")

    def get_data_from_store(self, store_name: str, key: str) -> Tuple[bool, dict]:
        """
        Retrieves data from the Dapr state store using the given key.

        Args:
            store_name (str): The name of the Dapr state store component.
            key (str): The key to fetch data from.

        Returns:
            Tuple[bool, dict]: A tuple indicating if data was found (bool) and the retrieved data (dict).
        """
        try:
            response: StateResponse = self.client.get_state(store_name=store_name, key=key)
            data = response.data
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Error retrieving data for key '{key}' from store '{store_name}'")
            return None

    def register_all_tasks(self):
        """
        Registers all collected tasks with Dapr while preserving execution logic.
        """
        current_module = sys.modules["__main__"]

        all_functions = {}
        for name, func in inspect.getmembers(current_module, inspect.isfunction):
            if hasattr(func, "_is_task") and func.__module__ == current_module.__name__:
                task_name = getattr(func, "_task_name", None) or name
                all_functions[task_name] = func
        
        # Load instance methods that are tasks
        task_methods = get_callable_decorated_methods(self, "_is_task")
        for method_name, method in task_methods.items():
            task_name = getattr(method, "_task_name", method_name)
            all_functions[task_name] = method

        logger.debug(f"Discovered tasks: {list(all_functions.keys())}")

        def make_task_wrapper(method):
            """Creates a unique task wrapper bound to a specific method reference."""
            # Extract stored metadata from the function
            task_name = getattr(method, "_task_name", method.__name__)
            explicit_llm = getattr(method, "_explicit_llm", False)

            # Always initialize `llm` as `None` explicitly first
            llm = None

            # If task is explicitly LLM-based, but has no LLM, use `self.llm`
            if explicit_llm and self.llm is not None:
                llm = self.llm 

            task_kwargs = getattr(method, "_task_kwargs", {})

            task_instance = WorkflowTask(
                func=method,
                description=getattr(method, "_task_description", None),
                agent=getattr(method, "_task_agent", None),
                llm=llm,
                include_chat_history=getattr(method, "_task_include_chat_history", False),
                workflow_app=self,
                **task_kwargs
            )

            def run_in_event_loop(coroutine):
                """Ensures that an async function runs synchronously if needed."""
                try:
                    loop = asyncio.get_running_loop()
                    return loop.run_until_complete(coroutine)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(coroutine)

            @functools.wraps(method)
            def task_wrapper(ctx: WorkflowActivityContext, *args, **kwargs):
                """Wrapper function for executing tasks in a Dapr workflow, handling both sync and async tasks."""
                wf_ctx = WorkflowActivityContext(ctx)

                try:
                    if inspect.iscoroutinefunction(method) or asyncio.iscoroutinefunction(task_instance.__call__):
                        return run_in_event_loop(task_instance(wf_ctx, *args, **kwargs))
                    else:
                        return task_instance(wf_ctx, *args, **kwargs)
                except Exception as e:
                    raise RuntimeError(f"Task '{task_name}' execution failed: {e}")

            return task_name, task_wrapper  # Return both name and wrapper

        for method in all_functions.values():
            # Ensure function reference is properly preserved inside a function scope
            task_name, task_wrapper = make_task_wrapper(method)

            # Register the task with Dapr Workflow using the correct task name
            activity_decorator = self.wf_runtime.activity(name=task_name)
            registered_activity = activity_decorator(task_wrapper)

            # Store task reference
            self.tasks[task_name] = registered_activity
    
    def register_all_workflows(self):
        """
        Registers all workflow functions dynamically with Dapr.
        """
        current_module = sys.modules["__main__"]

        all_workflows = {}
        # Load global-level workflow functions
        for name, func in inspect.getmembers(current_module, inspect.isfunction):
            if hasattr(func, "_is_workflow") and func.__module__ == current_module.__name__:
                workflow_name = getattr(func, "_workflow_name", None) or name
                all_workflows[workflow_name] = func

        # Load instance methods that are workflows
        workflow_methods = get_callable_decorated_methods(self, "_is_workflow")
        for method_name, method in workflow_methods.items():
            workflow_name = getattr(method, "_workflow_name", method_name)
            all_workflows[workflow_name] = method

        logger.info(f"Discovered workflows: {list(all_workflows.keys())}")

        def make_workflow_wrapper(method):
            """Creates a wrapper to prevent pointer overwrites during workflow registration."""
            workflow_name = getattr(method, "_workflow_name", method.__name__)

            @functools.wraps(method)
            def workflow_wrapper(*args, **kwargs):
                """Directly calls the method without modifying ctx injection (already handled)."""
                try:
                    return method(*args, **kwargs)
                except Exception as e:
                    raise RuntimeError(f"Workflow '{workflow_name}' execution failed: {e}")

            return workflow_name, workflow_wrapper

        for method in all_workflows.values():
            workflow_name, workflow_wrapper = make_workflow_wrapper(method)

            # Register the workflow with Dapr using the correct name
            workflow_decorator = self.wf_runtime.workflow(name=workflow_name)
            registered_workflow = workflow_decorator(workflow_wrapper)

            # Store workflow reference
            self.workflows[workflow_name] = registered_workflow
    
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
            task_name = getattr(task, "_task_name", task.__name__)
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
            workflow_name = getattr(workflow, "_workflow_name", workflow.__name__)
        else:
            raise ValueError(f"Invalid workflow reference: {workflow}")

        workflow_func = self.workflows.get(workflow_name)
        if not workflow_func:
            raise AttributeError(f"Workflow '{workflow_name}' not found.")

        return workflow_func

    def run_workflow(self, workflow: Union[str, Callable], input: Union[str, Dict[str, Any]] = None) -> str:
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
            # Start Workflow Runtime
            if not self.wf_runtime_is_running:
                self.start_runtime()
                self.wf_runtime_is_running = True

            # Generate unique instance ID
            instance_id = str(uuid.uuid4()).replace("-", "")

            # Resolve the workflow function
            workflow_func = self.resolve_workflow(workflow)

            # Schedule workflow execution
            instance_id = self.wf_client.schedule_new_workflow(
                workflow=workflow_func,
                input=input,
                instance_id=instance_id
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
            state = await self.monitor_workflow_state(instance_id)
            if not state:
                return  # Error already logged in monitor_workflow_state

            # Extract relevant details
            workflow_status = state.runtime_status.name
            failure_details = state.failure_details  # This is an object, not a dictionary

            if workflow_status == "COMPLETED":
                logger.info(f"Workflow '{instance_id}' completed successfully. Status: {workflow_status}.")

                if state.serialized_output:
                    logger.debug(f"Output: {json.dumps(state.serialized_output, indent=2)}")

            elif workflow_status == "FAILED":
                # Ensure `failure_details` exists before accessing attributes
                error_type = getattr(failure_details, "error_type", "Unknown")
                message = getattr(failure_details, "message", "No message provided")
                stack_trace = getattr(failure_details, "stack_trace", "No stack trace available")

                logger.error(
                    f"Workflow '{instance_id}' failed.\n"
                    f"Error Type: {error_type}\n"
                    f"Message: {message}\n"
                    f"Stack Trace:\n{stack_trace}\n"
                    f"Input: {json.dumps(state.serialized_input, indent=2)}"
                )

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
            logger.error(f"Error monitoring workflow '{instance_id}': {e}", exc_info=True)
        finally:
            logger.info(f"Finished monitoring workflow '{instance_id}'.")
    
    def run_and_monitor_workflow(self, workflow: Union[str, Callable], input: Optional[Union[str, Dict[str, Any]]] = None) -> Optional[str]:
        """
        Runs a workflow synchronously and monitors its completion.

        Args:
            workflow (Union[str, Callable]): The workflow name or callable.
            input (Optional[Union[str, Dict[str, Any]]]): The workflow input.

        Returns:
            Optional[str]: The serialized output of the workflow.
        """
        instance_id = None
        try:
            # Schedule the workflow
            instance_id = self.run_workflow(workflow, input=input)

            # Ensure we run within a new asyncio event loop
            state = asyncio.run(self.monitor_workflow_state(instance_id))

            if not state:
                raise RuntimeError(f"Workflow '{instance_id}' not found.")

            workflow_status = DaprWorkflowStatus[state.runtime_status.name] if state.runtime_status.name in DaprWorkflowStatus.__members__ else DaprWorkflowStatus.UNKNOWN

            if workflow_status == DaprWorkflowStatus.COMPLETED:
                logger.info(f"Workflow '{instance_id}' completed successfully!")
                logger.debug(f"Output: {state.serialized_output}")
            else:
                logger.error(f"Workflow '{instance_id}' ended with status '{workflow_status.value}'.")

            # Return the final state output
            return state.serialized_output

        except Exception as e:
            logger.error(f"Error during workflow '{instance_id}': {e}")
            raise
        finally:
            logger.info(f"Finished workflow with Instance ID: {instance_id}.")
            self.stop_runtime()
    
    def terminate_workflow(self, instance_id: str, *, output: Optional[Any] = None) -> None:
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
            logger.info(f"Successfully terminated workflow '{instance_id}' with output: {output}")
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
            logger.info(f"Retrieved state for workflow {instance_id}: {state.runtime_status}.")
            return state
        except Exception as e:
            logger.error(f"Failed to retrieve workflow state for {instance_id}: {e}")
            return None
    
    def wait_for_workflow_completion(self, instance_id: str, fetch_payloads: bool = True, timeout_in_seconds: int = 120) -> Optional[WorkflowState]:
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
                instance_id, fetch_payloads=fetch_payloads, timeout_in_seconds=timeout_in_seconds
            )
            if state:
                logger.info(f"Workflow {instance_id} completed with status: {state.runtime_status}.")
            else:
                logger.warning(f"Workflow {instance_id} did not complete within the timeout period.")
            return state
        except Exception as e:
            logger.error(f"Error while waiting for workflow {instance_id} completion: {e}")
            return None
    
    def raise_workflow_event(self, instance_id: str, event_name: str, *, data: Any | None = None) -> None:
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
            logger.info(f"Raising workflow event '{event_name}' for instance '{instance_id}'")
            self.wf_client.raise_workflow_event(instance_id=instance_id, event_name=event_name, data=data)
            logger.info(f"Successfully raised workflow event '{event_name}' for instance '{instance_id}'!")
        except Exception as e:
            logger.error(
                f"Error raising workflow event '{event_name}' for instance '{instance_id}'. "
                f"Data: {data}, Error: {e}"
            )
            raise Exception(f"Failed to raise workflow event '{event_name}' for instance '{instance_id}': {str(e)}")
    
    def invoke_service(self, service: str, method: str, http_method: str = "POST", input: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Any:
        """
        Invokes an external service via Dapr.

        Args:
            service (str): The service name.
            method (str): The method to call.
            http_method (str, optional): The HTTP method (default: "POST").
            input (Optional[Dict[str, Any]], optional): The request payload.
            timeout (Optional[int], optional): Timeout in seconds.

        Returns:
            Any: The response from the service.

        Raises:
            Exception: If the invocation fails.
        """
        try:
            resp = self.client.invoke_method(
                app_id=service,
                method_name=method,
                http_verb=http_method,
                data=json.dumps(input) if input else None,
                timeout=timeout
            )
            if resp.status_code != 200:
                raise Exception(f"Error calling {service}.{method}: {resp.status_code}: {resp.text}")

            agent_response = json.loads(resp.data.decode("utf-8"))
            logger.info(f"Agent's Response: {agent_response}")
            return agent_response
        except Exception as e:
            logger.error(f"Failed to invoke {service}.{method}: {e}")
            raise e
    
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
    
    def start_runtime(self):
        """
        Starts the Dapr workflow runtime
        """

        logger.info("Starting workflow runtime.")
        self.wf_runtime.start()
    
    def stop_runtime(self):
        """
        Stops the Dapr workflow runtime.
        """
        logger.info("Stopping workflow runtime.")
        self.wf_runtime.shutdown()