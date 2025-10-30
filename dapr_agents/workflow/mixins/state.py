import json
import logging
import os
import tempfile
import threading
from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)

state_lock = threading.Lock()


class StateManagementMixin:
    """
    Mixin providing workflow state initialization, validation, and persistence.
    """

    def _reconcile_workflow_statuses(self) -> None:
        """
        Reconcile workflow statuses between our Redis state and Dapr's actual workflow state.

        This method checks Dapr's actual status and updates our state to match,
        preventing stale "running" workflows from blocking new executions.
        """
        from dapr.clients import DaprClient

        instances = self.memory_store._current_state["instances"]
        updated_instances = []

        for instance_id, instance_data in instances.items():
            our_status = instance_data["status"].lower()

            # Only check running instances (completed/failed instances are already finalized)
            if our_status in ["running", "pending"]:
                try:
                    # Query Dapr for the actual workflow status
                    with DaprClient() as client:
                        state = client.get_workflow(
                            instance_id=instance_id,
                        )

                        dapr_status = state.runtime_status.upper()

                        # If Dapr says FAILED/TERMINATED but we say RUNNING, update our state
                        if dapr_status in ["FAILED", "TERMINATED", "CANCELED"]:
                            logger.warning(
                                f"Workflow {instance_id} is {dapr_status} in Dapr but 'running' in Redis. "
                                f"Updating Redis state to match Dapr."
                            )
                            instance_data["status"] = dapr_status.lower()
                            instance_data["end_time"] = datetime.now().isoformat()
                            updated_instances.append(instance_id)

                            # Save the updated instance back to Redis
                            instance_key = self.memory_store._get_instance_key(
                                instance_id
                            )
                            self.memory_store._save_state_with_metadata(
                                instance_key, instance_data
                            )

                        elif dapr_status == "COMPLETED":
                            logger.info(
                                f"Workflow {instance_id} completed in Dapr. Updating Redis state."
                            )
                            instance_data["status"] = "completed"
                            if not instance_data["end_time"]:
                                instance_data["end_time"] = datetime.now().isoformat()
                            updated_instances.append(instance_id)

                            # Save the updated instance
                            instance_key = self.memory_store._get_instance_key(
                                instance_id
                            )
                            self.memory_store._save_state_with_metadata(
                                instance_key, instance_data
                            )

                except Exception as e:
                    logger.debug(
                        f"Could not query Dapr status for workflow {instance_id}: {e}. "
                        f"Instance may have been purged or not exist in Dapr yet."
                    )

        if updated_instances:
            logger.info(
                f"Reconciled {len(updated_instances)} workflow status(es) with Dapr: {updated_instances}"
            )

    def _has_valid_message_sequence(self, instance_data: dict) -> bool:
        """
        Validate that all assistant messages with tool_calls have corresponding tool responses.
        This prevents loading instances with incomplete tool call sequences that would break LLM calls.

        Args:
            instance_data: The workflow instance data to validate

        Returns:
            bool: True if message sequence is valid, False otherwise
        """
        messages = instance_data["messages"]

        # Collect all tool_call_ids that need responses
        pending_tool_calls = set()

        for msg in messages:
            msg_dict = (
                msg
                if isinstance(msg, dict)
                else (msg.model_dump() if hasattr(msg, "model_dump") else {})
            )
            role = msg_dict.get("role", [])

            if role == "assistant" and msg_dict.get("tool_calls", []):
                # Add all tool_call_ids from this assistant message
                for tool_call in msg_dict.get("tool_calls", []):
                    if isinstance(tool_call, dict):
                        pending_tool_calls.add(tool_call["id"])

            elif role == "tool":
                # Remove this tool_call_id as it has a response
                tool_call_id = msg_dict.get("tool_call_id", [])
                if tool_call_id in pending_tool_calls:
                    pending_tool_calls.remove(tool_call_id)

        # If there are still pending tool calls, the sequence is invalid
        if pending_tool_calls:
            logger.debug(
                f"Invalid message sequence: pending tool_call_ids: {pending_tool_calls}"
            )
            return False

        return True

    # TODO: Delete this once we rm orchestrators in favor of agents as tools.
    @property
    def state(self) -> dict:
        """
        Get the current workflow state.

        Returns:
            dict: The current workflow state.
        """
        return self.memory_store._current_state if hasattr(self, "storage") else {}

    # TODO: Delete this once we rm orchestrators in favor of agents as tools.
    @state.setter
    def state(self, value: dict) -> None:
        """
        Set the current workflow state.

        Args:
            value (dict): The new workflow state.
        """
        if hasattr(self, "storage"):
            self.memory_store._current_state = value

    def initialize_state(self) -> None:
        """
        Initialize workflow state from provided value or storage.

        Raises:
            RuntimeError: If state initialization or loading from storage fails.
        """
        try:
            if self.memory_store._current_state is None:
                logger.debug("No user-provided state. Attempting to load from storage.")
                self.memory_store._current_state = self.load_state()

            if isinstance(self.memory_store._current_state, BaseModel):
                logger.debug(
                    "User provided a state as a Pydantic model. Converting to dict."
                )
                self.memory_store._current_state = (
                    self.memory_store._current_state.model_dump()
                )

            if not isinstance(self.memory_store._current_state, dict):
                raise TypeError(
                    f"Invalid state type: {type(self.memory_store._current_state)}. Expected dict."
                )

            logger.debug(
                f"Workflow state initialized with {len(self.memory_store._current_state)} key(s)."
            )
            self.save_state()
        except Exception as e:
            raise RuntimeError(f"Error initializing workflow state: {e}") from e

    def load_state(self) -> dict:
        """
        Load the workflow state from the configured Dapr state store.

        Returns:
            dict: The loaded and optionally validated state.

        Raises:
            RuntimeError: If the state store is not properly configured.
            TypeError: If the retrieved state is not a dictionary.
            ValidationError: If state schema validation fails.
        """
        try:
            if (
                not self._dapr_client
                or not self.memory_store.name
                or not self.memory_store._key
            ):
                logger.error("State store is not configured. Cannot load state.")
                raise RuntimeError(
                    "State store is not configured. Please provide 'storage.name'."
                )

            # For durable agents, always load from database to ensure it's the source of truth
            response = self._dapr_client.get_state(
                self.memory_store.name, self.memory_store._key
            )
            if response.data:
                state_data = self._deserialize_state(response.data)
                self.memory_store._current_state = state_data
            else:
                self.memory_store._current_state = {}

            # Load workflow instances from ALL sessions to support workflow resumption after restart
            # This ensures that if the app crashes mid-workflow and restarts, all in-flight
            # workflows across all sessions will be loaded and can be resumed by Dapr
            # Always ensure "instances" key exists
            self.memory_store._current_state.setdefault("instances", {})

            # Get all sessions for this agent
            sessions_index_key = self.memory_store._get_sessions_index_key()
            response = self._dapr_client.get_state(
                self.memory_store.name, sessions_index_key
            )

            if response.data:
                index_data = self._deserialize_state(response.data)
                session_ids = index_data.get("sessions", [])
                logger.debug(
                    f"Found {len(session_ids)} session(s) for agent '{self.memory_store._agent_name}'"
                )

                # Load workflow instances from each session
                for session_id in session_ids:
                    session_key = self.memory_store._get_session_key(session_id)
                    response = self._dapr_client.get_state(
                        self.memory_store.name, session_key
                    )

                    if response.data:
                        session_data = self._deserialize_state(response.data)

                        instance_ids = session_data.get("workflow_instances", [])
                        logger.debug(
                            f"Loading {len(instance_ids)} instance(s) from session '{session_id}'"
                        )

                        # Load each instance
                        for instance_id in instance_ids:
                            instance_key = self.memory_store._get_instance_key(
                                instance_id
                            )
                            response = self._dapr_client.get_state(
                                self.memory_store.name, instance_key
                            )
                            if response.data:
                                instance_data = self._deserialize_state(response.data)

                                # Validate message sequence before loading, but ONLY for completed workflows
                                # Running workflows are expected to have incomplete sequences mid-execution
                                status = instance_data["status"].lower()
                                if status in ["running", "pending"]:
                                    # Always load running/pending instances (they're allowed to be incomplete)
                                    self.memory_store._current_state["instances"][
                                        instance_id
                                    ] = instance_data
                                    logger.debug(
                                        f"Loaded active workflow instance {instance_id} from key '{instance_key}' (session: {session_id}, status: {status})"
                                    )
                                elif self._has_valid_message_sequence(instance_data):
                                    # For completed/failed instances, validate message sequence
                                    self.memory_store._current_state["instances"][
                                        instance_id
                                    ] = instance_data
                                    logger.debug(
                                        f"Loaded completed workflow instance {instance_id} from key '{instance_key}' (session: {session_id}, status: {status})"
                                    )
                                else:
                                    logger.warning(
                                        f"Skipping completed instance {instance_id} due to invalid message sequence (incomplete tool calls, status: {status})"
                                    )

            # Reconcile workflow statuses with Dapr's actual state
            self._reconcile_workflow_statuses()

            logger.debug(
                f"Set self.memory_store._current_state to loaded data: {self.memory_store._current_state}"
            )
            return self.memory_store._current_state
        except Exception as e:
            logger.error(
                f"Failed to load state for key '{self.memory_store._key}': {e}"
            )
            raise RuntimeError(f"Error loading workflow state: {e}") from e

    def get_local_state_file_path(self) -> str:
        """
        Return the file path for saving the local state.

        Returns:
            str: The absolute path to the local state file.
        """
        if not self.memory_store.local_directory:
            return os.path.join(os.getcwd(), f"{self.name}_state.json")
        os.makedirs(self.memory_store.local_directory, exist_ok=True)

        # If relative path, make it absolute from workspace root
        if not os.path.isabs(self.memory_store.local_directory):
            abs_path = os.path.join(os.getcwd(), self.memory_store.local_directory)
        else:
            abs_path = self.memory_store.local_directory

        return os.path.join(abs_path, f"{self.name}_state.json")

    def save_state_to_disk(
        self, state_data: str, filename: Optional[str] = None
    ) -> None:
        """
        Safely save the workflow state to a local JSON file.

        Args:
            state_data: The state data to save (as JSON string or dict).
            filename: Optional filename for the state file.

        Raises:
            RuntimeError: If saving to disk fails.
        """
        try:
            file_path = filename or self.get_local_state_file_path()
            save_directory = os.path.dirname(file_path)
            os.makedirs(save_directory, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                "w", dir=save_directory, delete=False
            ) as tmp_file:
                tmp_file.write(state_data)
                temp_path = tmp_file.name

            with state_lock:
                existing_state = {}
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as file:
                        try:
                            existing_state = json.load(file)
                        except json.JSONDecodeError:
                            logger.warning(
                                "Existing state file is corrupt or empty. Overwriting."
                            )

                new_state = (
                    json.loads(state_data)
                    if isinstance(state_data, str)
                    else state_data
                )
                merged_state = {**existing_state, **new_state}

                with open(temp_path, "w", encoding="utf-8") as file:
                    json.dump(merged_state, file, indent=4)

                os.replace(temp_path, file_path)

            logger.debug(f"Workflow state saved locally at '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to save workflow state to disk: {e}")
            raise RuntimeError(f"Error saving workflow state to disk: {e}")

    def save_state(
        self,
        state: Optional[Union[dict, BaseModel, str]] = None,
        force_reload: bool = False,
    ) -> None:
        """
        Save the current workflow state to Dapr and optionally to disk.

        Args:
            state: The new state to save. If not provided, saves the existing state.
            force_reload: If True, reloads the state from the store after saving.

        Raises:
            RuntimeError: If the state store is not configured.
            TypeError: If the provided state is not a supported type.
            ValueError: If the provided state is a string but not valid JSON.
        """
        try:
            if (
                not self._dapr_client
                or not self.memory_store.name
                or not self.memory_store._key
            ):
                logger.error("State store is not configured. Cannot save state.")
                raise RuntimeError(
                    "State store is not configured. Please provide 'storage.name'."
                )

            self.memory_store._current_state = state or self.memory_store._current_state
            if not self.memory_store._current_state:
                logger.warning("Skipping state save: Empty state.")
                return

            if isinstance(self.memory_store._current_state, BaseModel):
                state_to_save = self.memory_store._current_state.model_dump_json()
            elif isinstance(self.memory_store._current_state, dict):
                state_to_save = json.dumps(self.memory_store._current_state)
            elif isinstance(self.memory_store._current_state, str):
                try:
                    json.loads(self.memory_store._current_state)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON string provided as state: {e}")
                state_to_save = self.memory_store._current_state
            else:
                raise TypeError(
                    f"Invalid state type: {type(self.memory_store._current_state)}. Expected dict, BaseModel, or JSON string."
                )

            # Save each workflow instance separately
            if "instances" in self.memory_store._current_state:
                for instance_id, instance_data in self.memory_store._current_state[
                    "instances"
                ].items():
                    instance_key = self.memory_store._get_instance_key(instance_id)
                    # Handle both dict and already-serialized string
                    if isinstance(instance_data, dict):
                        instance_json = json.dumps(instance_data)
                    elif isinstance(instance_data, str):
                        instance_json = instance_data
                    else:
                        instance_json = json.dumps(instance_data)
                    self._dapr_client.save_state(
                        self.memory_store.name, instance_key, instance_json
                    )
                    logger.debug(
                        f"Saved workflow instance {instance_id} to key '{instance_key}'"
                    )

            # Save other state data (like chat_history) to main key
            other_state = {
                k: v
                for k, v in self.memory_store._current_state.items()
                if k != "instances"
            }
            if other_state:
                other_state_json = json.dumps(other_state)
                self._dapr_client.save_state(
                    self.memory_store.name, self.memory_store._key, other_state_json
                )
                logger.debug(
                    f"Saved non-instance state to key '{self.memory_store._key}'"
                )

            if self.memory_store.local_directory is not None:
                self.save_state_to_disk(state_data=state_to_save)

            if force_reload:
                self.memory_store._current_state = self.load_state()
                logger.debug(
                    f"State reloaded after saving for key '{self.memory_store._key}'."
                )
        except Exception as e:
            logger.error(
                f"Failed to save state for key '{self.memory_store._key}': {e}"
            )
            raise

    def _deserialize_state(self, raw: Union[bytes, str, dict]) -> dict:
        """
        Convert Dapr's raw payload (bytes, JSON string, or already a dict) into a dict.
        Raises helpful errors on failure.
        """
        if isinstance(raw, dict):
            return raw

        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("State bytes are not valid UTF-8") from exc

        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"State is not valid JSON: {exc}") from exc

        raise TypeError(f"Unsupported state type {type(raw)!r}")
