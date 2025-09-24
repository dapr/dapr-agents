import json
import logging
import os
import tempfile
import threading
from typing import Optional, Union

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

state_lock = threading.Lock()


class StateManagementMixin:
    """
    Mixin providing workflow state initialization, validation, and persistence.
    """

    def initialize_state(self) -> None:
        """
        Initialize workflow state from provided value or storage.

        Raises:
            RuntimeError: If state initialization or loading from storage fails.
        """
        try:
            if self.state is None:
                logger.debug("No user-provided state. Attempting to load from storage.")
                self.state = self.load_state()

            if isinstance(self.state, BaseModel):
                logger.debug(
                    "User provided a state as a Pydantic model. Converting to dict."
                )
                self.state = self.state.model_dump()

            if not isinstance(self.state, dict):
                raise TypeError(
                    f"Invalid state type: {type(self.state)}. Expected dict."
                )

            logger.debug(f"Workflow state initialized with {len(self.state)} key(s).")
            self.save_state()
        except Exception as e:
            raise RuntimeError(f"Error initializing workflow state: {e}") from e

    def validate_state(self, state_data: dict) -> dict:
        """
        Validate the workflow state against ``state_format`` if provided.

        Args:
            state_data: The raw state data to validate.

        Returns:
            dict: The validated and structured state.

        Raises:
            ValidationError: If the state data does not conform to the expected schema.
        """
        try:
            if not self.state_format:
                logger.warning(
                    "No schema (state_format) provided; returning state as-is."
                )
                return state_data

            logger.debug("Validating workflow state against schema.")
            validated_state: BaseModel = self.state_format(**state_data)
            return validated_state.model_dump()
        except ValidationError as e:
            raise ValidationError(f"Invalid workflow state: {e.errors()}") from e

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
                not self._state_store_client
                or not self.state_store_name
                or not self.state_key
            ):
                logger.error("State store is not configured. Cannot load state.")
                raise RuntimeError(
                    "State store is not configured. Please provide 'state_store_name' and 'state_key'."
                )

            # For durable agents, always load from database to ensure it's the source of truth
            has_state, state_data = self._state_store_client.try_get_state(
                self.state_key
            )
            if has_state and state_data:
                logger.debug(
                    f"Existing state found for key '{self.state_key}'. Validating it."
                )
                if not isinstance(state_data, dict):
                    raise TypeError(
                        f"Invalid state type retrieved: {type(state_data)}. Expected dict."
                    )

                # Set self.state to the loaded data
                if self.state_format:
                    loaded_state = self.validate_state(state_data)
                else:
                    loaded_state = state_data

                self.state = loaded_state
                logger.debug(f"Set self.state to loaded data: {self.state}")

                return loaded_state

            logger.debug(
                f"No existing state found for key '{self.state_key}'. Initializing empty state."
            )
            return {}
        except Exception as e:
            logger.error(f"Failed to load state for key '{self.state_key}': {e}")
            raise RuntimeError(f"Error loading workflow state: {e}") from e

    def get_local_state_file_path(self) -> str:
        """
        Return the file path for saving the local state.

        Returns:
            str: The absolute path to the local state file.
        """
        directory = self.local_state_path or os.getcwd()
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{self.state_key}.json")

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
            save_directory = self.local_state_path or os.getcwd()
            os.makedirs(save_directory, exist_ok=True)
            filename = filename or f"{self.name}_state.json"
            file_path = os.path.join(save_directory, filename)

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
                not self._state_store_client
                or not self.state_store_name
                or not self.state_key
            ):
                logger.error("State store is not configured. Cannot save state.")
                raise RuntimeError(
                    "State store is not configured. Please provide 'state_store_name' and 'state_key'."
                )

            self.state = state or self.state
            if not self.state:
                logger.warning("Skipping state save: Empty state.")
                return

            if isinstance(self.state, BaseModel):
                state_to_save = self.state.model_dump_json()
            elif isinstance(self.state, dict):
                state_to_save = json.dumps(self.state)
            elif isinstance(self.state, str):
                try:
                    json.loads(self.state)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON string provided as state: {e}")
                state_to_save = self.state
            else:
                raise TypeError(
                    f"Invalid state type: {type(self.state)}. Expected dict, BaseModel, or JSON string."
                )

            self._state_store_client.save_state(self.state_key, state_to_save)
            logger.debug(f"Successfully saved state for key '{self.state_key}'.")

            if self.save_state_locally:
                self.save_state_to_disk(state_data=state_to_save)

            if force_reload:
                self.state = self.load_state()
                logger.debug(f"State reloaded after saving for key '{self.state_key}'.")
        except Exception as e:
            logger.error(f"Failed to save state for key '{self.state_key}': {e}")
            raise
