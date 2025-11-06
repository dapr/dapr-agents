from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

from dapr.clients.grpc._response import BulkStateItem, StateResponse
from pydantic import BaseModel, ValidationError

from dapr_agents.storage.daprstores.statestore import (
    DaprStateStore,
    _coerce_state_options,
)

logger = logging.getLogger(__name__)


class StateStoreError(RuntimeError):
    """Raised when high-level state operations fail."""


class StateStoreService:
    """
    High-level state helper that composes a `DaprStateStore` instance.

    Prefer this in application code and workflow activities when you want dicts,
    validation, retries, and convenient TTL support.
    """

    _mirror_lock = threading.Lock()

    def __init__(
        self,
        *,
        store_name: str,
        key_prefix: str = "",
        model: Optional[Type[BaseModel]] = None,
        mirror_writes: bool = False,
        local_mirror_path: Optional[str] = None,
        store_factory: Optional[Callable[[], DaprStateStore]] = None,
        retry_attempts: int = 3,
        retry_initial_backoff: float = 0.1,
        retry_backoff_multiplier: float = 2.0,
        retry_jitter: float = 0.1,
    ) -> None:
        """
        Args:
            store_name: Dapr state component name (required).
            key_prefix: Optional logical prefix applied to all keys (e.g., "blog:").
            model: Optional Pydantic model used to validate/shape payloads.
            mirror_writes: If True, also mirror successful writes to local disk.
            local_mirror_path: Directory where mirror files are written (defaults to CWD).
            store_factory: Factory returning a `DaprStateStore` (DI/testing).
            retry_attempts: Max attempts per Dapr call (>=1).
            retry_initial_backoff: Initial backoff in seconds.
            retry_backoff_multiplier: Backoff multiplier per attempt (>=1.0).
            retry_jitter: Proportional jitter [0,1] applied to backoff.
        """
        if not store_name:
            raise StateStoreError("State store name is required (store_name).")

        self.store_name = store_name
        self.key_prefix = key_prefix
        self.model = model
        self.mirror_writes = mirror_writes
        self.local_mirror_path = local_mirror_path
        self.retry_attempts = max(1, retry_attempts)
        self.retry_initial_backoff = max(0.0, retry_initial_backoff)
        self.retry_backoff_multiplier = max(1.0, retry_backoff_multiplier)
        self.retry_jitter = max(0.0, retry_jitter)

        self._store_factory = store_factory or (
            lambda: DaprStateStore(store_name=self.store_name)
        )
        self._store_cached: Optional[DaprStateStore] = None

    def _store(self) -> DaprStateStore:
        """Return the lazily-constructed `DaprStateStore`."""
        if self._store_cached is None:
            self._store_cached = self._store_factory()
        return self._store_cached

    def _qualify(self, key: str) -> str:
        """Apply the configured key prefix to a logical key."""
        return f"{self.key_prefix}{key}" if self.key_prefix else key

    def _strip_prefix(self, qualified: str) -> str:
        """Remove the configured key prefix from a qualified key (best effort)."""
        if self.key_prefix and qualified.startswith(self.key_prefix):
            return qualified[len(self.key_prefix) :]
        return qualified

    def _with_retries(self, func: Callable[[], Any]) -> Any:
        """Execute a callable with retry/backoff/jitter."""
        delay = self.retry_initial_backoff
        attempt = 0
        while True:
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt >= self.retry_attempts:
                    raise
                sleep_for = delay * (
                    1 + random.uniform(-self.retry_jitter, self.retry_jitter)
                )
                if sleep_for > 0:
                    time.sleep(max(0.0, sleep_for))
                delay *= self.retry_backoff_multiplier
                logger.debug(
                    "Retrying state operation after error: %s", exc, exc_info=True
                )

    def _model_dump(self, model: BaseModel) -> Dict[str, Any]:
        """Dump a Pydantic model to dict (v2/v1 support)."""
        if hasattr(model, "model_dump"):
            return model.model_dump()
        if hasattr(model, "dict"):
            return model.dict()
        raise StateStoreError(f"Unsupported pydantic model type: {type(model)}")

    def _ensure_dict(self, value: Any) -> Dict[str, Any]:
        """
        Coerce value into a dict.

        Accepts:
            - dict (returned as-is)
            - pydantic BaseModel (dumped)
            - JSON str (parsed to dict)
            - JSON bytes (decoded to str, parsed to dict)
        """
        if isinstance(value, BaseModel):
            return self._model_dump(value)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise StateStoreError(f"State string is not valid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise StateStoreError(f"Expected dict JSON, got {type(parsed)}")
            return parsed
        if isinstance(value, bytes):
            return self._ensure_dict(value.decode("utf-8"))
        raise StateStoreError(
            f"Unsupported state type: {type(value)}. Expected dict, BaseModel, str, or bytes."
        )

    def _validate_model(
        self, payload: Dict[str, Any], *, return_model: bool = False
    ) -> Union[Dict[str, Any], BaseModel]:
        """Validate payload with configured Pydantic model (if any)."""
        if not self.model:
            return payload
        try:
            parsed = self.model(**payload)
        except ValidationError as exc:
            raise StateStoreError(f"State validation failed: {exc.errors()}") from exc
        return parsed if return_model else self._model_dump(parsed)

    def _save_local_copy(self, *, key: str, data: Dict[str, Any]) -> None:
        """
        Write/merge a pretty-printed JSON file for the qualified key (debug/dev).

        Uses a temp file and atomic replace to avoid partial writes.
        """
        directory = self.local_mirror_path or os.getcwd()
        os.makedirs(directory, exist_ok=True)
        filename = f"{key}.json"
        file_path = os.path.join(directory, filename)

        tmp_fd, tmp_path = tempfile.mkstemp(dir=directory)
        os.close(tmp_fd)
        try:
            with self._mirror_lock:
                existing: Dict[str, Any] = {}
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as current:
                            existing = json.load(current)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Existing state file corrupt; overwriting", exc_info=True
                        )

                merged = _deep_merge(existing, data)

                with open(tmp_path, "w", encoding="utf-8") as tmp_file:
                    json.dump(merged, tmp_file, indent=2)

                os.replace(tmp_path, file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to mirror state locally", exc_info=True)
            raise StateStoreError(f"Failed to save local state mirror: {exc}") from exc
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:  # noqa: BLE001
                    pass

    def load(
        self,
        *,
        key: str,
        default: Optional[Dict[str, Any]] = None,
        state_metadata: Optional[Dict[str, str]] = None,
        return_model: bool = False,
    ) -> Union[Dict[str, Any], BaseModel]:
        """
        Load a JSON dict from the state store.

        Args:
            key: Logical (unprefixed) key.
            default: Returned if the item does not exist (must be dict or None).
            state_metadata: Optional Dapr metadata.
            return_model: If True and a Pydantic model is configured, return model instance.

        Returns:
            Dict payload or model instance.
        """
        qualified = self._qualify(key)
        logger.debug("Loading state from %s key=%s", self.store_name, qualified)

        def call() -> StateResponse:
            return self._store().get_state(
                qualified,
                state_metadata=state_metadata,
            )

        try:
            response = self._with_retries(call)
        except Exception as exc:  # noqa: BLE001
            raise StateStoreError(
                f"Failed to load state for key '{qualified}': {exc}"
            ) from exc

        if not response or not getattr(response, "data", None):
            return default.copy() if isinstance(default, dict) else (default or {})

        try:
            state_data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise StateStoreError(
                f"State for key '{qualified}' is not valid JSON: {exc}"
            ) from exc

        if not isinstance(state_data, dict):
            raise StateStoreError(
                f"State for key '{qualified}' must be a dict, got {type(state_data)}"
            )

        return self._validate_model(state_data, return_model=return_model)

    def load_with_etag(
        self,
        *,
        key: str,
        default: Optional[Dict[str, Any]] = None,
        state_metadata: Optional[Dict[str, str]] = None,
        return_model: bool = False,
    ) -> Tuple[Union[Dict[str, Any], BaseModel], Optional[str]]:
        """
        Load a JSON dict and return `(payload, etag)`.

        Args:
            key: Logical (unprefixed) key.
            default: Returned payload when not found.
            state_metadata: Optional Dapr metadata.
            return_model: If True and model configured, return model instance.

        Returns:
            (dict_or_model, etag_or_none)
        """
        qualified = self._qualify(key)
        logger.debug(
            "Loading state with etag from %s key=%s", self.store_name, qualified
        )

        def call() -> StateResponse:
            return self._store().get_state(
                qualified,
                state_metadata=state_metadata,
            )

        try:
            response = self._with_retries(call)
        except Exception as exc:  # noqa: BLE001
            raise StateStoreError(
                f"Failed to load state for key '{qualified}': {exc}"
            ) from exc

        if not response or not getattr(response, "data", None):
            data = default.copy() if isinstance(default, dict) else (default or {})
            return data, None

        try:
            state_data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise StateStoreError(
                f"State for key '{qualified}' is not valid JSON: {exc}"
            ) from exc

        if not isinstance(state_data, dict):
            raise StateStoreError(
                f"State for key '{qualified}' must be a dict, got {type(state_data)}"
            )

        payload = self._validate_model(state_data, return_model=return_model)
        etag = getattr(response, "etag", None)
        return payload, etag

    def load_many(
        self,
        keys: Sequence[str],
        *,
        parallelism: int = 1,
        state_metadata: Optional[Dict[str, str]] = None,
        return_model: bool = False,
    ) -> Dict[str, Union[Dict[str, Any], BaseModel]]:
        """
        Bulk load multiple keys.

        Args:
            keys: Logical (unprefixed) keys.
            parallelism: Backend-specific parallelism.
            state_metadata: Optional Dapr metadata.
            return_model: If True and model configured, return model instances.

        Returns:
            Mapping of logical key -> dict/model for keys that existed.
        """
        qualified_keys = [self._qualify(k) for k in keys]
        logger.debug(
            "Loading bulk state from %s keys=%s", self.store_name, qualified_keys
        )

        def call() -> Sequence[BulkStateItem]:
            return self._store().get_bulk_state(
                qualified_keys,
                parallelism=parallelism,
                states_metadata=state_metadata,
            )

        try:
            items = self._with_retries(call)
        except Exception as exc:  # noqa: BLE001
            raise StateStoreError(f"Failed to bulk load state: {exc}") from exc

        results: Dict[str, Union[Dict[str, Any], BaseModel]] = {}
        for item in items or []:
            data_raw = item.data
            if not data_raw:
                continue
            if isinstance(data_raw, bytes):
                data_raw = data_raw.decode("utf-8")
            try:
                parsed = json.loads(data_raw)
            except Exception as exc:  # noqa: BLE001
                raise StateStoreError(
                    f"State for key '{item.key}' is not valid JSON: {exc}"
                ) from exc
            logical_key = self._strip_prefix(item.key)
            results[logical_key] = self._validate_model(
                parsed, return_model=return_model
            )
        return results

    def save(
        self,
        *,
        key: str,
        value: Any,
        etag: Optional[str] = None,
        state_metadata: Optional[Dict[str, str]] = None,
        state_options: Optional[Dict[str, Any]] = None,
        ttl_in_seconds: Optional[int] = None,
    ) -> None:
        """
        Save a JSON payload under a logical key.

        Args:
            key: Logical (unprefixed) key.
            value: dict | BaseModel | JSON str | JSON bytes.
            etag: Optional ETag for optimistic concurrency.
            state_metadata: Optional Dapr metadata.
            state_options: Dict of `StateOptions` fields (or a `StateOptions` instance).
            ttl_in_seconds: Optional TTL; backend must support TTL via metadata.
        """
        qualified = self._qualify(key)
        payload_dict = self._ensure_dict(value)
        payload_str = json.dumps(payload_dict)

        metadata = dict(state_metadata or {})
        if ttl_in_seconds is not None:
            metadata.setdefault("ttlInSeconds", str(ttl_in_seconds))

        logger.debug(
            "Saving state to %s key=%s etag=%s ttl=%s",
            self.store_name,
            qualified,
            etag,
            ttl_in_seconds,
        )

        def call() -> None:
            self._store().save_state(
                qualified,
                payload_str,
                state_metadata=metadata or None,
                etag=etag,
                state_options=_coerce_state_options(state_options),
            )

        try:
            self._with_retries(call)
        except Exception as exc:  # noqa: BLE001
            raise StateStoreError(
                f"Failed to save state for key '{qualified}': {exc}"
            ) from exc

        if self.mirror_writes:
            self._save_local_copy(key=qualified, data=payload_dict)

    def delete(
        self,
        *,
        key: str,
        etag: Optional[str] = None,
        state_metadata: Optional[Dict[str, str]] = None,
        state_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Delete a logical key.

        Args:
            key: Logical (unprefixed) key.
            etag: Optional ETag for concurrency.
            state_metadata: Optional Dapr metadata.
            state_options: Dict or `StateOptions` controlling delete behavior.
        """
        qualified = self._qualify(key)
        logger.debug(
            "Deleting state from %s key=%s etag=%s", self.store_name, qualified, etag
        )

        def call() -> None:
            self._store().delete_state(
                qualified,
                etag=etag,
                state_metadata=state_metadata,
                state_options=_coerce_state_options(state_options),
            )

        try:
            self._with_retries(call)
        except Exception as exc:  # noqa: BLE001
            raise StateStoreError(
                f"Failed to delete state for key '{qualified}': {exc}"
            ) from exc

    def exists(self, *, key: str) -> bool:
        """
        Return True if the key exists (uses presence of an ETag as heuristic).
        """
        _, etag = self.load_with_etag(key=key, default=None)
        return etag is not None

    def save_many(
        self,
        items: Dict[str, Any],
        *,
        state_metadata: Optional[Dict[str, str]] = None,
        state_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save multiple logical keys (non-atomic sequence of individual saves).

        Args:
            items: Mapping key -> JSON-serializable payload (same accepted types as `save()`).
            state_metadata: Optional metadata applied to each save.
            state_options: Dict or `StateOptions` applied to each save.
        """
        logger.debug(
            "Saving bulk state to %s keys=%s", self.store_name, list(items.keys())
        )
        metadata = state_metadata or {}
        options = _coerce_state_options(state_options)

        def call() -> None:
            store = self._store()
            for key, value in items.items():
                payload_dict = self._ensure_dict(value)
                payload_str = json.dumps(payload_dict)
                store.save_state(
                    self._qualify(key),
                    payload_str,
                    state_metadata=metadata,
                    state_options=options,
                )
                if self.mirror_writes:
                    self._save_local_copy(key=self._qualify(key), data=payload_dict)

        try:
            self._with_retries(call)
        except Exception as exc:  # noqa: BLE001
            raise StateStoreError(f"Failed to bulk save state: {exc}") from exc

    def execute_transaction(
        self,
        operations: Sequence[Dict[str, Any]],
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Execute a transactional batch (backend must support transactions).

        Args:
            operations: Dapr transaction operations (upserts/deletes).
            metadata: Optional request metadata.
        """
        logger.debug(
            "Executing state transaction on %s operations=%s",
            self.store_name,
            operations,
        )

        def call() -> None:
            self._store().execute_state_transaction(operations, metadata=metadata)

        try:
            self._with_retries(call)
        except Exception as exc:  # noqa: BLE001
            raise StateStoreError(
                f"Failed to execute state transaction: {exc}"
            ) from exc


def _deep_merge(original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries (values in `updates` override `original`).
    """
    result = dict(original)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ----------------------------------------------------------------------
# Optional convenience helpers (explicit service; no env lookups)
# ----------------------------------------------------------------------


def load_state_dict(
    service: StateStoreService,
    key: str,
    *,
    default: Optional[Dict[str, Any]] = None,
    state_metadata: Optional[Dict[str, str]] = None,
    return_model: bool = False,
) -> Union[Dict[str, Any], BaseModel]:
    """
    Convenience wrapper for `service.load(...)` using an injected StateStoreService.
    """
    return service.load(
        key=key,
        default=default,
        state_metadata=state_metadata,
        return_model=return_model,
    )


def load_state_with_etag(
    service: StateStoreService,
    key: str,
    *,
    default: Optional[Dict[str, Any]] = None,
    state_metadata: Optional[Dict[str, str]] = None,
    return_model: bool = False,
) -> Tuple[Union[Dict[str, Any], BaseModel], Optional[str]]:
    """
    Convenience wrapper for `service.load_with_etag(...)`.
    """
    return service.load_with_etag(
        key=key,
        default=default,
        state_metadata=state_metadata,
        return_model=return_model,
    )


def load_state_many(
    service: StateStoreService,
    keys: Sequence[str],
    *,
    parallelism: int = 1,
    state_metadata: Optional[Dict[str, str]] = None,
    return_model: bool = False,
) -> Dict[str, Union[Dict[str, Any], BaseModel]]:
    """
    Convenience wrapper for `service.load_many(...)`.
    """
    return service.load_many(
        keys,
        parallelism=parallelism,
        state_metadata=state_metadata,
        return_model=return_model,
    )


def save_state_dict(
    service: StateStoreService,
    key: str,
    value: Any,
    *,
    etag: Optional[str] = None,
    state_metadata: Optional[Dict[str, str]] = None,
    state_options: Optional[Dict[str, Any]] = None,
    ttl_in_seconds: Optional[int] = None,
) -> None:
    """
    Convenience wrapper for `service.save(...)`.
    """
    service.save(
        key=key,
        value=value,
        etag=etag,
        state_metadata=state_metadata,
        state_options=state_options,
        ttl_in_seconds=ttl_in_seconds,
    )


def save_state_many(
    service: StateStoreService,
    items: Dict[str, Any],
    *,
    state_metadata: Optional[Dict[str, str]] = None,
    state_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience wrapper for `service.save_many(...)`.
    """
    service.save_many(
        items,
        state_metadata=state_metadata,
        state_options=state_options,
    )


def delete_state(
    service: StateStoreService,
    key: str,
    *,
    etag: Optional[str] = None,
    state_metadata: Optional[Dict[str, str]] = None,
    state_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience wrapper for `service.delete(...)`.
    """
    service.delete(
        key=key,
        etag=etag,
        state_metadata=state_metadata,
        state_options=state_options,
    )


def state_exists(
    service: StateStoreService,
    key: str,
) -> bool:
    """
    Convenience wrapper for `service.exists(...)`.
    """
    return service.exists(key=key)


def execute_state_transaction(
    service: StateStoreService,
    operations: Sequence[Dict[str, Any]],
    *,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Convenience wrapper for `service.execute_transaction(...)`.
    """
    service.execute_transaction(operations, metadata=metadata)


__all__ = [
    "StateStoreError",
    "StateStoreService",
    "DaprStateStore",
    "_coerce_state_options",
    "load_state_dict",
    "load_state_with_etag",
    "load_state_many",
    "save_state_dict",
    "save_state_many",
    "delete_state",
    "state_exists",
    "execute_state_transaction",
]
