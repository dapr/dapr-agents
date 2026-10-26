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

"""Private durable intent adapter over the agent's configured state store."""

from __future__ import annotations

import json
import logging
from typing import Literal, NoReturn

from dapr.clients.exceptions import DaprGrpcError, DaprInternalError
from dapr.clients.grpc._state import Concurrency, Consistency, StateOptions
from dapr.clients.retry import RetryPolicy
from dapr.proto import api_v1
from grpc import RpcError, StatusCode
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from dapr_agents.storage.daprstores.stateservice import StateStoreService

from ._interfaces import IntentRepository, IntentStoreError
from ._models import (
    IntentDocument,
    IntentSnapshot,
    SubscriptionIntent,
    SubscriptionScope,
)

logger = logging.getLogger(__name__)


class DaprIntentRepository(IntentRepository):
    """Persist one scoped intent document without owning the supplied store.

    Pass the agent's ``AgentStateConfig.store`` / ``agent.state_store``.
    Preparation owns initialization; callers reload and merge after conflicts.
    Every read goes to storage, including reads through ``IntentReader``.
    """

    def __init__(self, *, scope: SubscriptionScope, store: StateStoreService) -> None:
        self._scope = scope
        self._service = store
        self._key = store._qualify(f"drasi:intent:{scope.inbox_topic}")
        self._metadata = {
            "contentType": "application/json",
            "partitionKey": self._key,
        }

    @property
    def scope(self) -> SubscriptionScope:
        return self._scope

    def load(self) -> IntentSnapshot | None:
        try:
            store = self._service._store()
            with store._build_client() as client:
                # The SDK's get_state wrapper does not expose read consistency.
                # Keep its configured channel and read retry policy.
                request = api_v1.GetStateRequest(
                    store_name=store.store_name,
                    key=self._key,
                    metadata=self._metadata.copy(),
                    consistency=Consistency.strong.value,
                )
                response, _ = client.retry_policy.run_rpc(
                    client._stub.GetState.with_call, request, metadata=None
                )
        except (RpcError, DaprInternalError, OSError):
            self._fail("load", "unavailable")

        # The SDK represents not-found with empty bytes and etag="", not None.
        if not response.data:
            if response.etag:
                self._fail("load", "corrupt")
            return None
        if not isinstance(response.etag, str) or not response.etag:
            self._fail("load", "unavailable")

        return IntentSnapshot(
            document=self._decode(response.data, operation="load"),
            etag=response.etag,
        )

    def get(self, query_id: str) -> SubscriptionIntent | None:
        snapshot = self.load()
        return None if snapshot is None else snapshot.document.intents.get(query_id)

    def initialize(self, document: IntentDocument) -> None:
        self._write(document, expected_etag=None, operation="initialize")

    def save(self, document: IntentDocument, *, expected_etag: str) -> None:
        if not isinstance(expected_etag, str) or not expected_etag:
            logger.error("Drasi intent save requires a nonempty expected_etag.")
            raise ValueError("expected_etag must be nonempty.")
        self._write(document, expected_etag=expected_etag, operation="save")

    def _decode(self, data: bytes | str, *, operation: str) -> IntentDocument:
        try:
            payload = json.loads(data)
        except (ValueError, RecursionError):
            self._fail(operation, "corrupt")

        if not isinstance(payload, dict):
            self._fail(operation, "corrupt")
        version = payload.get("format_version")
        if type(version) is not int:
            self._fail(operation, "corrupt")
        if version != 1:
            self._fail(operation, "unsupported_version")

        try:
            document = IntentDocument.model_validate(payload)
        except ValidationError:
            self._fail(operation, "corrupt")
        if document.scope != self._scope:
            self._fail(operation, "corrupt")
        return document

    def _write(
        self,
        document: IntentDocument,
        *,
        expected_etag: str | None,
        operation: Literal["initialize", "save"],
    ) -> None:
        if document.scope != self._scope:
            logger.error("Drasi intent document scope does not match repository scope.")
            raise ValueError("Intent document scope does not match repository scope.")
        try:
            value = document.model_dump_json(warnings="error")
        except PydanticSerializationError:
            self._fail(operation, "corrupt")
        # Nested mappings/models can be mutated without assignment validation.
        self._decode(value, operation=operation)

        try:
            store = self._service._store()
            # The raw wrapper has no per-call SDK retry override.
            with store._build_client() as client:
                client.retry_policy = RetryPolicy(max_attempts=0)
                client.save_state(
                    store_name=store.store_name,
                    key=self._key,
                    value=value,
                    etag=expected_etag,
                    state_metadata=self._metadata.copy(),
                    options=StateOptions(
                        concurrency=Concurrency.first_write,
                        consistency=Consistency.strong,
                    ),
                )
        except DaprGrpcError as error:
            if expected_etag is not None and error.code() == StatusCode.ABORTED:
                self._fail(operation, "conflict")
            self._fail(operation, "unavailable")
        except (RpcError, DaprInternalError, OSError):
            self._fail(operation, "unavailable")

    def _fail(
        self,
        operation: str,
        category: Literal["unavailable", "corrupt", "unsupported_version", "conflict"],
    ) -> NoReturn:
        logger.error(
            "Drasi intent %s failed (%s), scope=%s.",
            operation,
            category,
            self._scope.inbox_topic,
        )
        # Validation and transport exceptions can contain instructions or secrets.
        raise IntentStoreError(category) from None
