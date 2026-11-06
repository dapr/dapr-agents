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

"""Transport-independent admission of agent-managed Drasi deliveries."""

from __future__ import annotations

import logging

from drasi_agent_router_contracts import AgentDelivery, parse
from drasi_agent_router_contracts.models.Operation import Operation
from jsonschema.exceptions import ValidationError as WireValidationError
from pydantic import ConfigDict, JsonValue, TypeAdapter

from ._interfaces import AdmissionHandler, IntentReader, IntentStoreError
from ._models import AdmissionResult, Discard, Poison, Retry, SubscriptionScope
from .task_builder import build_event_task, render_event_data

logger = logging.getLogger(__name__)

_JSON_OBJECT = TypeAdapter(
    dict[str, JsonValue], config=ConfigDict(strict=True, allow_inf_nan=False)
)


class DrasiAdmissionHandler(AdmissionHandler):
    """Read one scoped intent snapshot without scheduling or acknowledging work."""

    def __init__(self, *, scope: SubscriptionScope, intents: IntentReader) -> None:
        if intents.scope != scope:
            logger.error(
                "Drasi admission configuration failed: intent_reader_scope_mismatch."
            )
            raise ValueError("Drasi admission and intent reader scopes must match.")
        self._scope = scope
        self._intents = intents

    def admit(self, data: object) -> AdmissionResult:
        """Classify decoded M2 CloudEvent data; intent reads may block."""
        try:
            # Validate before to_wire can coerce non-JSON row values, such as NaN.
            document = _JSON_OBJECT.validate_python(data)
            delivery = parse(AgentDelivery, document)
            # Encoding failures belong to the delivery, not the intent store.
            render_event_data(delivery)
        except (WireValidationError, ValueError):
            logger.warning("Drasi event admission rejected: invalid_delivery.")
            return Poison(reason="invalid_delivery")

        if delivery.routerId != self._scope.router_id:
            logger.warning("Drasi event admission rejected: router_mismatch.")
            return Poison(reason="router_mismatch")

        query_id = delivery.event.payload.source.queryId
        try:
            intent = self._intents.get(query_id)
        except IntentStoreError as error:
            if error.category == "corrupt":
                result = Retry(reason="state_corrupt")
            elif error.category == "unsupported_version":
                result = Retry(reason="unsupported_state_version")
            else:
                result = Retry(reason="state_unavailable")
            logger.warning(
                "Drasi event admission deferred: intent_store_%s.", error.category
            )
            return result

        if intent is None:
            return Discard(reason="no_intent")
        if (
            intent.query_id != query_id
            or intent.catalog_snapshot.router_id != self._scope.router_id
        ):
            logger.error("Drasi event admission deferred: state_corrupt.")
            return Retry(reason="state_corrupt")
        if delivery.subscriptionIncarnation != intent.incarnation:
            return Discard(reason="stale_incarnation")

        match intent.status:
            case "unavailable":
                return Discard(reason="unavailable")
            case "pending_unsubscribe":
                return Discard(reason="pending_unsubscribe")
            case "pending_subscribe" | "pending_update":
                return Retry(reason="pending_subscription")
            case "active":
                if Operation(delivery.event.op) not in intent.operations:
                    return Discard(reason="operation_excluded")
            case _:
                logger.error("Drasi event admission deferred: state_corrupt.")
                return Retry(reason="state_corrupt")

        try:
            return build_event_task(scope=self._scope, intent=intent, delivery=delivery)
        except (WireValidationError, ValueError):
            logger.error("Drasi event admission deferred: state_corrupt.")
            return Retry(reason="state_corrupt")
