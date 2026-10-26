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

"""Extension-local fixtures without global Dapr SDK patching."""

from __future__ import annotations

import json
from copy import deepcopy
from importlib.resources import files
from typing import cast

import pytest
from drasi_agent_router_contracts import ListQueriesResponse, parse
from drasi_agent_router_contracts.models.Operation import Operation

from dapr_agents.ext.drasi._models import (
    IntentDocument,
    SubscriptionIntent,
    SubscriptionScope,
    SubscriptionStatus,
)

_CURRENT_INCARNATION = "incarnation-service-errors"


def _published_messages() -> dict[str, dict[str, object]]:
    resource = files("drasi_agent_router_contracts").joinpath(
        "fixtures", "messages.json"
    )
    entries = cast(
        list[dict[str, object]], json.loads(resource.read_text(encoding="utf-8"))
    )
    result: dict[str, dict[str, object]] = {}
    for entry in entries:
        if entry.get("valid") is True and isinstance(entry.get("message"), dict):
            result[cast(str, entry["name"])] = cast(dict[str, object], entry["message"])
    return result


@pytest.fixture
def published_messages() -> dict[str, dict[str, object]]:
    return deepcopy(_published_messages())


@pytest.fixture
def scope() -> SubscriptionScope:
    return SubscriptionScope(
        router_id="drasi-system/sre-router-reaction",
        namespace="applications",
        app_id="checkout-sre",
        agent_name="CheckoutSRE",
    )


@pytest.fixture
def catalog(
    published_messages: dict[str, dict[str, object]],
) -> ListQueriesResponse:
    return parse(ListQueriesResponse, published_messages["catalog"])


@pytest.fixture
def empty_catalog(scope: SubscriptionScope) -> ListQueriesResponse:
    return parse(
        ListQueriesResponse,
        {
            "protocol_version": 1,
            "router_id": scope.router_id,
            "queries": [],
        },
    )


@pytest.fixture
def active_intent(catalog: ListQueriesResponse) -> SubscriptionIntent:
    return SubscriptionIntent(
        query_id="service-errors",
        operations=(Operation.i, Operation.u),
        instructions="Investigate newly matching service errors.",
        catalog_snapshot=catalog,
        incarnation=_CURRENT_INCARNATION,
        status="active",
    )


@pytest.fixture
def status_intents(
    catalog: ListQueriesResponse,
) -> dict[SubscriptionStatus, SubscriptionIntent]:
    statuses: tuple[SubscriptionStatus, ...] = (
        "pending_subscribe",
        "pending_update",
        "active",
        "pending_unsubscribe",
        "unavailable",
    )
    return {
        status: SubscriptionIntent(
            query_id="service-errors",
            operations=(Operation.i,),
            instructions=f"Handle service errors in {status} state.",
            catalog_snapshot=catalog,
            incarnation=_CURRENT_INCARNATION,
            status=status,
        )
        for status in statuses
    }


@pytest.fixture
def intent_document(
    scope: SubscriptionScope,
    catalog: ListQueriesResponse,
    active_intent: SubscriptionIntent,
) -> IntentDocument:
    rollout_intent = SubscriptionIntent(
        query_id="rollout-status",
        operations=(Operation.i, Operation.u, Operation.d),
        instructions="Track rollout status changes.",
        catalog_snapshot=catalog,
        incarnation="incarnation-rollout-status",
        status="pending_update",
    )
    return IntentDocument(
        format_version=1,
        scope=scope,
        intents={
            active_intent.query_id: active_intent,
            rollout_intent.query_id: rollout_intent,
        },
    )
