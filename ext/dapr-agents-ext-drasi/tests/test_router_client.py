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

"""Exercise the real MCP client/session against a scripted HTTP endpoint."""

from __future__ import annotations

import asyncio
import json
import threading
import traceback
import warnings
from collections import defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

import httpx
import pytest
from drasi_agent_router_contracts import (
    ListQueriesResponse,
    SubscribeRequest,
    UnsubscribeRequest,
    parse,
    to_wire,
)
from httpx import AsyncClient

from dapr_agents.ext.drasi import router_client
from dapr_agents.ext.drasi._interfaces import RouterClient, RouterError
from dapr_agents.ext.drasi._models import (
    ResolvedDrasiConfig,
    RouterFailureCategory,
    SubscriptionScope,
)
from dapr_agents.ext.drasi.router_client import MCPRouterClient


def _success(document: object, representation: str = "both") -> dict[str, Any]:
    result: dict[str, Any] = {"isError": False, "content": []}
    if representation != "text":
        result["structuredContent"] = deepcopy(document)
    if representation != "structured":
        result["content"] = [{"type": "text", "text": json.dumps(document)}]
    return result


def _error(code: str) -> dict[str, Any]:
    return {
        "isError": True,
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {"code": code, "message": "private-router-response"}
                ),
            }
        ],
    }


def _subscribe_response(scope: SubscriptionScope, **changes: Any) -> dict[str, Any]:
    return {
        "query_id": "service-errors",
        "operations": ["u", "i"],
        "subscription_incarnation": "incarnation-service-errors",
        "topic_name": scope.inbox_topic,
        "status": "created",
        **changes,
    }


class _TrackedClient(AsyncClient):
    async def __aenter__(self) -> _TrackedClient:
        self.owner = (
            threading.current_thread(),
            asyncio.get_running_loop(),
            asyncio.current_task(),
        )
        await super().__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        assert self.owner == (
            threading.current_thread(),
            asyncio.get_running_loop(),
            asyncio.current_task(),
        )
        await super().__aexit__(*args)


class _Server:
    def __init__(self, catalog: ListQueriesResponse) -> None:
        self.catalog = to_wire(catalog)
        self.results: dict[str, deque[dict[str, Any] | Exception]] = defaultdict(deque)
        self.requests: list[httpx.Request] = []
        self.messages: list[dict[str, Any]] = []
        self.clients: list[_TrackedClient] = []
        self.initialization_error: Exception | int | None = None
        self.protocol_version: str | None = None
        self.rpc_error: dict[str, Any] | None = None
        self.required_token: str | None = None
        self.stall: str | None = None
        self.started = threading.Event()
        self.release = threading.Event()

    def client(self, **kwargs: Any) -> _TrackedClient:
        client = _TrackedClient(transport=httpx.MockTransport(self.handle), **kwargs)
        self.clients.append(client)
        return client

    @property
    def calls(self) -> list[dict[str, Any]]:
        return [
            message["params"]
            for message in self.messages
            if message["method"] == "tools/call"
        ]

    def queue(self, tool: str, result: dict[str, Any] | Exception) -> None:
        self.results[tool].append(deepcopy(result))

    async def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.headers["accept"] == "application/json, text/event-stream"
        self.requests.append(request)
        if (
            self.required_token is not None
            and request.headers.get("dapr-api-token") != self.required_token
        ):
            return httpx.Response(401)
        message = json.loads(request.content)
        self.messages.append(message)
        method = message["method"]

        if method == "initialize":
            if isinstance(self.initialization_error, Exception):
                raise self.initialization_error
            if isinstance(self.initialization_error, int):
                return httpx.Response(self.initialization_error)
            result = {
                "protocolVersion": self.protocol_version
                or message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "DaprAgentRouter-test", "version": "1"},
            }
        elif method == "notifications/initialized":
            return httpx.Response(202)
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": name,
                        "inputSchema": {"type": "object"},
                        "outputSchema": {"type": "object"},
                    }
                    for name in ("list_queries", "subscribe", "unsubscribe")
                ]
            }
        else:
            assert method == "tools/call"
            tool = message["params"]["name"]
            if self.rpc_error is not None:
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": self.rpc_error,
                    },
                )
            if self.stall == tool:
                self.started.set()
                while not self.release.is_set():
                    await asyncio.sleep(0.005)
            if self.results[tool]:
                queued = self.results[tool].popleft()
                if isinstance(queued, Exception):
                    raise queued
                result = queued
            else:
                assert tool == "list_queries", f"Unscripted mutation: {tool}"
                result = _success(self.catalog)

        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": message["id"], "result": result},
        )


@pytest.fixture
def config(scope: SubscriptionScope) -> ResolvedDrasiConfig:
    return ResolvedDrasiConfig(
        scope=scope,
        pubsub_name="application-bus",
        state_store_name="agent-runtime",
        workflow_name="CheckoutSRE.agent_workflow",
        dapr_http_port=3517,
    )


@pytest.fixture
def server(catalog: ListQueriesResponse, monkeypatch: pytest.MonkeyPatch) -> _Server:
    server = _Server(catalog)
    monkeypatch.setattr(router_client.httpx, "AsyncClient", server.client)
    return server


@pytest.fixture
def client(config: ResolvedDrasiConfig, server: _Server) -> Iterator[MCPRouterClient]:
    client = MCPRouterClient(config, timeout_seconds=2)
    try:
        yield client
    finally:
        client.close()
        assert all(http.is_closed for http in server.clients)
        assert all(not http.owner[0].is_alive() for http in server.clients)


@pytest.fixture
def subscription(scope: SubscriptionScope) -> SubscribeRequest:
    return parse(
        SubscribeRequest,
        {
            "query_id": "service-errors",
            "operations": ["i", "u"],
            "subscriber": to_wire(scope.subscriber),
            "subscription_incarnation": "incarnation-service-errors",
        },
    )


@pytest.fixture
def removal(subscription: SubscribeRequest) -> UnsubscribeRequest:
    document = to_wire(subscription)
    del document["operations"]
    return parse(UnsubscribeRequest, document)


@pytest.mark.parametrize("token", [None, "test-only-sidecar-token"])
def test_complete_mcp_flow_uses_configured_sidecar_and_frozen_interface(
    client: MCPRouterClient,
    server: _Server,
    config: ResolvedDrasiConfig,
    subscription: SubscribeRequest,
    removal: UnsubscribeRequest,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    token: str | None,
) -> None:
    monkeypatch.setattr(router_client.settings, "DAPR_API_TOKEN", token)
    server.required_token = token
    router: RouterClient = client
    catalog = router.list_queries()
    server.queue("subscribe", _success(_subscribe_response(config.scope)))
    server.queue(
        "unsubscribe",
        _success({"query_id": removal.query_id, "removed": True}),
    )

    assert router.scope == config.scope
    assert catalog.router_id == config.scope.router_id
    assert router.subscribe(subscription).status.value == "created"
    assert router.unsubscribe(removal).removed is True
    assert server.calls == [
        {"name": "list_queries", "arguments": {}},
        {"name": "subscribe", "arguments": to_wire(subscription)},
        {"name": "unsubscribe", "arguments": to_wire(removal)},
    ]
    assert all(str(request.url) == config.router_mcp_url for request in server.requests)
    assert all(
        request.headers.get("dapr-api-token") == token for request in server.requests
    )
    assert all("instructions" not in call["arguments"] for call in server.calls)
    assert [message["method"] for message in server.messages].count("initialize") == 3
    assert [message["method"] for message in server.messages].count(
        "notifications/initialized"
    ) == 3
    assert all(http.is_closed for http in server.clients)
    if token is not None:
        assert token not in caplog.text
        assert all(token not in str(request.url) for request in server.requests)


def test_invalid_sidecar_token_is_a_transport_error_without_token_disclosure(
    client: MCPRouterClient,
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = "test-only-invalid-sidecar-token"
    monkeypatch.setattr(router_client.settings, "DAPR_API_TOKEN", token)
    server.required_token = "test-only-expected-sidecar-token"
    with pytest.raises(RouterError) as failure:
        client.list_queries()
    assert failure.value.category == "transport"
    assert server.calls == []
    assert token not in caplog.text
    assert token not in "".join(traceback.format_exception(failure.value))


def test_loopback_requests_do_not_use_environment_proxies(
    client: MCPRouterClient,
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(variable, "http://proxy.invalid:8080")
    monkeypatch.setenv("NO_PROXY", "")
    assert client.list_queries().queries
    assert all(http.trust_env is False for http in server.clients)


@pytest.mark.parametrize("empty", [False, True])
def test_catalog_is_fetched_once_and_returns_detached_snapshots(
    client: MCPRouterClient, server: _Server, empty: bool
) -> None:
    if empty:
        server.catalog["queries"] = []
    expected = deepcopy(server.catalog)
    first = client.list_queries()
    if first.queries:
        first.queries[0].title = "caller mutation"
    first.queries.clear()
    server.catalog["queries"] = []

    assert to_wire(client.list_queries()) == expected
    assert len(server.calls) == 1
    if not empty:
        assert "usage" not in to_wire(client.list_queries())["queries"][1]


@pytest.mark.parametrize("representation", ["structured", "text", "both"])
def test_all_operations_accept_documented_result_representations(
    client: MCPRouterClient,
    server: _Server,
    subscription: SubscribeRequest,
    removal: UnsubscribeRequest,
    representation: str,
) -> None:
    server.queue("list_queries", _success(server.catalog, representation))
    server.queue(
        "subscribe",
        _success(_subscribe_response(client.scope), representation),
    )
    server.queue(
        "unsubscribe",
        _success({"query_id": removal.query_id, "removed": False}, representation),
    )
    assert client.list_queries().queries
    assert client.subscribe(subscription).query_id == subscription.query_id
    assert client.unsubscribe(removal).removed is False


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol_version": 2},
        {"protocol_version": "1"},
        {"protocol_version": True},
        {"router_id": "other/router"},
        {"capabilities": []},
        {"queries": None},
        {
            "queries": [
                {"query_id": "q", "title": "Q", "description": "Q", "usage": None}
            ]
        },
        {"queries": [{"query_id": "q", "title": "", "description": "Q"}]},
        {
            "queries": [
                {"query_id": "q", "title": "Q", "description": "Q"},
                {"query_id": "q", "title": "Another", "description": "Another"},
            ]
        },
    ],
)
def test_invalid_catalogs_fail_preparation_without_poisoning_cache(
    client: MCPRouterClient, server: _Server, changes: dict[str, Any]
) -> None:
    server.queue("list_queries", _success({**server.catalog, **changes}))
    with pytest.raises(RouterError) as failure:
        client.list_queries()
    assert failure.value.category == "invalid_response"
    assert to_wire(client.list_queries()) == server.catalog
    assert len(server.calls) == 2


def test_mutations_cannot_precede_router_identity_validation(
    client: MCPRouterClient, server: _Server, subscription: SubscribeRequest
) -> None:
    server.catalog["router_id"] = "wrong/router"
    with pytest.raises(RouterError) as failure:
        client.subscribe(subscription)
    assert failure.value.category == "invalid_response"
    assert [call["name"] for call in server.calls] == ["list_queries"]


@pytest.mark.parametrize("field", ["namespace", "app_id", "agent_name"])
@pytest.mark.parametrize("operation", ["subscribe", "unsubscribe"])
def test_requests_for_other_subscribers_are_rejected_before_network_io(
    client: MCPRouterClient,
    server: _Server,
    subscription: SubscribeRequest,
    removal: UnsubscribeRequest,
    field: str,
    operation: str,
) -> None:
    request = subscription if operation == "subscribe" else removal
    request.subscriber = request.subscriber.model_copy(update={field: "other"})
    with pytest.raises(RouterError) as failure:
        if operation == "subscribe":
            client.subscribe(subscription)
        else:
            client.unsubscribe(removal)
    assert failure.value.category == "invalid_arguments"
    assert failure.value.mutation_outcome == "rejected"
    assert server.requests == []


@pytest.mark.parametrize(
    "changes",
    [
        {"operations": []},
        {"operations": ["i", "i"]},
        {"operations": ["invalid"]},
        {"subscription_incarnation": ""},
        {"query_id": ""},
    ],
)
def test_mutated_models_still_pass_through_shared_request_validation(
    client: MCPRouterClient,
    server: _Server,
    subscription: SubscribeRequest,
    changes: dict[str, Any],
) -> None:
    with pytest.raises(RouterError) as failure:
        client.subscribe(subscription.model_copy(update=changes))
    assert failure.value.category == "invalid_arguments"
    assert server.requests == []


def test_invalid_models_do_not_emit_input_values_in_serializer_warnings(
    client: MCPRouterClient,
    server: _Server,
    subscription: SubscribeRequest,
    caplog: pytest.LogCaptureFixture,
) -> None:
    malformed = subscription.model_copy(update={"operations": ["private-client-input"]})
    with warnings.catch_warnings(record=True) as emitted:
        with pytest.raises(RouterError) as failure:
            client.subscribe(malformed)
    assert failure.value.category == "invalid_arguments"
    assert emitted == []
    assert "private-client-input" not in caplog.text
    assert "private-client-input" not in "".join(
        traceback.format_exception(failure.value)
    )
    assert server.requests == []


@pytest.mark.parametrize("status", ["created", "updated"])
def test_subscribe_accepts_both_statuses_and_operation_set_order(
    client: MCPRouterClient,
    server: _Server,
    subscription: SubscribeRequest,
    status: str,
) -> None:
    server.queue(
        "subscribe", _success(_subscribe_response(client.scope, status=status))
    )
    result = client.subscribe(subscription)
    assert result.status.value == status
    assert set(result.operations) == set(subscription.operations)
    assert [operation.value for operation in result.operations] == ["u", "i"]


@pytest.mark.parametrize(
    "changes",
    [
        {"query_id": "other-query"},
        {"subscription_incarnation": "different-lifecycle"},
        {"topic_name": "another-topic"},
        {"operations": ["d"]},
        {"operations": ["i", "i"]},
        {"operations": []},
        {"status": "failed"},
        {"extra": "private-router-response"},
    ],
)
def test_invalid_subscribe_confirmations_are_uncertain(
    client: MCPRouterClient,
    server: _Server,
    subscription: SubscribeRequest,
    changes: dict[str, Any],
) -> None:
    server.queue("subscribe", _success(_subscribe_response(client.scope, **changes)))
    with pytest.raises(RouterError) as failure:
        client.subscribe(subscription)
    assert failure.value.category == "invalid_response"
    assert failure.value.mutation_outcome == "uncertain"


@pytest.mark.parametrize("removed", [False, True])
def test_retired_queries_can_be_unsubscribed(
    client: MCPRouterClient,
    server: _Server,
    removal: UnsubscribeRequest,
    removed: bool,
) -> None:
    server.catalog["queries"] = []
    server.queue(
        "unsubscribe", _success({"query_id": removal.query_id, "removed": removed})
    )
    assert client.unsubscribe(removal).removed is removed


@pytest.mark.parametrize(
    "document",
    [
        {"query_id": "other-query", "removed": True},
        {"query_id": "service-errors", "removed": "false"},
        {"query_id": "service-errors", "removed": 0},
        {"query_id": "service-errors"},
    ],
)
def test_invalid_unsubscribe_confirmations_are_uncertain(
    client: MCPRouterClient,
    server: _Server,
    removal: UnsubscribeRequest,
    document: dict[str, Any],
) -> None:
    server.queue("unsubscribe", _success(document))
    with pytest.raises(RouterError) as failure:
        client.unsubscribe(removal)
    assert failure.value.category == "invalid_response"
    assert failure.value.mutation_outcome == "uncertain"


@pytest.mark.parametrize("operation", ["list_queries", "subscribe", "unsubscribe"])
@pytest.mark.parametrize(
    "category",
    ["invalid_arguments", "unknown_query", "incarnation_conflict", "state_unavailable"],
)
def test_domain_errors_preserve_categories_and_do_not_leak_payloads(
    client: MCPRouterClient,
    server: _Server,
    subscription: SubscribeRequest,
    removal: UnsubscribeRequest,
    caplog: pytest.LogCaptureFixture,
    operation: str,
    category: RouterFailureCategory,
) -> None:
    if operation != "list_queries":
        client.list_queries()
    server.queue(operation, _error(category))
    with pytest.raises(RouterError) as failure:
        if operation == "subscribe":
            client.subscribe(subscription)
        elif operation == "unsubscribe":
            client.unsubscribe(removal)
        else:
            client.list_queries()
    assert failure.value.category == category
    assert failure.value.mutation_outcome == (
        "uncertain" if category == "state_unavailable" else "rejected"
    )
    assert "private-router-response" not in "".join(
        traceback.format_exception(failure.value)
    )
    assert "private-router-response" not in caplog.text


@pytest.mark.parametrize(
    "result",
    [
        {"isError": False, "content": []},
        {"isError": False, "content": [{"type": "text", "text": "not JSON"}]},
        {
            "isError": False,
            "content": [
                {"type": "text", "text": "{}"},
                {"type": "text", "text": "{}"},
            ],
        },
        _error("unsupported-error-code"),
        {"isError": True, "content": []},
        {
            "isError": True,
            "structuredContent": {"query_id": "service-errors", "removed": False},
            "content": [],
        },
        _success({"code": "state_unavailable", "message": "not a success"}),
    ],
)
def test_malformed_results_never_look_like_success(
    client: MCPRouterClient, server: _Server, result: dict[str, Any]
) -> None:
    server.queue("list_queries", result)
    with pytest.raises(RouterError) as failure:
        client.list_queries()
    assert failure.value.category == "invalid_response"


def test_invalid_structured_data_is_not_rescued_by_valid_text(
    client: MCPRouterClient, server: _Server
) -> None:
    result = _success(server.catalog)
    result["structuredContent"] = {"protocol_version": 2}
    server.queue("list_queries", result)
    with pytest.raises(RouterError) as failure:
        client.list_queries()
    assert failure.value.category == "invalid_response"


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("private-connection-error"),
        503,
    ],
)
def test_failed_initialization_closes_http_resources_and_is_not_an_empty_catalog(
    client: MCPRouterClient, server: _Server, error: Exception | int
) -> None:
    server.initialization_error = error
    with pytest.raises(RouterError) as failure:
        client.list_queries()
    assert failure.value.category == "transport"
    assert server.calls == []
    assert all(http.is_closed for http in server.clients)
    assert "private-connection-error" not in "".join(
        traceback.format_exception(failure.value)
    )


def test_incompatible_mcp_version_fails_as_invalid_response(
    client: MCPRouterClient, server: _Server
) -> None:
    server.protocol_version = "incompatible"
    with pytest.raises(RouterError) as failure:
        client.list_queries()
    assert failure.value.category == "invalid_response"
    assert server.calls == []


@pytest.mark.parametrize(
    "message",
    [
        "SDK initialization invariant failed",
        "Unsupported protocol version in client configuration",
    ],
)
def test_unexpected_initialization_runtime_errors_are_preserved(
    client: MCPRouterClient,
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    error = RuntimeError(message)

    async def broken_initialize(
        self: router_client.ClientSession,
    ) -> router_client.types.InitializeResult:
        raise error

    monkeypatch.setattr(router_client.ClientSession, "initialize", broken_initialize)
    with pytest.raises(ExceptionGroup) as failure:
        client.list_queries()
    assert failure.value.subgroup(lambda part: part is error) is not None
    assert failure.value.subgroup(RouterError) is None
    assert server.calls == []
    assert all(http.is_closed for http in server.clients)


def test_malformed_mcp_envelope_is_an_invalid_response(
    client: MCPRouterClient, server: _Server
) -> None:
    server.queue("list_queries", {"content": "private-invalid-content"})
    with pytest.raises(RouterError) as failure:
        client.list_queries()
    assert failure.value.category == "invalid_response"
    assert "private-invalid-content" not in "".join(
        traceback.format_exception(failure.value)
    )


def test_jsonrpc_failure_is_not_a_domain_rejection(
    client: MCPRouterClient,
    server: _Server,
    subscription: SubscribeRequest,
) -> None:
    client.list_queries()
    server.rpc_error = {"code": -32603, "message": "private-server-failure"}
    with pytest.raises(RouterError) as failure:
        client.subscribe(subscription)
    assert failure.value.category == "transport"
    assert failure.value.mutation_outcome == "uncertain"
    assert "private-server-failure" not in "".join(
        traceback.format_exception(failure.value)
    )


def test_timeout_closes_resources_and_preserves_uncertain_mutation(
    config: ResolvedDrasiConfig, server: _Server, subscription: SubscribeRequest
) -> None:
    client = MCPRouterClient(config, timeout_seconds=0.2)
    try:
        client.list_queries()
        server.stall = "subscribe"
        with pytest.raises(RouterError) as failure:
            client.subscribe(subscription)
        assert server.started.is_set()
        assert failure.value.category == "transport"
        assert failure.value.mutation_outcome == "uncertain"
        assert all(http.is_closed for http in server.clients)
    finally:
        client.close()
    assert all(not http.owner[0].is_alive() for http in server.clients)


def test_close_is_idempotent_and_rejects_even_cached_operations(
    client: MCPRouterClient,
    server: _Server,
    subscription: SubscribeRequest,
    removal: UnsubscribeRequest,
) -> None:
    client.list_queries()
    before = len(server.requests)
    client.close()
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.list_queries()
    with pytest.raises(RuntimeError, match="closed"):
        client.subscribe(subscription)
    with pytest.raises(RuntimeError, match="closed"):
        client.unsubscribe(removal)
    assert len(server.requests) == before


def test_close_before_preparation_never_creates_a_connection(
    client: MCPRouterClient, server: _Server
) -> None:
    client.close()
    client.close()
    assert server.clients == []


async def test_sync_client_does_not_borrow_the_callers_event_loop(
    client: MCPRouterClient, server: _Server
) -> None:
    caller_loop = asyncio.get_running_loop()
    assert client.list_queries().queries
    assert all(http.owner[1] is not caller_loop for http in server.clients)


def test_concurrent_preparation_fetches_once(
    client: MCPRouterClient, server: _Server
) -> None:
    with ThreadPoolExecutor(max_workers=4) as callers:
        results = list(callers.map(lambda _: client.list_queries(), range(8)))
    assert len(server.calls) == 1
    assert all(result is not results[0] for result in results[1:])


def test_shutdown_waits_for_inflight_operation_without_unsubscribing(
    client: MCPRouterClient,
    server: _Server,
    subscription: SubscribeRequest,
) -> None:
    client.list_queries()
    server.queue("subscribe", _success(_subscribe_response(client.scope)))
    server.stall = "subscribe"
    with ThreadPoolExecutor(max_workers=2) as callers:
        mutation = callers.submit(client.subscribe, subscription)
        try:
            assert server.started.wait(timeout=2)
            closing = callers.submit(client.close)
            assert not closing.done()
        finally:
            server.release.set()
        assert mutation.result(timeout=2).status.value == "created"
        closing.result(timeout=2)
    assert [call["name"] for call in server.calls] == ["list_queries", "subscribe"]
    assert all(http.is_closed for http in server.clients)


def test_unexpected_programming_errors_are_not_relabelled(
    client: MCPRouterClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ExceptionGroup("programming error", [TypeError("implementation bug")])

    def broken(*args: Any) -> None:
        raise error

    monkeypatch.setattr(client, "_run_exchange", broken)
    with pytest.raises(ExceptionGroup) as failure:
        client.list_queries()
    assert failure.value is error


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(
    config: ResolvedDrasiConfig, timeout: float
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MCPRouterClient(config, timeout_seconds=timeout)
