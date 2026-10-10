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

"""Private, scope-bound MCP access to the stateless DaprAgentRouter."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Lock
from typing import Any, NoReturn, TypeVar

import httpx
from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from dapr.clients.http.conf import DAPR_API_TOKEN_HEADER
from dapr.conf import settings
from drasi_agent_router_contracts import (
    ListQueriesRequest,
    ListQueriesResponse,
    SubscribeRequest,
    SubscribeResponse,
    ToolError,
    UnsubscribeRequest,
    UnsubscribeResponse,
    parse,
    parse_catalog,
    to_wire,
)
from drasi_agent_router_contracts.models.ToolError import Code
from jsonschema.exceptions import ValidationError as SchemaValidationError
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from pydantic import ValidationError

from ._interfaces import RouterClient, RouterError
from ._models import ResolvedDrasiConfig, RouterFailureCategory, SubscriptionScope

logger = logging.getLogger(__name__)

_Request = TypeVar("_Request", SubscribeRequest, UnsubscribeRequest)
_WIRE_ERRORS = (SchemaValidationError, ValidationError, ValueError)
_TRANSPORT_ERRORS = (
    httpx.HTTPError,
    McpError,
    TimeoutError,
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    OSError,
)
_ERROR_CATEGORIES: dict[Code, RouterFailureCategory] = {
    Code.invalid_arguments: "invalid_arguments",
    Code.unknown_query: "unknown_query",
    Code.incarnation_conflict: "incarnation_conflict",
    Code.state_unavailable: "state_unavailable",
}


def _transport_failure(error: Exception) -> RouterFailureCategory | None:
    if isinstance(error, RouterError):
        return error.category
    if isinstance(error, (ValidationError, json.JSONDecodeError)):
        return "invalid_response"
    if isinstance(error, _TRANSPORT_ERRORS):
        return "transport"
    if isinstance(error, ExceptionGroup):
        categories = [_transport_failure(part) for part in error.exceptions]
        if None in categories:
            return None
        if "transport" in categories:
            return "transport"
        return categories[0]
    return None


def _text_document(result: types.CallToolResult) -> Any:
    if len(result.content) != 1 or not isinstance(result.content[0], types.TextContent):
        raise ValueError("Expected one JSON text result.")
    return json.loads(result.content[0].text)


class MCPRouterClient(RouterClient):
    """Blocking router operations, with async resources confined to a worker.

    The owner calls ``list_queries`` during preparation and owns ``close``. Mutations
    also establish the catalog identity if needed, before sending any write.
    Each exchange initializes and closes its own stateless MCP session in one
    task; no HTTP client, task group, or event loop crosses calls.

    Calls are serialized with shutdown and return completed, detached results.
    Async application callers must offload these blocking methods. Closing
    waits for the current bounded exchange and never deletes durable rules.
    """

    def __init__(
        self,
        config: ResolvedDrasiConfig,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            logger.error("Drasi router timeout must be finite and positive.")
            raise ValueError("Router timeout must be finite and positive.")
        self._config = config
        self._timeout_seconds = timeout_seconds
        self._lock = Lock()
        self._closed = False
        self._catalog: ListQueriesResponse | None = None
        self._worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="drasi-router"
        )

    @property
    def scope(self) -> SubscriptionScope:
        return self._config.scope

    def list_queries(self) -> ListQueriesResponse:
        with self._lock:
            self._require_open()
            return self._prepare_catalog().model_copy(deep=True)

    def subscribe(self, request: SubscribeRequest) -> SubscribeResponse:
        with self._lock:
            self._require_open()
            expected = self._validate_request(SubscribeRequest, request, "subscribe")
            self._prepare_catalog()
            document = self._call("subscribe", to_wire(expected))
            try:
                response = parse(SubscribeResponse, document)
                if (
                    response.query_id != expected.query_id
                    or response.subscription_incarnation
                    != expected.subscription_incarnation
                    or set(response.operations) != set(expected.operations)
                    or response.topic_name != self.scope.inbox_topic
                ):
                    raise ValueError("Subscription confirmation does not match.")
            except _WIRE_ERRORS:
                self._fail("invalid_response", "subscribe")
            return response

    def unsubscribe(self, request: UnsubscribeRequest) -> UnsubscribeResponse:
        with self._lock:
            self._require_open()
            expected = self._validate_request(
                UnsubscribeRequest, request, "unsubscribe"
            )
            self._prepare_catalog()
            document = self._call("unsubscribe", to_wire(expected))
            try:
                response = parse(UnsubscribeResponse, document)
                if response.query_id != expected.query_id:
                    raise ValueError("Unsubscribe confirmation does not match.")
            except _WIRE_ERRORS:
                self._fail("invalid_response", "unsubscribe")
            return response

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._worker.shutdown(wait=True)
            self._catalog = None

    def _require_open(self) -> None:
        if self._closed:
            logger.error("Drasi router client used after close.")
            raise RuntimeError("Drasi router client is closed.")

    def _prepare_catalog(self) -> ListQueriesResponse:
        if self._catalog is None:
            document = self._call("list_queries", to_wire(ListQueriesRequest()))
            try:
                self._catalog = parse_catalog(document, self.scope.router_id)
            except _WIRE_ERRORS:
                self._fail("invalid_response", "list_queries")
        return self._catalog

    def _validate_request(
        self, model: type[_Request], request: _Request, operation: str
    ) -> _Request:
        try:
            document = request.model_dump(
                mode="json", exclude_unset=True, warnings="error"
            )
            validated = parse(model, document)
            if validated.subscriber != self.scope.subscriber:
                raise ValueError("Request subscriber does not match client scope.")
            return validated
        except _WIRE_ERRORS:
            self._fail("invalid_arguments", operation)

    def _call(self, operation: str, arguments: dict[str, Any]) -> Any:
        try:
            result = self._worker.submit(
                self._run_exchange, operation, arguments
            ).result()
        except (
            ExceptionGroup,
            RouterError,
            ValidationError,
            json.JSONDecodeError,
            *_TRANSPORT_ERRORS,
        ) as error:
            category = _transport_failure(error)
            if category is None:
                raise
            self._fail(category, operation)

        if result.isError:
            try:
                if result.structuredContent is not None:
                    raise ValueError("Tool errors must not carry success data.")
                error_result = parse(ToolError, _text_document(result))
            except _WIRE_ERRORS:
                self._fail("invalid_response", operation)
            self._fail(_ERROR_CATEGORIES[error_result.code], operation)

        if result.structuredContent is not None:
            return result.structuredContent
        try:
            return _text_document(result)
        except _WIRE_ERRORS:
            self._fail("invalid_response", operation)

    def _run_exchange(
        self, operation: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._exchange(operation, arguments))
        raise RuntimeError("Drasi router worker already has an active event loop.")

    async def _exchange(
        self, operation: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        headers: dict[str, str] = {}
        token = settings.DAPR_API_TOKEN
        if token is not None:
            headers[DAPR_API_TOKEN_HEADER] = token
        async with asyncio.timeout(self._timeout_seconds):
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_seconds),
                headers=headers,
                trust_env=False,
            ) as http_client:
                async with streamable_http_client(
                    self._config.router_mcp_url,
                    http_client=http_client,
                    terminate_on_close=False,
                ) as (read_stream, write_stream, _):
                    async with ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=self._timeout_seconds),
                    ) as session:
                        try:
                            await session.initialize()
                        except RuntimeError as error:
                            # The SDK has no typed version-negotiation exception.
                            if not str(error).startswith(
                                "Unsupported protocol version from the server: "
                            ):
                                raise
                            raise RouterError("invalid_response") from None
                        # call_tool insists on structuredContent for output schemas.
                        # Validate both documented representations ourselves instead.
                        return await session.send_request(
                            types.ClientRequest(
                                types.CallToolRequest(
                                    params=types.CallToolRequestParams(
                                        name=operation, arguments=arguments
                                    )
                                )
                            ),
                            types.CallToolResult,
                        )

    def _fail(self, category: RouterFailureCategory, operation: str) -> NoReturn:
        logger.error(
            "Drasi router operation failed: router=%s operation=%s category=%s",
            self.scope.router_id,
            operation,
            category,
        )
        raise RouterError(category) from None
