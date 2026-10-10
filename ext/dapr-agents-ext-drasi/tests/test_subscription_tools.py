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

"""Focused tool, schema, naming, and management-boundary coverage."""

from __future__ import annotations

import json
import logging
import re
from itertools import permutations

import pytest
from drasi_agent_router_contracts import ListQueriesResponse, parse, to_wire
from drasi_agent_router_contracts.models.Operation import Operation
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as WireValidationError

from dapr_agents.ext.drasi import subscription_tools
from dapr_agents.ext.drasi._interfaces import (
    IntentStoreError,
    RouterError,
    SubscriptionCommandError,
)
from dapr_agents.ext.drasi._models import (
    SubscriptionIntent,
    SubscriptionStatus,
)
from dapr_agents.ext.drasi.subscription_tools import build_subscription_tools
from dapr_agents.tool import AgentTool
from dapr_agents.tool.executor import AgentToolExecutor
from dapr_agents.types import AgentToolExecutorError, ToolError, ToolResult

from .fakes import ScriptedSubscriptionManager


@pytest.fixture
def manager() -> ScriptedSubscriptionManager:
    return ScriptedSubscriptionManager()


@pytest.fixture
def tools(
    catalog: ListQueriesResponse, manager: ScriptedSubscriptionManager
) -> list[AgentTool]:
    return build_subscription_tools(catalog, manager)


def _catalog_with_ids(
    catalog: ListQueriesResponse, query_ids: list[str]
) -> ListQueriesResponse:
    document = to_wire(catalog)
    document["queries"] = [
        {
            "query_id": query_id,
            "title": f"Query {index}",
            "description": f"Meaningful query {index}.",
        }
        for index, query_id in enumerate(query_ids)
    ]
    return parse(ListQueriesResponse, document)


def test_factory_is_inert_and_builds_two_tools_per_query(
    catalog: ListQueriesResponse,
    manager: ScriptedSubscriptionManager,
    tools: list[AgentTool],
) -> None:
    assert len(tools) == 2 * len(catalog.queries) + 1
    assert tools[-1].name == "list_drasi_subscriptions"
    assert manager.subscribe_calls == ()
    assert manager.unsubscribe_calls == ()
    assert manager.reconciled_catalogs == ()
    for tool in tools:
        assert type(tool) is AgentTool
        assert tool._is_async is False
        assert tool.signature.startswith(f"{tool.name}(")


@pytest.mark.parametrize("format_type", ("openai", "anthropic"))
def test_schemas_expose_only_required_model_decisions(
    tools: list[AgentTool], format_type: str
) -> None:
    for index, tool in enumerate(tools):
        definition = tool.to_function_call(format_type=format_type)
        if format_type == "openai":
            schema = definition["function"]["parameters"]
            assert definition["function"]["name"] == tool.name
        else:
            schema = definition["input_schema"]
            assert definition["name"] == tool.name
        assert schema["additionalProperties"] is False
        is_subscribe = index < len(tools) - 1 and index % 2 == 0
        expected = {"operations", "instructions"} if is_subscribe else set()
        assert set(schema["properties"]) == expected
        assert set(schema.get("required", ())) == expected
        if is_subscribe:
            operations = schema["properties"]["operations"]
            assert operations["type"] == "array"
            assert operations["minItems"] == 1
            assert operations["maxItems"] == 3
            assert operations["uniqueItems"] is True
            validator = Draft202012Validator(schema)
            validator.validate(
                {"operations": ["u", "i"], "instructions": "Handle each change."}
            )
            for invalid in (
                {"operations": [], "instructions": "Handle each change."},
                {"operations": ["i", "i"], "instructions": "Handle each change."},
                {"operations": ["x"], "instructions": "Handle each change."},
                {"operations": ["i"], "instructions": " \n\t"},
                {"operations": ["i"], "instructions": 42},
            ):
                with pytest.raises(WireValidationError):
                    validator.validate(invalid)
        else:
            Draft202012Validator(schema).validate({})


def test_descriptions_supply_catalog_context_and_monitoring_semantics(
    catalog: ListQueriesResponse, tools: list[AgentTool]
) -> None:
    for query, subscribe, unsubscribe in zip(
        catalog.queries, tools[:-1:2], tools[1:-1:2], strict=True
    ):
        for tool in (subscribe, unsubscribe):
            assert query.query_id in tool.description
            assert query.title in tool.description
            assert query.description in tool.description
            if query.usage is not None:
                assert query.usage in tool.description
            else:
                assert "Usage:" not in tool.description
            for phrase in (
                "projected query-result rows",
                "entering the result set",
                "before and after",
                "leaving the result set",
                "not necessarily database",
                "persists across tasks",
                "ordinary restarts",
                "no historical catch-up",
                "duplicate deliveries",
            ):
                assert phrase in tool.description
        assert "independent workflow" in subscribe.description
        assert "idempotently replaces" in subscribe.description
        assert "does not add another subscription" in subscribe.description
        assert "does not cancel already scheduled workflows" in unsubscribe.description
    assert "locally persisted" in tools[-1].description
    assert "does not query or verify live router state" in tools[-1].description


def test_names_are_bounded_unique_and_stable_under_catalog_reordering(
    catalog: ListQueriesResponse, manager: ScriptedSubscriptionManager
) -> None:
    query_ids = [
        "A_B",
        "a b",
        "ab",
        "a.b",
        "a/b",
        "a-b",
        "A-B",
        "a" * 200,
        "a" * 199 + "b",
        " \t\n",
        "\u96ea",
        "\u00e9",
        "e\u0301",
        "!!!",
    ]
    catalog = _catalog_with_ids(catalog, query_ids)
    tools = build_subscription_tools(catalog, manager)
    normalized = [AgentToolExecutor._normalize(tool.name) for tool in tools]
    assert len(normalized) == len(set(normalized))
    for tool in tools:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", tool.name)
        assert tool.name == tool.to_function_call()["function"]["name"]
    names = {
        query_id: (tools[2 * index].name, tools[2 * index + 1].name)
        for index, query_id in enumerate(query_ids)
    }
    reverse_catalog = _catalog_with_ids(catalog, list(reversed(query_ids)))
    reverse_tools = build_subscription_tools(reverse_catalog, manager)
    reverse_names = {
        query.query_id: (
            reverse_tools[2 * index].name,
            reverse_tools[2 * index + 1].name,
        )
        for index, query in enumerate(reverse_catalog.queries)
    }
    assert reverse_names == names


@pytest.mark.parametrize(
    ("subscribe_name", "unsubscribe_name"),
    (
        ("Collision_Name", "collisionname"),
        ("list_drasi_subscriptions", "different_name"),
    ),
)
def test_generated_name_collisions_fail_before_attachment(
    catalog: ListQueriesResponse,
    manager: ScriptedSubscriptionManager,
    monkeypatch: pytest.MonkeyPatch,
    subscribe_name: str,
    unsubscribe_name: str,
) -> None:
    catalog = _catalog_with_ids(catalog, ["query"])
    monkeypatch.setattr(
        subscription_tools,
        "_tool_name",
        lambda command, query_id: (
            subscribe_name if command == "subscribe" else unsubscribe_name
        ),
    )
    with pytest.raises(ValueError, match="tool name collision"):
        build_subscription_tools(catalog, manager)
    assert manager.subscribe_calls == ()


def test_factory_rejects_duplicate_catalog_queries(
    catalog: ListQueriesResponse, manager: ScriptedSubscriptionManager
) -> None:
    catalog.queries.append(catalog.queries[0].model_copy(deep=True))
    with pytest.raises(ValueError, match="Invalid Drasi subscription tool catalog"):
        build_subscription_tools(catalog, manager)


def test_invalid_catalog_diagnostics_do_not_echo_catalog_payload(
    catalog: ListQueriesResponse,
    manager: ScriptedSubscriptionManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "private-catalog-payload"
    catalog.queries[0].description = marker
    catalog.queries[0].usage = None
    with pytest.raises(
        ValueError, match="Invalid Drasi subscription tool catalog"
    ) as failure:
        build_subscription_tools(catalog, manager)
    assert marker not in str(failure.value)
    assert marker not in caplog.text


def test_each_tool_binds_its_own_query_and_detaches_catalog_metadata(
    catalog: ListQueriesResponse,
    manager: ScriptedSubscriptionManager,
    tools: list[AgentTool],
) -> None:
    original = parse(ListQueriesResponse, to_wire(catalog))
    instructions = "  Handle each event independently.\nDeduplicate external actions.  "
    for query in catalog.queries:
        query.query_id = "mutated"
        query.title = "mutated title"
    for query, subscribe, unsubscribe in zip(
        original.queries, tools[:-1:2], tools[1:-1:2], strict=True
    ):
        manager.queue_subscribe(
            SubscriptionIntent(
                query_id=query.query_id,
                operations=(Operation.u, Operation.i),
                instructions=instructions,
                catalog_snapshot=original,
                incarnation=f"incarnation-{query.query_id}",
                status="active",
            )
        )
        manager.queue_unsubscribe()
        result = subscribe.run(operations=["u", "i"], instructions=instructions)
        assert isinstance(result, ToolResult)
        assert result.isError is False
        assert result.structuredContent == {
            "result": {
                "query_id": query.query_id,
                "operations": ["u", "i"],
                "status": "active",
            }
        }
        assert "mutated title" not in subscribe.description
        stopped = unsubscribe.run()
        assert stopped.structuredContent == {
            "result": {"query_id": query.query_id, "subscribed": False}
        }
        assert stopped.isError is False
    assert [call.query_id for call in manager.subscribe_calls] == [
        query.query_id for query in original.queries
    ]
    assert all(
        call.operations == (Operation.u, Operation.i)
        and call.instructions == instructions
        for call in manager.subscribe_calls
    )
    assert manager.unsubscribe_calls == tuple(
        query.query_id for query in original.queries
    )


@pytest.mark.parametrize(
    "operations",
    [list(items) for size in (1, 2, 3) for items in permutations("iud", size)],
)
def test_all_nonempty_operation_subsets_and_orders_delegate(
    tools: list[AgentTool],
    manager: ScriptedSubscriptionManager,
    active_intent: SubscriptionIntent,
    operations: list[str],
) -> None:
    manager.queue_subscribe(active_intent)
    result = tools[0].run(operations=operations, instructions="Handle each change.")
    assert result.isError is False
    assert manager.subscribe_calls[-1].operations == tuple(
        Operation(operation) for operation in operations
    )


@pytest.mark.parametrize(
    "operations",
    ([], ["i", "i"], ["x"], ["I"], ["i", "u", "d", "i"], "i", None, [True], [1], {}),
)
def test_invalid_operations_never_reach_manager(
    tools: list[AgentTool],
    manager: ScriptedSubscriptionManager,
    operations: object,
) -> None:
    with pytest.raises(ToolError):
        tools[0].run(operations=operations, instructions="Handle each change.")
    assert manager.subscribe_calls == ()


@pytest.mark.parametrize("instructions", ("", " \n\t", None, 123, True, {}, []))
def test_invalid_instructions_never_reach_manager(
    tools: list[AgentTool],
    manager: ScriptedSubscriptionManager,
    instructions: object,
) -> None:
    with pytest.raises(ToolError):
        tools[0].run(operations=["i"], instructions=instructions)
    assert manager.subscribe_calls == ()


@pytest.mark.parametrize(
    "arguments",
    ({}, {"operations": ["i"]}, {"instructions": "Handle each change."}),
)
def test_subscribe_requires_both_decisions(
    tools: list[AgentTool],
    manager: ScriptedSubscriptionManager,
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ToolError):
        tools[0].run(**arguments)
    assert manager.subscribe_calls == ()


@pytest.mark.parametrize(
    "field",
    (
        "query_id",
        "subscriber",
        "namespace",
        "app_id",
        "agent_name",
        "router_id",
        "pubsub_name",
        "topic_name",
        "incarnation",
        "subscription_incarnation",
    ),
)
def test_hidden_coordinates_cannot_be_injected(
    tools: list[AgentTool],
    manager: ScriptedSubscriptionManager,
    field: str,
) -> None:
    with pytest.raises(ToolError):
        tools[0].run(
            operations=["i"], instructions="Handle each change.", **{field: "override"}
        )
    for tool in (tools[1], tools[-1]):
        with pytest.raises(ToolError):
            tool.run(**{field: "override"})
    assert manager.subscribe_calls == ()
    assert manager.unsubscribe_calls == ()


def test_repeated_commands_delegate_without_local_subscription_state(
    tools: list[AgentTool],
    manager: ScriptedSubscriptionManager,
    active_intent: SubscriptionIntent,
) -> None:
    for instructions in ("Handle initial changes.", "Use updated handling."):
        manager.queue_subscribe(active_intent)
        manager.queue_unsubscribe()
        assert (
            tools[0].run(operations=["i"], instructions=instructions).isError is False
        )
        assert tools[1].run().isError is False
    assert [call.instructions for call in manager.subscribe_calls] == [
        "Handle initial changes.",
        "Use updated handling.",
    ]
    assert {call.query_id for call in manager.subscribe_calls} == {
        active_intent.query_id
    }
    assert manager.unsubscribe_calls == (active_intent.query_id, active_intent.query_id)


def test_empty_catalog_still_has_local_inspection(
    empty_catalog: ListQueriesResponse, manager: ScriptedSubscriptionManager
) -> None:
    tools = build_subscription_tools(empty_catalog, manager)
    assert [tool.name for tool in tools] == ["list_drasi_subscriptions"]
    manager.set_subscriptions(())
    result = tools[0].run()
    assert result.isError is False
    assert result.structuredContent == {
        "result": {"source": "local_intent", "subscriptions": []}
    }


def test_listing_reports_all_local_statuses_without_internal_metadata(
    tools: list[AgentTool],
    manager: ScriptedSubscriptionManager,
    status_intents: dict[SubscriptionStatus, SubscriptionIntent],
) -> None:
    manager.set_subscriptions(tuple(status_intents.values()))
    result = tools[-1].run()
    assert result.isError is False
    assert result.structuredContent == {
        "result": {
            "source": "local_intent",
            "subscriptions": [
                {
                    "query_id": intent.query_id,
                    "operations": [operation.value for operation in intent.operations],
                    "instructions": intent.instructions,
                    "status": intent.status,
                }
                for intent in status_intents.values()
            ],
        }
    }
    assert "not live router state" in result.content[0].text
    serialized = result.model_dump_json()
    assert "catalog_snapshot" not in serialized
    assert "incarnation" not in serialized
    assert "last_outcome" not in serialized
    assert json.loads(serialized)["isError"] is False


def test_listing_includes_retired_intent_missing_from_current_catalog(
    empty_catalog: ListQueriesResponse,
    manager: ScriptedSubscriptionManager,
    status_intents: dict[SubscriptionStatus, SubscriptionIntent],
) -> None:
    retired = status_intents["unavailable"]
    manager.set_subscriptions((retired,))
    tools = build_subscription_tools(empty_catalog, manager)
    result = tools[0].run()
    assert result.isError is False
    assert retired.query_id in result.content[0].text
    assert "unavailable" in result.content[0].text


@pytest.mark.parametrize("command", ("subscribe", "unsubscribe", "list"))
@pytest.mark.parametrize(
    "error",
    (
        SubscriptionCommandError("invalid_input"),
        SubscriptionCommandError("unknown_query"),
        SubscriptionCommandError("pending_operation"),
        SubscriptionCommandError("unavailable"),
        RouterError("invalid_arguments"),
        RouterError("unknown_query"),
        RouterError("incarnation_conflict"),
        RouterError("state_unavailable"),
        RouterError("transport"),
        RouterError("invalid_response"),
        IntentStoreError("unavailable"),
        IntentStoreError("corrupt"),
        IntentStoreError("unsupported_version"),
        IntentStoreError("conflict"),
    ),
)
def test_classified_failures_remain_readable_errors(
    tools: list[AgentTool],
    manager: ScriptedSubscriptionManager,
    command: str,
    error: SubscriptionCommandError | RouterError | IntentStoreError,
) -> None:
    if command == "subscribe":
        manager.queue_subscribe_error(error)
        result = tools[0].run(operations=["i"], instructions="Handle each change.")
    elif command == "unsubscribe":
        manager.queue_unsubscribe(error)
        result = tools[1].run()
    else:
        manager.queue_list_error(error)
        result = tools[-1].run()
    assert result.isError is True
    assert result.structuredContent is None
    assert f"Drasi {command} failed" in result.content[0].text
    if isinstance(error, RouterError) and error.mutation_outcome == "uncertain":
        if command != "list":
            assert "may have taken effect" in result.content[0].text
        else:
            assert "may have taken effect" not in result.content[0].text
    if isinstance(error, IntentStoreError):
        if command == "list":
            assert "read failed" in result.content[0].text
            assert "Retry listing after storage recovery" in result.content[0].text
            assert "rolled back" not in result.content[0].text
        else:
            assert "an attempted change was rolled back" in result.content[0].text


def test_failure_diagnostics_do_not_echo_instructions_or_exception_payload(
    tools: list[AgentTool],
    manager: ScriptedSubscriptionManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "private-handling-instructions-and-server-payload"

    class PayloadRouterError(RouterError):
        def __str__(self) -> str:
            return marker

    caplog.set_level(logging.DEBUG)
    error = PayloadRouterError("transport")
    error.__cause__ = RuntimeError(marker)
    manager.queue_subscribe_error(error)
    result = tools[0].run(operations=["i"], instructions=marker)
    assert result.isError is True
    assert marker not in result.model_dump_json()
    assert marker not in caplog.text
    assert "category=transport" in caplog.text
    with pytest.raises(ToolError) as failure:
        tools[0].run(operations=["invalid"], instructions=marker, unknown=marker)
    assert marker not in str(failure.value)
    assert marker not in caplog.text


def test_success_and_listing_do_not_log_handling_instructions(
    tools: list[AgentTool],
    manager: ScriptedSubscriptionManager,
    active_intent: SubscriptionIntent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    marker = "private-success-and-listing-instructions"
    active_intent.instructions = marker
    manager.queue_subscribe(active_intent)
    manager.set_subscriptions((active_intent,))
    assert tools[0].run(operations=["i"], instructions=marker).isError is False
    result = tools[-1].run()
    assert marker in result.content[0].text
    assert marker not in caplog.text


@pytest.mark.asyncio
async def test_generated_tools_use_existing_executor_without_replacing_tools(
    catalog: ListQueriesResponse,
    manager: ScriptedSubscriptionManager,
    active_intent: SubscriptionIntent,
) -> None:
    ordinary = AgentTool(
        name="existing_action", description="An ordinary action.", func=lambda: "done"
    )
    executor = AgentToolExecutor(tools=[ordinary])
    tools = build_subscription_tools(catalog, manager)
    assert executor.list_tools() == [ordinary]
    for tool in tools:
        executor.register_tool(tool)
    assert executor.get_tool(ordinary.name) is ordinary
    assert await executor.run_tool(ordinary.name) == "done"
    manager.queue_subscribe(active_intent)
    success = await executor.run_tool(
        tools[0].name, operations=["i"], instructions="Handle each change."
    )
    assert success.isError is False
    invalid = await executor.run_tool(
        tools[0].name, operations=[], instructions="Handle each change."
    )
    assert isinstance(invalid, ToolResult)
    assert invalid.isError is True
    assert len(manager.subscribe_calls) == 1
    assert executor.list_tools() == [ordinary, *tools]


@pytest.mark.parametrize("index", (0, -1))
def test_collisions_with_existing_tools_are_not_silently_overwritten(
    tools: list[AgentTool], index: int
) -> None:
    generated = tools[index]
    ordinary = AgentTool(
        name=generated.name.upper().replace("_", ""),
        description="An existing action with a normalized name collision.",
        func=lambda: "existing",
    )
    executor = AgentToolExecutor(tools=[ordinary])
    with pytest.raises(AgentToolExecutorError, match="already registered"):
        executor.register_tool(generated)
    assert executor.get_tool(generated.name) is ordinary
    assert executor.list_tools() == [ordinary]


@pytest.mark.asyncio
async def test_unexpected_manager_failure_is_not_a_successful_empty_listing(
    tools: list[AgentTool],
) -> None:
    executor = AgentToolExecutor(tools=tools)
    result = await executor.run_tool("list_drasi_subscriptions")
    assert isinstance(result, ToolResult)
    assert result.isError is True
    assert result.structuredContent is None
    assert "No scripted manager subscription listing" in result.content[0].text
