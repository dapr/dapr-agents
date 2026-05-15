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

"""Unit tests for the before_llm_call / after_llm_call hook wiring in call_llm."""

import os
from datetime import timedelta
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from dapr.ext.workflow import WorkflowActivityContext

from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.hooks import (
    Deny,
    HookContext,
    Hooks,
    Mutate,
    Proceed,
    RequireApproval,
    Skip,
)
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService


@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
    import dapr.ext.workflow as wf

    mock_runtime = Mock(spec=wf.WorkflowRuntime)
    monkeypatch.setattr(wf, "WorkflowRuntime", lambda: mock_runtime)

    class MockRetryPolicy:
        def __init__(
            self,
            max_number_of_attempts=1,
            first_retry_interval=timedelta(seconds=1),
            max_retry_interval=timedelta(seconds=60),
            backoff_coefficient=2.0,
            retry_timeout: Optional[timedelta] = None,
        ):
            pass

    monkeypatch.setattr(wf, "RetryPolicy", MockRetryPolicy)
    yield mock_runtime


class _MockDaprClient:
    def __init__(self, *_, **__):
        self.get_state = MagicMock(return_value=Mock(data=None, json=lambda: {}))
        self.save_state = MagicMock()
        self.delete_state = MagicMock()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __call__(self, *_, **__):
        return self

    def get_metadata(self):
        resp = MagicMock()
        resp.registered_components = []
        resp.application_id = "test-app"
        return resp


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    os.environ["OPENAI_API_KEY"] = "test-key"
    client = _MockDaprClient()
    monkeypatch.setattr("dapr.clients.DaprClient", lambda *a, **kw: client)
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.statestore.DaprClient",
        lambda *a, **kw: client,
    )
    yield
    os.environ.pop("OPENAI_API_KEY", None)


def _mock_llm_response(content: str = "ok"):
    """Build a fake llm.generate(...) return value that behaves like a chat response."""
    fake_assistant = Mock()
    fake_assistant.model_dump.return_value = {"role": "assistant", "content": content}
    fake_response = Mock()
    fake_response.get_message.return_value = fake_assistant
    return fake_response


@pytest.fixture
def mock_llm():
    # spec=OpenAIChatClient gives the Mock the right interface for isinstance
    # checks. We don't override __class__.__name__ globally (which would leak
    # across tests) — the agent doesn't depend on the LLM client's class name.
    m = Mock(spec=OpenAIChatClient)
    m.generate = Mock(return_value=_mock_llm_response("from-llm"))
    m.prompt_template = None
    m.provider = "MockProvider"
    m.api = "MockAPI"
    m.model = "gpt-4o-mock"
    return m


@pytest.fixture
def mock_activity_ctx():
    return Mock(spec=WorkflowActivityContext)


def _make_agent(mock_llm, hooks=None):
    return DurableAgent(
        name="LLMHookTestAgent",
        role="LLM Hook Tester",
        goal="Test hook dispatch",
        instructions=[],
        llm=mock_llm,
        tools=[],
        pubsub=AgentPubSubConfig(
            pubsub_name="testpubsub",
            agent_topic="LLMHookTestAgent",
        ),
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
        execution=AgentExecutionConfig(max_iterations=1),
        hooks=hooks,
    )


def _patch_activity_deps(agent):
    """
    Patch the parent-class collaborators call_llm depends on so the activity
    body can run end-to-end against in-memory fakes. Returns the patcher list;
    each must be __enter__'d (or use as context manager).
    """
    fake_entry = MagicMock()
    history_msgs = [
        {"role": "system", "content": "you are a helpful assistant"},
    ]
    initial_msgs = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "what is dapr?"},
    ]
    return [
        patch.object(agent._infra, "get_state", return_value=fake_entry),
        patch.object(
            agent,
            "_reconstruct_conversation_history",
            return_value=history_msgs,
        ),
        patch.object(
            agent.prompting_helper,
            "build_initial_messages",
            return_value=initial_msgs,
        ),
        patch.object(agent, "_sync_system_messages_with_state"),
        patch.object(agent, "_process_user_message"),
        patch.object(agent, "_get_last_user_message", return_value=initial_msgs[-1]),
        patch.object(agent, "get_llm_tools", return_value=[]),
        patch.object(agent, "_save_assistant_message"),
        patch.object(agent, "save_state"),
        patch.object(agent.text_formatter, "print_message"),
    ]


def _run_call_llm(agent, mock_ctx, payload=None):
    """Invoke agent.call_llm with all collaborators patched out."""
    payload = payload or {"instance_id": "wf-test", "task": "what is dapr?"}
    patchers = _patch_activity_deps(agent)
    for p in patchers:
        p.start()
    try:
        return agent.call_llm(mock_ctx, payload)
    finally:
        for p in patchers:
            p.stop()


class TestBeforeLLMCallHook:
    def test_no_hooks_calls_llm_normally(self, mock_llm, mock_activity_ctx):
        agent = _make_agent(mock_llm)
        result = _run_call_llm(agent, mock_activity_ctx)

        assert mock_llm.generate.call_count == 1
        assert result == {"role": "assistant", "content": "from-llm"}

    def test_proceed_does_not_block(self, mock_llm, mock_activity_ctx):
        captured: list[HookContext] = []

        def hook(ctx):
            captured.append(ctx)
            return Proceed()

        agent = _make_agent(mock_llm, Hooks(before_llm_call=[hook]))
        result = _run_call_llm(agent, mock_activity_ctx)

        assert len(captured) == 1
        assert captured[0].step_kind == "llm"
        assert captured[0].step_name == "llm"
        assert captured[0].source == "agent"
        assert "messages" in captured[0].payload
        assert mock_llm.generate.call_count == 1
        assert result == {"role": "assistant", "content": "from-llm"}

    def test_mutate_merges_into_generate_kwargs(self, mock_llm, mock_activity_ctx):
        """before_llm_call's Mutate payload is shallow-merged, so a hook that
        returns only `messages` still gets its messages applied without having
        to spread `**ctx.payload` to preserve other kwargs."""
        new_messages = [
            {"role": "system", "content": "REWRITTEN system"},
            {"role": "user", "content": "REWRITTEN user"},
        ]

        def hook(_):
            return Mutate(payload={"messages": new_messages})

        agent = _make_agent(mock_llm, Hooks(before_llm_call=[hook]))
        _run_call_llm(agent, mock_activity_ctx)

        # LLM client must have seen the rewritten messages, not the originals
        called_kwargs = mock_llm.generate.call_args.kwargs
        assert called_kwargs["messages"] == new_messages

    def test_mutate_preserves_other_kwargs(self, mock_llm, mock_activity_ctx):
        """A hook that returns Mutate with only `messages` must NOT drop
        `tools` (or any other generate kwargs originally on the call). This is
        the regression test for the framework-side merge behavior."""
        original_tools = [
            {
                "type": "function",
                "function": {"name": "do_thing", "parameters": {}},
            }
        ]

        def hook(_):
            return Mutate(payload={"messages": [{"role": "user", "content": "x"}]})

        agent = _make_agent(mock_llm, Hooks(before_llm_call=[hook]))
        # Reuse the default activity patchers but swap in a non-empty tools list
        # so we can verify the merge preserves them.
        patchers = [
            p
            for p in _patch_activity_deps(agent)
            if getattr(p, "attribute", None) != "get_llm_tools"
        ]
        patchers.append(
            patch.object(agent, "get_llm_tools", return_value=original_tools)
        )
        for p in patchers:
            p.start()
        try:
            agent.call_llm(
                mock_activity_ctx,
                {"instance_id": "wf-test", "task": "x"},
            )
        finally:
            for p in patchers:
                p.stop()

        called_kwargs = mock_llm.generate.call_args.kwargs
        assert called_kwargs["tools"] == original_tools
        assert called_kwargs["messages"] == [{"role": "user", "content": "x"}]

    def test_mutate_empty_payload_is_noop(self, mock_llm, mock_activity_ctx):
        """Mutate(payload={}) shallow-merges nothing into generate_kwargs, and
        Mutate(payload=None) skips the merge branch entirely. Both must leave
        the LLM call unaffected so a hook can no-op explicitly without breaking
        downstream dispatch."""

        def empty_payload_hook(_):
            return Mutate(payload={})

        def none_payload_hook(_):
            return Mutate(payload=None)

        for hook in (empty_payload_hook, none_payload_hook):
            mock_llm.generate.reset_mock()
            agent = _make_agent(mock_llm, Hooks(before_llm_call=[hook]))
            result = _run_call_llm(agent, mock_activity_ctx)
            assert mock_llm.generate.call_count == 1
            # The original messages from _patch_activity_deps survive unchanged.
            called_messages = mock_llm.generate.call_args.kwargs["messages"]
            assert called_messages == [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "what is dapr?"},
            ]
            assert result == {"role": "assistant", "content": "from-llm"}

    def test_in_place_mutation_does_not_leak_into_llm_call(
        self, mock_llm, mock_activity_ctx
    ):
        """A hook that mutates ctx.payload in-place but returns Proceed must NOT
        affect the actual LLM call — only Mutate(payload=...) is honored."""

        def sneaky_hook(ctx):
            # Try to mutate the live payload AND the nested messages list
            ctx.payload["model"] = "rewritten-by-sneaky-hook"
            ctx.payload["messages"].append({"role": "system", "content": "injected"})
            return Proceed()

        agent = _make_agent(mock_llm, Hooks(before_llm_call=[sneaky_hook]))
        _run_call_llm(agent, mock_activity_ctx)

        kwargs = mock_llm.generate.call_args.kwargs
        # The mutation must NOT have reached the LLM client — neither the
        # model swap nor the appended message should be visible.
        assert kwargs.get("model") != "rewritten-by-sneaky-hook"
        assert not any(m.get("content") == "injected" for m in kwargs["messages"])

    def test_skip_short_circuits_and_returns_canned_result(
        self, mock_llm, mock_activity_ctx
    ):
        def hook(_):
            return Skip(result="cached answer")

        agent = _make_agent(mock_llm, Hooks(before_llm_call=[hook]))
        result = _run_call_llm(agent, mock_activity_ctx)

        assert mock_llm.generate.call_count == 0
        assert result == {"role": "assistant", "content": "cached answer"}

    def test_deny_synthesizes_blocked_assistant_message(
        self, mock_llm, mock_activity_ctx
    ):
        def hook(_):
            return Deny(reason="off-topic")

        agent = _make_agent(mock_llm, Hooks(before_llm_call=[hook]))
        result = _run_call_llm(agent, mock_activity_ctx)

        assert mock_llm.generate.call_count == 0
        assert result["role"] == "assistant"
        assert "LLM call blocked: off-topic" in result["content"]

    def test_require_approval_raises_notimplementederror(
        self, mock_llm, mock_activity_ctx
    ):
        def hook(_):
            return RequireApproval(timeout_seconds=30)

        agent = _make_agent(mock_llm, Hooks(before_llm_call=[hook]))
        with pytest.raises(NotImplementedError) as exc_info:
            _run_call_llm(agent, mock_activity_ctx)

        msg = str(exc_info.value)
        # Error message must reference non-determinism so callers know why
        assert "non-deterministic" in msg or "deterministic" in msg
        assert "before_tool_call" in msg

    def test_first_nonproceed_decision_wins(self, mock_llm, mock_activity_ctx):
        def hook_a(_):
            return Mutate(payload={"messages": [{"role": "user", "content": "from A"}]})

        def hook_b_unreached(_):
            return Skip(result="should not be reached")

        agent = _make_agent(mock_llm, Hooks(before_llm_call=[hook_a, hook_b_unreached]))
        _run_call_llm(agent, mock_activity_ctx)

        # Mutate won — LLM was called with hook_a's payload, not Skip-ed
        assert mock_llm.generate.call_count == 1
        called_messages = mock_llm.generate.call_args.kwargs["messages"]
        assert called_messages == [{"role": "user", "content": "from A"}]

    def test_proceed_then_mutate_chains(self, mock_llm, mock_activity_ctx):
        """First hook returns Proceed → second hook still runs."""
        new_messages = [{"role": "user", "content": "from B"}]

        def hook_a(_):
            return Proceed()

        def hook_b(_):
            return Mutate(payload={"messages": new_messages})

        agent = _make_agent(mock_llm, Hooks(before_llm_call=[hook_a, hook_b]))
        _run_call_llm(agent, mock_activity_ctx)

        called_messages = mock_llm.generate.call_args.kwargs["messages"]
        assert called_messages == new_messages


class TestAfterLLMCallHook:
    def test_mutate_replaces_assistant_message(self, mock_llm, mock_activity_ctx):
        replacement = {"role": "assistant", "content": "rewritten by after-hook"}

        def hook(_, _msg):
            return Mutate(payload=replacement)

        agent = _make_agent(mock_llm, Hooks(after_llm_call=[hook]))
        result = _run_call_llm(agent, mock_activity_ctx)

        assert mock_llm.generate.call_count == 1
        assert result == replacement

    def test_after_hook_receives_built_message(self, mock_llm, mock_activity_ctx):
        captured: list = []

        def hook(ctx, msg):
            captured.append((ctx, msg))
            return None

        agent = _make_agent(mock_llm, Hooks(after_llm_call=[hook]))
        result = _run_call_llm(agent, mock_activity_ctx)

        assert len(captured) == 1
        seen_ctx, seen_msg = captured[0]
        assert seen_ctx.step_kind == "llm"
        assert seen_msg == {"role": "assistant", "content": "from-llm"}
        # Hook returned None → message unchanged
        assert result == {"role": "assistant", "content": "from-llm"}

    def test_after_hook_in_place_mutation_does_not_leak(
        self, mock_llm, mock_activity_ctx
    ):
        """A hook that mutates the assistant_message dict in-place but returns
        Proceed must NOT affect what gets persisted — only Mutate is honored."""

        def sneaky_hook(_, msg):
            msg["content"] = "secretly rewritten by sneaky hook"
            return Proceed()

        agent = _make_agent(mock_llm, Hooks(after_llm_call=[sneaky_hook]))
        result = _run_call_llm(agent, mock_activity_ctx)

        # The mutation must not have leaked back into the saved message.
        assert result == {"role": "assistant", "content": "from-llm"}

    def test_proceed_means_no_op(self, mock_llm, mock_activity_ctx):
        def hook(_, _msg):
            return Proceed()

        agent = _make_agent(mock_llm, Hooks(after_llm_call=[hook]))
        result = _run_call_llm(agent, mock_activity_ctx)

        assert result == {"role": "assistant", "content": "from-llm"}

    def test_skip_on_after_path_is_noop(self, mock_llm, mock_activity_ctx):
        """Skip / Deny / RequireApproval are documented no-ops on after_llm_call."""

        def hook(_, _msg):
            return Skip(result="ignored")

        agent = _make_agent(mock_llm, Hooks(after_llm_call=[hook]))
        result = _run_call_llm(agent, mock_activity_ctx)

        assert result == {"role": "assistant", "content": "from-llm"}


class TestBeforeAfterCombined:
    def test_mutate_kwargs_then_mutate_response(self, mock_llm, mock_activity_ctx):
        before_messages = [{"role": "user", "content": "BEFORE"}]
        after_message = {"role": "assistant", "content": "AFTER"}

        def before_hook(_):
            return Mutate(payload={"messages": before_messages})

        def after_hook(_, _msg):
            return Mutate(payload=after_message)

        agent = _make_agent(
            mock_llm,
            Hooks(before_llm_call=[before_hook], after_llm_call=[after_hook]),
        )
        result = _run_call_llm(agent, mock_activity_ctx)

        assert mock_llm.generate.call_args.kwargs["messages"] == before_messages
        assert result == after_message

    def test_skip_short_circuit_still_runs_after_hook(
        self, mock_llm, mock_activity_ctx
    ):
        """When before-hook short-circuits with Skip, after-hooks still see the synthesized message."""
        after_seen: list = []

        def before_hook(_):
            return Skip(result="canned")

        def after_hook(_, msg):
            after_seen.append(msg)
            return None

        agent = _make_agent(
            mock_llm,
            Hooks(before_llm_call=[before_hook], after_llm_call=[after_hook]),
        )
        result = _run_call_llm(agent, mock_activity_ctx)

        # LLM was skipped, but the synthesized assistant message still flowed
        # through the after-hook chain — that's the intended behavior because
        # the after-path operates on assistant_message regardless of origin.
        assert mock_llm.generate.call_count == 0
        assert after_seen == [{"role": "assistant", "content": "canned"}]
        assert result == {"role": "assistant", "content": "canned"}
