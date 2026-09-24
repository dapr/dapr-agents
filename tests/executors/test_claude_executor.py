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

"""Tests for ``ClaudeAgentExecutor`` against a scripted SDK client (no network)."""

import os
import uuid

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ClaudeSDKError,
    CLINotFoundError,
    HookMatcher,
    InMemorySessionStore,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk.types import DeferredToolUse  # noqa: E402

import dapr_agents.agents.executors.claude as claude_module  # noqa: E402
from dapr_agents.agents.executors import (  # noqa: E402
    CONTEXT_TOOL_DECISIONS,
    ClaudeAgentExecutorConfig,
    ExecutorBinding,
    ToolCallDecision,
)
from dapr_agents.agents.executors.claude import (  # noqa: E402
    CONTINUE_PROMPT,
    STALE_CALL_REASON,
    ClaudeAgentExecutor,
    session_uuid,
)
from dapr_agents.agents.executors.claude_config import (  # noqa: E402
    FIELD_OPTION_KEYS,
    RESERVED_OPTION_KEYS,
)
from dapr_agents.hooks import RequireApproval  # noqa: E402
from dapr_agents.tool import tool  # noqa: E402
from tests.executors.claude_fakes import (  # noqa: E402
    SESSION_ID,
    FakeClaudeClient,
    result_message,
)


@tool
def transfer(to: str, amount: int) -> str:
    """Transfer money."""
    return f"sent {amount} to {to}"


TOOL_NAME = f"mcp__dapr__{transfer.name}"


def _require_approval(ctx):
    return RequireApproval(reason="money") if ctx.step_name == transfer.name else None


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    FakeClaudeClient.reset()
    monkeypatch.setattr(claude_module, "ClaudeSDKClient", FakeClaudeClient)
    yield FakeClaudeClient


@pytest.fixture
def cwd(tmp_path):
    return str(tmp_path / "claude-cwd")


def _executor(cwd, **kwargs):
    return ClaudeAgentExecutor(ClaudeAgentExecutorConfig(cwd=cwd, **kwargs))


async def _collect(executor, prompt="go", **kwargs):
    return [event async for event in executor.run(prompt, **kwargs)]


def _decisions(call_id, approved=True):
    return {
        CONTEXT_TOOL_DECISIONS: {call_id: ToolCallDecision(call_id, approved).to_dict()}
    }


async def _seed(store, executor, *entries):
    key = {"project_key": executor.project_key, "session_id": SESSION_ID}
    await store.append(key, list(entries))


def _deferred_entry(call_id="t9"):
    return {
        "type": "attachment",
        "uuid": f"a-{call_id}",
        "attachment": {
            "type": "hook_deferred_tool",
            "toolUseID": call_id,
            "toolName": TOOL_NAME,
            "toolInput": {"to": "b"},
        },
    }


FULL_TURN = [
    SystemMessage(subtype="init", data={"session_id": SESSION_ID}),
    StreamEvent(
        uuid="1",
        session_id=SESSION_ID,
        event={
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hi"},
        },
    ),
    AssistantMessage(
        content=[ToolUseBlock(id="t1", name=TOOL_NAME, input={"to": "a", "amount": 1})],
        model="m",
        session_id=SESSION_ID,
    ),
    UserMessage(
        content=[
            ToolResultBlock(tool_use_id="t1", content=[{"type": "text", "text": "ok"}])
        ]
    ),
    AssistantMessage(content=[TextBlock(text="Sent.")], model="m"),
    result_message(result="Sent."),
]


class TestConstruction:
    def test_session_uuid(self):
        assert session_uuid(SESSION_ID) == SESSION_ID
        derived = session_uuid("my-session")
        assert uuid.UUID(derived) and derived == session_uuid("my-session")

    def test_cwd_defaults_to_process_cwd(self):
        executor = ClaudeAgentExecutor()
        assert executor.cwd == os.getcwd()
        assert executor.config.cwd == os.getcwd()
        assert executor.project_key

    def test_supports_tool_approval(self):
        assert ClaudeAgentExecutor.supports_tool_approval is True

    def test_bind_returns_new_bound_executor(self, cwd):
        executor = _executor(cwd)
        bound = executor.bind(
            ExecutorBinding(agent_name="a", system_prompt="SYS", tools=(transfer,))
        )
        assert bound is not executor
        assert bound.config.system_prompt == "SYS"
        assert bound.config.tools == (transfer,)
        assert executor.config.system_prompt is None


class TestFreshRun:
    async def test_maps_full_turn(self, cwd):
        FakeClaudeClient.reset(FULL_TURN)
        events = await _collect(_executor(cwd, tools=[transfer]), "go")
        assert [e.type for e in events] == [
            "session",
            "text_delta",
            "tool_call",
            "tool_result",
            "message",
            "session",
            "complete",
        ]
        assert events[-1].content == {"role": "assistant", "content": "Sent."}
        assert events[-1].session_id == SESSION_ID
        assert FakeClaudeClient.last().prompt == "go"

    async def test_options_for_fresh_run(self, cwd):
        FakeClaudeClient.reset([result_message()])
        executor = _executor(
            cwd,
            model="claude-x",
            tools=[transfer],
            allowed_tools=["mcp__github"],
            disallowed_tools=["Bash"],
            env={"ANTHROPIC_API_KEY": "k"},
            permission_mode="default",
            extra_options={"effort": "low"},
        )
        await _collect(executor)
        options = FakeClaudeClient.last().options
        assert options.model == "claude-x"
        assert options.tools == []
        assert options.setting_sources == []
        assert options.allowed_tools == [TOOL_NAME, "mcp__github"]
        assert options.disallowed_tools == ["Bash"]
        assert set(options.mcp_servers) == {"dapr"}
        assert options.hooks is None
        assert options.env["ANTHROPIC_API_KEY"] == "k"
        assert options.permission_mode == "default"
        assert options.effort == "low"
        assert options.cwd == cwd and os.path.isdir(cwd)
        assert options.resume is None
        assert uuid.UUID(options.session_id)

    async def test_every_explicit_option_is_protected_from_extra_options(
        self, cwd, monkeypatch
    ):
        passed = {}
        real_options = claude_module.ClaudeAgentOptions

        def recording_options(**kwargs):
            passed.update(kwargs)
            return real_options(**kwargs)

        monkeypatch.setattr(claude_module, "ClaudeAgentOptions", recording_options)
        FakeClaudeClient.reset([result_message()])
        await _collect(_executor(cwd))
        protected = RESERVED_OPTION_KEYS | set(FIELD_OPTION_KEYS)
        assert set(passed) <= protected

    async def test_builtin_tools_none_keeps_cli_default(self, cwd):
        FakeClaudeClient.reset([result_message()])
        await _collect(_executor(cwd, builtin_tools=None))
        assert FakeClaudeClient.last().options.tools is None

    async def test_gate_owns_dapr_tools_when_hooks_are_set(self, cwd):
        FakeClaudeClient.reset([result_message()])
        extra = HookMatcher(matcher="Bash", hooks=[])
        executor = _executor(
            cwd,
            tools=[transfer],
            before_tool_call=[_require_approval],
            hooks={"PreToolUse": [extra], "Stop": []},
        )
        await _collect(executor)
        options = FakeClaudeClient.last().options
        assert options.allowed_tools == []
        pre_tool_use = options.hooks["PreToolUse"]
        assert pre_tool_use[1] is extra
        gate = pre_tool_use[0].hooks[0]
        out = await gate({"tool_name": TOOL_NAME, "tool_input": {}}, "t9", None)
        assert out["hookSpecificOutput"]["permissionDecision"] == "defer"
        assert options.hooks["Stop"] == []

    async def test_non_uuid_session_starts_derived_session(self, cwd):
        FakeClaudeClient.reset([result_message()])
        store = InMemorySessionStore()
        await _collect(_executor(cwd, session_store=store), session_id="chat-1")
        options = FakeClaudeClient.last().options
        assert options.resume is None
        assert options.session_id == session_uuid("chat-1")
        assert options.session_store is store


class TestHostIsolation:
    """The CLI must not load the host user's Claude Code context."""

    async def test_isolated_by_default(self, cwd):
        FakeClaudeClient.reset([result_message()])
        await _collect(_executor(cwd, env={"ANTHROPIC_API_KEY": "k"}))
        options = FakeClaudeClient.last().options
        assert options.strict_mcp_config is True
        assert options.env == {
            "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "ANTHROPIC_API_KEY": "k",
        }

    async def test_explicit_env_overrides_isolation(self, cwd):
        FakeClaudeClient.reset([result_message()])
        await _collect(_executor(cwd, env={"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "0"}))
        env = FakeClaudeClient.last().options.env
        assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "0"
        assert env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"

    async def test_setting_sources_keep_claude_md_files(self, cwd):
        FakeClaudeClient.reset([result_message()])
        await _collect(_executor(cwd, setting_sources=["project"]))
        env = FakeClaudeClient.last().options.env
        assert "CLAUDE_CODE_DISABLE_CLAUDE_MDS" not in env
        assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"

    async def test_opt_out_passes_env_through(self, cwd):
        FakeClaudeClient.reset([result_message()])
        await _collect(
            _executor(cwd, isolate_host_config=False, env={"ANTHROPIC_API_KEY": "k"})
        )
        options = FakeClaudeClient.last().options
        assert options.strict_mcp_config is False
        assert options.env == {"ANTHROPIC_API_KEY": "k"}


class TestPause:
    async def test_deferred_call_pauses_with_approval_details(self, cwd):
        deferred = DeferredToolUse(id="t9", name=TOOL_NAME, input={"to": "b"})

        class GateCallingClient(FakeClaudeClient):
            async def receive_response(self):
                gate = self.options.hooks["PreToolUse"][0].hooks[0]
                await gate({"tool_name": TOOL_NAME, "tool_input": {"to": "b"}}, "t9", 0)
                yield result_message(
                    result="", stop_reason="tool_deferred", deferred_tool_use=deferred
                )

        claude_module.ClaudeSDKClient = GateCallingClient
        executor = _executor(
            cwd, tools=[transfer], before_tool_call=[_require_approval]
        )
        events = await _collect(executor, session_id=SESSION_ID)
        paused = events[-1]
        assert paused.type == "paused"
        assert paused.content == {
            "tool_call_id": "t9",
            "name": transfer.name,
            "arguments": {"to": "b"},
            "approval": {
                "timeout_seconds": None,
                "instructions": None,
                "reason": "money",
            },
            "source": "local",
        }


class TestResume:
    async def test_existing_session_resumes_with_prompt(self, cwd):
        store = InMemorySessionStore()
        executor = _executor(cwd, session_store=store)
        await _seed(
            store,
            executor,
            {"type": "user", "uuid": "u1"},
            {"type": "cost-state", "totalCostUSD": 0.004},
        )
        FakeClaudeClient.reset([result_message(total_cost_usd=0.01)])
        events = await _collect(executor, "next", session_id=SESSION_ID)
        client = FakeClaudeClient.last()
        assert client.prompt == "next"
        assert client.options.resume == SESSION_ID
        assert client.options.session_id is None
        assert events[-1].metadata["cost_usd"] == pytest.approx(0.006)

    async def test_decision_resumes_pending_call_without_prompt(self, cwd):
        store = InMemorySessionStore()
        executor = _executor(
            cwd,
            tools=[transfer],
            before_tool_call=[_require_approval],
            session_store=store,
        )
        await _seed(store, executor, {"type": "user", "uuid": "u1"}, _deferred_entry())
        FakeClaudeClient.reset([result_message(result="approved!")])
        events = await _collect(
            executor, "ignored", session_id=SESSION_ID, context=_decisions("t9")
        )
        client = FakeClaudeClient.last()
        assert events[-1].type == "complete"
        assert client.prompt is None
        assert client.options.resume == SESSION_ID
        gate = client.options.hooks["PreToolUse"][0].hooks[0]
        out = await gate({"tool_name": TOOL_NAME, "tool_input": {}}, "t9", None)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"

    async def test_retry_after_finished_resume_replays_answer(self, cwd):
        store = InMemorySessionStore()
        executor = _executor(cwd, session_store=store)
        await _seed(
            store,
            executor,
            _deferred_entry(),
            {
                "type": "user",
                "uuid": "u2",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "t9"}]},
            },
            {
                "type": "assistant",
                "uuid": "u3",
                "message": {"id": "m1", "content": [{"type": "text", "text": "Sent."}]},
            },
        )
        events = await _collect(
            executor, session_id=SESSION_ID, context=_decisions("t9")
        )
        assert FakeClaudeClient.instances == []
        assert [e.type for e in events] == ["complete"]
        assert events[0].content == {"role": "assistant", "content": "Sent."}
        assert events[0].metadata == {"replayed": True, "cost_usd": 0.0}

    async def test_retry_after_crash_mid_resume_continues(self, cwd):
        # The approved tool ran but no final answer was written; a resume
        # without a prompt would wait for input, so the turn is continued.
        store = InMemorySessionStore()
        executor = _executor(cwd, session_store=store)
        await _seed(
            store,
            executor,
            {
                "type": "assistant",
                "uuid": "u1",
                "message": {"id": "m0", "content": [{"type": "text", "text": "Hi"}]},
            },
            _deferred_entry(),
            {
                "type": "user",
                "uuid": "u2",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "t9"}]},
            },
        )
        FakeClaudeClient.reset([result_message(result="Sent")])
        events = await _collect(
            executor, session_id=SESSION_ID, context=_decisions("t9")
        )
        client = FakeClaudeClient.last()
        assert client.options.resume == SESSION_ID
        assert client.prompts == [CONTINUE_PROMPT]
        assert events[-1].type == "complete"

    async def test_stale_deferred_call_is_denied_before_new_task(self, cwd):
        store = InMemorySessionStore()
        executor = _executor(cwd, session_store=store, tools=[transfer])
        await _seed(store, executor, _deferred_entry())
        FakeClaudeClient.reset([result_message(result="Hello")])
        FakeClaudeClient.responses = [[result_message(result="denied")]]

        events = await _collect(executor, "next task", session_id=SESSION_ID)

        client = FakeClaudeClient.last()
        assert client.options.resume == SESSION_ID
        assert client.prompts == ["next task"]
        assert events[-1].content["content"] == "Hello"
        gate = client.options.hooks["PreToolUse"][0].hooks[0]
        output = await gate({"tool_name": TOOL_NAME, "tool_input": {}}, "t9", None)
        decision = output["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"
        assert decision["permissionDecisionReason"] == STALE_CALL_REASON

    async def test_decisions_without_session_id_is_an_error(self, cwd):
        events = await _collect(_executor(cwd), context=_decisions("t9"))
        assert events[0].type == "error"
        assert "without a session_id" in events[0].content
        assert events[0].session_id is None
        assert FakeClaudeClient.instances == []

    async def test_decisions_for_unknown_session_is_an_error(self, cwd):
        executor = _executor(cwd, session_store=InMemorySessionStore())
        events = await _collect(
            executor, session_id=SESSION_ID, context=_decisions("t")
        )
        assert events[0].type == "error"
        assert f"No Claude session {SESSION_ID}" in events[0].content

    async def test_local_disk_session_lookup_without_store(self, cwd, monkeypatch):
        monkeypatch.setattr(
            claude_module, "get_session_info", lambda sid, directory: object()
        )
        FakeClaudeClient.reset([result_message()])
        await _collect(_executor(cwd), session_id=SESSION_ID)
        assert FakeClaudeClient.last().options.resume == SESSION_ID

    async def test_store_failure_while_planning_is_an_error(self, cwd):
        class BrokenStore:
            async def load(self, key):
                raise RuntimeError("state store down")

        executor = _executor(cwd, session_store=BrokenStore())
        events = await _collect(executor, session_id=SESSION_ID)
        assert [e.type for e in events] == ["error"]
        assert "state store down" in events[0].content


class TestErrors:
    async def test_error_result_then_sdk_exception(self, cwd):
        FakeClaudeClient.reset(
            [
                result_message(
                    is_error=True,
                    subtype="error_max_turns",
                    terminal_reason="max_turns",
                    errors=["Reached maximum number of turns (1)"],
                ),
                ClaudeSDKError("Command failed with exit code 1"),
            ]
        )
        events = await _collect(_executor(cwd))
        assert events[-1].type == "error"
        assert "error_max_turns" in events[-1].content
        assert "exit code" not in events[-1].content

    async def test_non_sdk_exception_after_result_keeps_result(self, cwd):
        FakeClaudeClient.reset([result_message(), RuntimeError("shutdown hiccup")])
        events = await _collect(_executor(cwd))
        assert events[-1].type == "complete"

    async def test_failure_without_result_includes_stderr(self, cwd):
        FakeClaudeClient.reset([ClaudeSDKError("Command failed")])
        FakeClaudeClient.stderr_lines = ["", "Session ID is already in use.\n"]
        events = await _collect(_executor(cwd))
        assert events[-1].type == "error"
        assert "Command failed" in events[-1].content
        assert "stderr: Session ID is already in use." in events[-1].content

    async def test_missing_cli_gets_install_hint(self, cwd):
        FakeClaudeClient.reset()
        FakeClaudeClient.enter_error = CLINotFoundError("Claude Code not found")
        events = await _collect(_executor(cwd))
        assert events[-1].type == "error"
        assert "cli_path" in events[-1].content

    async def test_stream_without_result_is_an_error(self, cwd):
        FakeClaudeClient.reset(
            [SystemMessage(subtype="init", data={"session_id": SESSION_ID})]
        )
        events = await _collect(_executor(cwd))
        assert events[-1].type == "error"
        assert "without a result message" in events[-1].content


class TestGetSession:
    async def test_reads_from_store(self, cwd):
        store = InMemorySessionStore()
        executor = _executor(cwd, session_store=store)
        await _seed(
            store,
            executor,
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "sessionId": SESSION_ID,
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "hi"},
            },
        )
        session = await executor.get_session(SESSION_ID)
        assert session["session_id"] == SESSION_ID
        assert session["messages"][0]["uuid"] == "u1"
        assert session["metadata"]["first_prompt"] == "hi"

    async def test_missing_session_is_none(self, cwd):
        executor = _executor(cwd, session_store=InMemorySessionStore())
        assert await executor.get_session(SESSION_ID) is None

    async def test_reads_local_disk_without_store(self, cwd, monkeypatch):
        monkeypatch.setattr(
            claude_module, "get_session_info", lambda sid, directory: None
        )
        monkeypatch.setattr(
            claude_module, "get_session_messages", lambda sid, directory: []
        )
        assert await _executor(cwd).get_session("chat-1") is None
