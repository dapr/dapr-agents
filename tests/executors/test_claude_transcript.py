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

"""Tests for the pure transcript helpers used to plan Claude resumes."""

from dapr_agents.agents.executors.claude_transcript import (
    DeferredCall,
    final_assistant_text,
    last_total_cost,
    pending_deferred_call,
)


def _deferred(call_id, name="mcp__dapr__pay", tool_input=None):
    return {
        "type": "attachment",
        "attachment": {
            "type": "hook_deferred_tool",
            "toolUseID": call_id,
            "toolName": name,
            "toolInput": tool_input if tool_input is not None else {"to": "a"},
        },
    }


def _tool_result(call_id):
    return {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "tool_use_id": call_id}]},
    }


def _assistant(message_id, *texts, tool_use=False):
    content = [{"type": "text", "text": t} for t in texts]
    if tool_use:
        content.append({"type": "tool_use", "id": "x"})
    return {"type": "assistant", "message": {"id": message_id, "content": content}}


class TestPendingDeferredCall:
    def test_none_without_deferrals(self):
        assert pending_deferred_call([_assistant("m", "hi")]) is None

    def test_returns_unresolved_deferral(self):
        pending = pending_deferred_call([_deferred("t1")])
        assert pending == DeferredCall("t1", "mcp__dapr__pay", {"to": "a"})

    def test_resolved_by_tool_result(self):
        assert pending_deferred_call([_deferred("t1"), _tool_result("t1")]) is None

    def test_other_tool_result_does_not_resolve(self):
        pending = pending_deferred_call([_deferred("t1"), _tool_result("t2")])
        assert pending is not None and pending.tool_call_id == "t1"

    def test_latest_deferral_wins(self):
        entries = [_deferred("t1"), _tool_result("t1"), _deferred("t2")]
        assert pending_deferred_call(entries).tool_call_id == "t2"

    def test_ignores_malformed_entries(self):
        entries = [
            {"type": "attachment", "attachment": "bad"},
            {"type": "attachment", "attachment": {"type": "other"}},
            _deferred("", tool_input="not-a-dict"),
            {"type": "user", "message": "text"},
        ]
        assert pending_deferred_call(entries) is None

    def test_non_dict_tool_input_becomes_empty(self):
        pending = pending_deferred_call([_deferred("t1", tool_input=["x"])])
        assert pending.arguments == {}


class TestFinalAssistantText:
    def test_joins_blocks_of_the_last_message(self):
        entries = [
            {"type": "user", "message": {"content": "q"}},
            _assistant("m1", "old"),
            _assistant("m2", "Hello "),
            {"type": "cost-state"},
            _assistant("m2", "world"),
        ]
        assert final_assistant_text(entries) == "Hello world"

    def test_none_when_the_run_did_not_finish(self):
        entries = [_assistant("m1", "before"), _tool_result("t1")]
        assert final_assistant_text(entries) is None

    def test_none_when_the_last_message_requested_a_tool(self):
        # A crash after "let me check" + tool_use must not look finished.
        entries = [
            _tool_result("t1"),
            _assistant("m2", "Let me check"),
            _assistant("m2", tool_use=True),
        ]
        assert final_assistant_text(entries) is None

    def test_none_without_text(self):
        assert final_assistant_text([_assistant("m1", tool_use=True)]) is None
        assert final_assistant_text([]) is None


class TestLastTotalCost:
    def test_reads_last_cost_state(self):
        entries = [
            {"type": "cost-state", "totalCostUSD": 0.1},
            {"type": "cost-state", "totalCostUSD": 2},
        ]
        assert last_total_cost(entries) == 2.0

    def test_none_when_absent_or_invalid(self):
        assert last_total_cost([]) is None
        assert last_total_cost([{"type": "cost-state", "totalCostUSD": "x"}]) is None
