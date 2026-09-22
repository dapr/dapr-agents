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

"""Unit tests for ChatPromptHelper.format_message()."""

from dapr_agents.prompt.utils.chat import ChatPromptHelper
from dapr_agents.types.message import AssistantMessage, ToolMessage


class TestFormatMessage:
    """Tests for ChatPromptHelper.format_message()."""

    def test_formats_dict_tool_message(self):
        result = ChatPromptHelper.format_message(
            {"role": "tool", "content": "result: {x}", "tool_call_id": "call_1"},
            "f-string",
            x=42,
        )
        assert result.content == "result: 42"
        assert result.tool_call_id == "call_1"

    def test_formats_tuple_assistant_message_without_crashing(self):
        result = ChatPromptHelper.format_message(
            ("assistant", "hello {x}"), "f-string", x=42
        )
        assert result.content == "hello 42"

    def test_formats_base_message_tool_instance(self):
        tool_message = ToolMessage(content="result: {x}", tool_call_id="call_1")
        result = ChatPromptHelper.format_message(tool_message, "f-string", x=42)
        assert result.content == "result: 42"
        assert result.tool_call_id == "call_1"

    def test_formats_base_message_assistant_with_tool_calls(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ]
        assistant_message = AssistantMessage(
            content="calling {x}", tool_calls=tool_calls
        )
        result = ChatPromptHelper.format_message(assistant_message, "f-string", x=1)
        assert result.content == "calling 1"
        assert result.tool_calls[0].id == "call_1"
