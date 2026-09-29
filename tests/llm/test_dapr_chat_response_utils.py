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

from dapr_agents.llm.dapr.utils import process_dapr_chat_response
from dapr_agents.types.message import LLMChatResponse


def test_basic_message_response():
    response = {
        "id": "resp-1",
        "model": "gpt-oss",
        "object": "chat.completion",
        "created": 1700000000,
        "usage": {"total_tokens": 10},
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": "hello there"},
            }
        ],
    }
    result = process_dapr_chat_response(response)
    assert isinstance(result, LLMChatResponse)
    assert len(result.results) == 1
    candidate = result.results[0]
    assert candidate.message.content == "hello there"
    assert candidate.message.get_tool_calls() is None
    assert candidate.finish_reason == "stop"
    assert result.metadata["provider"] == "dapr"
    assert result.metadata["model"] == "gpt-oss"
    assert result.metadata["usage"] == {"total_tokens": 10}


def test_tool_calls_are_built():
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "SF"}',
                            },
                        }
                    ],
                },
            }
        ]
    }
    result = process_dapr_chat_response(response)
    msg = result.results[0].message
    tool_calls = msg.get_tool_calls()
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"city": "SF"}'
    assert getattr(msg, "function_call", None) is None


def test_malformed_tool_call_entry_is_skipped_not_fatal():
    response = {
        "choices": [
            {
                "index": 0,
                "message": {"content": None, "tool_calls": ["not-a-dict"]},
            }
        ]
    }
    result = process_dapr_chat_response(response)
    assert result.results[0].message.tool_calls == []


def test_dict_content_is_serialized_to_json_string():
    response = {
        "choices": [{"index": 0, "message": {"content": {"answer": 42, "ok": True}}}]
    }
    result = process_dapr_chat_response(response)
    content = result.results[0].message.content
    assert isinstance(content, str)
    assert '"answer": 42' in content


def test_unserializable_dict_content_falls_back_to_string_instead_of_crashing():
    response = {"choices": [{"index": 0, "message": {"content": {"bad": object()}}}]}
    result = process_dapr_chat_response(response)
    content = result.results[0].message.content
    assert isinstance(content, str)
    assert "bad" in content


def test_empty_choices_returns_no_results_but_keeps_metadata():
    response = {"id": "resp-2", "model": "gpt-oss", "choices": []}
    result = process_dapr_chat_response(response)
    assert result.results == []
    assert result.metadata["id"] == "resp-2"


def test_missing_optional_top_level_fields_default_safely():
    response = {"choices": [{"index": 0, "message": {"content": "hi"}}]}
    result = process_dapr_chat_response(response)
    assert result.metadata["id"] is None
    assert result.metadata["model"] is None
    assert result.metadata["usage"] == {}
    assert isinstance(result.metadata["created"], int)
