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

"""Unit tests for StructureHandler.extract_structured_response error messages.

Regression guard for the cross-process orchestration failure where a
"Extraction failed: No content found for JSON mode" error bubbled up
through two layers of nested tasks with no diagnostic context. The
enriched messages name the provider, indicate whether tool calls were
present, and point the user at the likely remediation.
"""

from unittest.mock import MagicMock

import pytest

from dapr_agents.llm.utils.structure import StructureHandler, _preview_content
from dapr_agents.types.exceptions import StructureError


def _message(content=None, tool_calls=None, refusal=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.refusal = refusal
    return msg


class TestJsonModeErrors:
    def test_empty_content_error_names_provider_and_tool_calls_state(self):
        msg = _message(content=None, tool_calls=None)
        with pytest.raises(StructureError) as exc_info:
            StructureHandler.extract_structured_response(msg, "openai", "json")
        text = str(exc_info.value)
        assert "No content found for JSON mode" in text
        assert "provider='openai'" in text
        assert "tool_calls_present=False" in text

    def test_empty_content_with_tool_calls_hints_at_root_cause(self):
        """When the model emitted tool calls instead of content, hint at it."""
        tool_calls = [MagicMock()]  # non-empty marker
        msg = _message(content="", tool_calls=tool_calls)
        with pytest.raises(StructureError) as exc_info:
            StructureHandler.extract_structured_response(msg, "openai", "json")
        text = str(exc_info.value)
        assert "tool_calls_present=True" in text

    def test_refusal_error_includes_refusal_text(self):
        msg = _message(content=None, refusal="I cannot comply.")
        with pytest.raises(StructureError) as exc_info:
            StructureHandler.extract_structured_response(msg, "openai", "json")
        assert "I cannot comply" in str(exc_info.value)


class TestFunctionCallModeErrors:
    def test_no_tool_calls_error_names_provider(self):
        msg = _message(content=None, tool_calls=None)
        with pytest.raises(StructureError) as exc_info:
            StructureHandler.extract_structured_response(msg, "openai", "function_call")
        text = str(exc_info.value)
        assert "No tool_calls found for function_call mode" in text
        assert "provider='openai'" in text
        assert "content_present=False" in text


class TestPreviewContent:
    def test_none_returns_marker(self):
        assert _preview_content(None) == "<none>"

    def test_short_string_returned_verbatim(self):
        assert _preview_content("hello") == "hello"

    def test_long_string_truncated(self):
        long_text = "x" * 500
        preview = _preview_content(long_text, limit=100)
        assert preview.startswith("x" * 100)
        assert "truncated" in preview
        assert "400 more chars" in preview

    def test_non_string_coerced_to_string(self):
        assert _preview_content(42) == "42"
