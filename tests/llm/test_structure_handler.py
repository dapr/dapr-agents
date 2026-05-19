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

"""Tests for StructureHandler error enrichment, JSON repair, and validate_response.

The enriched error messages and JSON repair heuristics protect against the
class of orchestration failure where the LLM emits structured output that
needs minor cleanup (markdown fences, prose wrap) or where the failure
needs to surface enough context to diagnose without unwrapping multiple
nested workflow activity errors.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from dapr_agents.llm.utils.structure import StructureHandler, _repair_json_content
from dapr_agents.types.exceptions import StructureError


def _message(content=None, tool_calls=None, refusal=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.refusal = refusal
    return msg


class _SingleField(BaseModel):
    summary: str = Field(...)


class TestRepairJsonContent:
    def test_returns_empty_string_for_empty_input(self):
        assert _repair_json_content("") == ""
        assert _repair_json_content("   \n  ") == ""

    def test_returns_empty_string_when_no_json_braces_present(self):
        assert _repair_json_content("hello world") == ""

    def test_strips_markdown_code_fences_with_language_tag(self):
        text = '```json\n{"summary": "ok"}\n```'
        repaired = _repair_json_content(text)
        assert repaired == '{"summary": "ok"}'

    def test_strips_markdown_code_fences_without_language_tag(self):
        text = '```\n{"summary": "ok"}\n```'
        repaired = _repair_json_content(text)
        assert repaired == '{"summary": "ok"}'

    def test_strips_leading_prose_before_json_object(self):
        text = 'Here is your response: {"summary": "ok"} hope this helps!'
        repaired = _repair_json_content(text)
        assert repaired == '{"summary": "ok"}'

    def test_handles_array_payload(self):
        text = "Some prose [1, 2, 3] trailing text"
        repaired = _repair_json_content(text)
        assert repaired == "[1, 2, 3]"

    def test_picks_object_when_object_appears_before_array(self):
        text = '{"k": [1]} extra'
        repaired = _repair_json_content(text)
        assert repaired == '{"k": [1]}'

    def test_ignores_braces_inside_string_literals(self):
        text = '{"note": "} fake closer"} trailing'
        repaired = _repair_json_content(text)
        assert repaired == '{"note": "} fake closer"}'

    def test_returns_truncated_content_when_unbalanced(self):
        text = 'prose {"a": 1, "b":'  # unbalanced
        repaired = _repair_json_content(text)
        assert repaired
        assert repaired.startswith("{")

    def test_returns_empty_string_when_repair_yields_input_unchanged(self):
        # Already-clean braces with no fences/prose: no repair needed.
        text = '{"summary": "ok"}'
        assert _repair_json_content(text) == ""


class TestExtractStructuredResponseErrors:
    def test_json_mode_empty_content_names_provider_and_tool_call_state(self):
        msg = _message(content=None, tool_calls=None)
        with pytest.raises(StructureError) as exc_info:
            StructureHandler.extract_structured_response(msg, "openai", "json")
        text = str(exc_info.value)
        assert "No content found for structured_mode='json'" in text
        assert "provider='openai'" in text
        assert "tool_calls_present=False" in text

    def test_json_mode_empty_content_with_tool_calls_hints_at_root_cause(self):
        tool_calls = [MagicMock()]
        msg = _message(content=None, tool_calls=tool_calls)
        with pytest.raises(StructureError) as exc_info:
            StructureHandler.extract_structured_response(msg, "openai", "json")
        text = str(exc_info.value)
        assert "tool_calls_present=True" in text
        # Should suggest the symmetric mode or removing tools.
        assert "function_call" in text or "tools" in text

    def test_json_mode_refusal_includes_refusal_text(self):
        msg = _message(content=None, refusal="I cannot comply.")
        with pytest.raises(StructureError) as exc_info:
            StructureHandler.extract_structured_response(msg, "openai", "json")
        assert "I cannot comply" in str(exc_info.value)

    def test_function_call_mode_no_tool_calls_names_provider(self):
        msg = _message(content=None, tool_calls=None)
        with pytest.raises(StructureError) as exc_info:
            StructureHandler.extract_structured_response(msg, "openai", "function_call")
        text = str(exc_info.value)
        assert "No tool_calls found for structured_mode='function_call'" in text
        assert "provider='openai'" in text
        assert "content_present=False" in text

    def test_json_mode_repairs_fenced_content(self):
        msg = _message(content='```json\n{"summary": "hi"}\n```')
        result = StructureHandler.extract_structured_response(msg, "openai", "json")
        assert result == {"summary": "hi"}

    def test_json_mode_repairs_prose_wrapped_content(self):
        msg = _message(content='Here you go: {"summary": "hi"} cheers')
        result = StructureHandler.extract_structured_response(msg, "openai", "json")
        assert result == {"summary": "hi"}

    def test_json_mode_clean_content_parses_without_repair(self):
        msg = _message(content='{"summary": "hi"}')
        result = StructureHandler.extract_structured_response(msg, "openai", "json")
        assert result == {"summary": "hi"}


class TestValidateResponse:
    def test_validates_clean_json_string(self):
        result = StructureHandler.validate_response('{"summary": "ok"}', _SingleField)
        assert result.summary == "ok"

    def test_validates_dict(self):
        result = StructureHandler.validate_response({"summary": "ok"}, _SingleField)
        assert result.summary == "ok"

    def test_repairs_fenced_json_string_before_validating(self):
        result = StructureHandler.validate_response(
            '```json\n{"summary": "ok"}\n```', _SingleField
        )
        assert result.summary == "ok"

    def test_raises_structure_error_for_list_input(self):
        with pytest.raises(StructureError) as exc_info:
            StructureHandler.validate_response([], _SingleField)
        text = str(exc_info.value)
        assert "_SingleField" in text
        assert "list" in text

    def test_raises_structure_error_for_none_input(self):
        with pytest.raises(StructureError) as exc_info:
            StructureHandler.validate_response(None, _SingleField)
        assert "NoneType" in str(exc_info.value)

    def test_raises_structure_error_for_unparseable_string(self):
        with pytest.raises(StructureError) as exc_info:
            StructureHandler.validate_response("not json", _SingleField)
        text = str(exc_info.value)
        assert "_SingleField" in text
        assert "not parseable as JSON" in text
