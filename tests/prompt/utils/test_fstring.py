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

"""Unit tests for f-string template variable extraction and rendering."""

from dapr_agents.prompt.utils.fstring import (
    extract_fstring_variables,
    render_fstring_template,
)


class TestExtractFstringVariables:
    """Tests for extract_fstring_variables()."""

    def test_extracts_simple_variable(self):
        """A single placeholder is returned as a variable."""
        assert extract_fstring_variables("Hello {name}") == ["name"]

    def test_extracts_multiple_variables_in_sentence(self):
        """Placeholders embedded in text are all extracted, in order."""
        assert extract_fstring_variables("Hi {name}, you are {age}") == [
            "name",
            "age",
        ]

    def test_preserves_duplicate_variables(self):
        """Repeated placeholders are preserved (not de-duplicated)."""
        assert extract_fstring_variables("{a} {b} {a}") == ["a", "b", "a"]

    def test_ignores_escaped_braces(self):
        """Escaped braces ({{ and }}) are literal characters, not variables."""
        assert extract_fstring_variables("{{literal}}") == []

    def test_mixes_escaped_braces_and_real_variables(self):
        """A literal JSON example does not leak into the extracted variables."""
        template = 'JSON: {{"k": "v"}} and {q}'
        assert extract_fstring_variables(template) == ["q"]

    def test_ignores_format_spec(self):
        """A format spec is dropped, matching str.format field names."""
        assert extract_fstring_variables("{value:>10}") == ["value"]

    def test_supports_attribute_and_index_access(self):
        """Attribute and index access field names are preserved."""
        assert extract_fstring_variables("{obj.attr} {arr[0]}") == [
            "obj.attr",
            "arr[0]",
        ]

    def test_returns_empty_for_no_placeholders(self):
        """Plain text without placeholders yields no variables."""
        assert extract_fstring_variables("no placeholders here") == []


class TestRenderFstringTemplate:
    """Tests for render_fstring_template()."""

    def test_renders_escaped_braces_as_literals(self):
        """Escaped braces render as literal braces and need no variable."""
        template = 'Respond as JSON like {{"answer": "..."}}. Q: {question}'
        assert (
            render_fstring_template(template, question="What is 2+2?")
            == 'Respond as JSON like {"answer": "..."}. Q: What is 2+2?'
        )

    def test_extract_and_render_agree_on_required_variables(self):
        """Every variable render needs is reported by extract, and no extras."""
        template = 'Respond as JSON like {{"answer": "..."}}. Q: {question}'
        variables = extract_fstring_variables(template)
        assert variables == ["question"]
        # Providing exactly the extracted variables renders without error.
        rendered = render_fstring_template(
            template, **{name: "x" for name in variables}
        )
        assert rendered == 'Respond as JSON like {"answer": "..."}. Q: x'
