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

"""Unit tests for StringPromptTemplate.format_prompt() method."""

import pytest
from dapr_agents.prompt.string import StringPromptTemplate


class TestStringPromptTemplateFormatPrompt:
    """Tests for StringPromptTemplate.format_prompt() method."""

    def test_format_prompt_replaces_variable(self):
        """A simple placeholder is replaced with the provided value."""
        template = StringPromptTemplate.from_template("Hello {name}")
        assert template.format_prompt(name="Alice") == "Hello Alice"

    def test_format_prompt_with_escaped_braces(self):
        """Escaped braces in a template do not trigger a missing-variable error.

        A literal JSON example (using {{ }}) is a common prompting pattern; the
        escaped content must not be treated as a required variable.
        """
        template = StringPromptTemplate.from_template(
            'Respond as JSON like {{"a": 1}}. Q: {question}'
        )
        assert template.input_variables == ["question"]
        result = template.format_prompt(question="x")
        assert result == 'Respond as JSON like {"a": 1}. Q: x'

    def test_from_template_ignores_escaped_braces_in_variables(self):
        """Escaped braces are excluded from the detected input variables."""
        template = StringPromptTemplate.from_template("{{literal}} and {real}")
        assert template.input_variables == ["real"]

    def test_format_prompt_raises_on_missing_variable(self):
        """A genuinely missing variable still raises a ValueError."""
        template = StringPromptTemplate.from_template("Hello {name}")
        with pytest.raises(ValueError, match="Missing required variables"):
            template.format_prompt()
