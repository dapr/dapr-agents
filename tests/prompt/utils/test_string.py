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

"""Unit tests for StringPromptHelper.extract_variables()."""

from typing import List, get_type_hints

import pytest

from dapr_agents.prompt.utils.string import StringPromptHelper


class TestStringPromptHelperExtractVariables:
    """Tests for StringPromptHelper.extract_variables() return value and type."""

    def test_extract_variables_fstring_returns_list_of_names(self):
        """f-string templates return a list of variable names, not a dict."""
        result = StringPromptHelper.extract_variables(
            "Hello {name}, you are a {role}.", "f-string"
        )
        assert isinstance(result, list)
        assert sorted(result) == ["name", "role"]

    def test_extract_variables_jinja2_returns_list_of_names(self):
        """jinja2 templates return a list of variable names, not a dict."""
        result = StringPromptHelper.extract_variables(
            "Hello {{ name }}, you are a {{ role }}.", "jinja2"
        )
        assert isinstance(result, list)
        assert sorted(result) == ["name", "role"]

    def test_extract_variables_no_variables_returns_empty_list(self):
        """A template without placeholders returns an empty list."""
        assert (
            StringPromptHelper.extract_variables("no variables here", "f-string") == []
        )

    def test_extract_variables_unsupported_format_raises(self):
        """An unknown template format raises ValueError."""
        with pytest.raises(ValueError):
            StringPromptHelper.extract_variables("{name}", "unsupported")

    def test_extract_variables_return_annotation_is_list(self):
        """The return annotation matches the List[str] the callers rely on."""
        hints = get_type_hints(StringPromptHelper.extract_variables)
        assert hints["return"] == List[str]
