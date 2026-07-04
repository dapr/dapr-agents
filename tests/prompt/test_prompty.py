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

"""Unit tests for PromptyHelper.normalize() environment variable resolution."""

from dapr_agents.prompt.prompty import Prompty
from dapr_agents.prompt.utils.prompty import PromptyHelper


class TestPromptyHelperNormalize:
    """Tests for PromptyHelper.normalize() with ${VAR} env references."""

    def test_legacy_default_used_when_var_unset(self, monkeypatch):
        """${VAR:default} returns the default when the env var is unset."""
        monkeypatch.delenv("DA_NORMALIZE_MISSING", raising=False)
        assert PromptyHelper.normalize("${DA_NORMALIZE_MISSING:fallback}") == "fallback"

    def test_unset_var_without_default_returns_none(self, monkeypatch):
        """${VAR} with no default returns None when env_error is False."""
        monkeypatch.delenv("DA_NORMALIZE_MISSING", raising=False)
        assert (
            PromptyHelper.normalize("${DA_NORMALIZE_MISSING}", env_error=False) is None
        )

    def test_set_var_round_trips(self, monkeypatch):
        """${VAR} returns the env value when the variable is set."""
        monkeypatch.setenv("DA_NORMALIZE_SET", "xyz")
        assert PromptyHelper.normalize("${DA_NORMALIZE_SET}") == "xyz"

    def test_env_prefixed_default_still_works(self, monkeypatch):
        """${env:NAME:default} still falls back to the default when unset."""
        monkeypatch.delenv("DA_NORMALIZE_MISSING", raising=False)
        assert PromptyHelper.normalize("${env:DA_NORMALIZE_MISSING:def}") == "def"

    def test_plain_string_unchanged(self):
        """A plain string is returned unchanged."""
        assert PromptyHelper.normalize("hello") == "hello"


class TestPromptyExtractInputVariables:
    """Tests for Prompty.extract_input_variables() return contract."""

    PROMPTY_CONTENT = """---
name: Extract Vars Test
model:
  api: chat
  configuration:
    type: openai
    name: gpt-4o
  parameters:
    max_tokens: 128
    temperature: 0.2
inputs:
  question:
    type: string
sample:
  "topic": "history"
---
system:
You are a helpful assistant.

user:
{{question}} about {{subject}}
"""

    def test_returns_flat_list_of_strings(self):
        """extract_input_variables returns a single flat list of variable names.

        Regression test: the method used to be annotated
        ``-> Tuple[List[str], List[str]]`` while it only ever returns one list,
        so callers (e.g. to_prompt_template) treat the result as ``List[str]``.
        """
        prompty = Prompty.load(self.PROMPTY_CONTENT)

        result = prompty.extract_input_variables("jinja2")

        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)
        # Content placeholders (question, subject), declared inputs (question),
        # and sample keys (topic), deduplicated into one flat list.
        assert sorted(result) == ["question", "subject", "topic"]

    def test_result_feeds_prompt_template_input_variables(self):
        """The returned list is used directly as the template input variables."""
        prompty = Prompty.load(self.PROMPTY_CONTENT)

        template = prompty.to_prompt_template("jinja2")

        assert sorted(template.input_variables) == ["question", "subject", "topic"]
