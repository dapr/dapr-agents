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
