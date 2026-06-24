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

"""Tests for config helper functions used by agents."""

import pytest

from dapr_agents.agents.configs import coerce_config_value


class TestCoerceConfigValue:
    """Tests for coerce_config_value type coercion."""

    def test_str_passthrough(self):
        assert coerce_config_value("hello", str) == "hello"

    def test_str_from_int(self):
        assert coerce_config_value(42, str) == "42"

    def test_int_from_string(self):
        assert coerce_config_value("42", int) == 42

    def test_int_from_float_string(self):
        assert coerce_config_value("10.0", int) == 10

    def test_int_already_int(self):
        assert coerce_config_value(7, int) == 7

    def test_int_invalid_raises(self):
        with pytest.raises((ValueError, TypeError)):
            coerce_config_value("not_a_number", int)

    def test_bool_true_variants(self):
        for v in ("true", "True", "1", "yes"):
            assert coerce_config_value(v, bool) is True

    def test_bool_false_variants(self):
        for v in ("false", "False", "0", "no"):
            assert coerce_config_value(v, bool) is False

    def test_bool_invalid_raises(self):
        with pytest.raises(ValueError):
            coerce_config_value("maybe", bool)

    def test_list_from_json(self):
        result = coerce_config_value('["a", "b"]', list)
        assert result == ["a", "b"]

    def test_list_wraps_single_string(self):
        result = coerce_config_value("single", list)
        assert result == ["single"]

    def test_list_already_list(self):
        result = coerce_config_value(["already"], list)
        assert result == ["already"]

    def test_dict_from_json(self):
        result = coerce_config_value('{"key": "val"}', dict)
        assert result == {"key": "val"}

    def test_dict_already_dict(self):
        result = coerce_config_value({"key": "val"}, dict)
        assert result == {"key": "val"}

    def test_dict_non_dict_json_raises(self):
        with pytest.raises(ValueError):
            coerce_config_value("[1, 2]", dict)

    def test_union_type_members(self):
        result = coerce_config_value("67", int | None)
        assert result == 67

        result = coerce_config_value(None, int | None)
        assert result is None

    def test_union_type_invalid_member_raises(self):
        with pytest.raises(ValueError):
            coerce_config_value("foobar", int | None)
