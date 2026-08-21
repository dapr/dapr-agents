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

from enum import Enum
from types import SimpleNamespace

import pytest

from dapr_agents.utils.config import (
    ConfigFieldDescriptor,
    apply_config_map,
    apply_config_update,
    coerce_config_value,
    normalize_config_key,
    process_config_update,
)


# ---------------------------------------------------------------------------
# Config helper tests
# ---------------------------------------------------------------------------


class FakeEnum(Enum):
    """A simple enum for testing purposes."""

    OPTION_A = "option_a"
    OPTION_B = "option_b"
    OPTION_C = "option_c"


class TestCoerceConfigValue:
    """Tests for coerce_config_value type coercion."""

    def test_coerce_config_value_str_passthrough(self):
        assert coerce_config_value("hello", str) == "hello"

    def test_coerce_config_value_str_from_int(self):
        assert coerce_config_value(42, str) == "42"

    def test_coerce_config_value_int_from_string(self):
        assert coerce_config_value("42", int) == 42

    def test_coerce_config_value_int_from_float_string(self):
        assert coerce_config_value("10.0", int) == 10

    def test_coerce_config_value_int_already_int(self):
        assert coerce_config_value(7, int) == 7

    def test_coerce_config_value_int_invalid_raises(self):
        with pytest.raises((ValueError, TypeError)):
            coerce_config_value("not_a_number", int)

    def test_coerce_config_value_float_from_string(self):
        assert coerce_config_value("10.5", float) == 10.5

    def test_coerce_config_value_bool_true_variants(self):
        for v in ("true", "True", "1", "yes"):
            assert coerce_config_value(v, bool) is True

    def test_coerce_config_value_bool_false_variants(self):
        for v in ("false", "False", "0", "no"):
            assert coerce_config_value(v, bool) is False

    def test_coerce_config_value_bool_invalid_raises(self):
        with pytest.raises(ValueError):
            coerce_config_value("maybe", bool)

    def test_coerce_config_value_list_from_json(self):
        result = coerce_config_value('["a", "b"]', list)
        assert result == ["a", "b"]

    def test_coerce_config_value_list_wraps_single_string(self):
        result = coerce_config_value("single", list)
        assert result == ["single"]

    def test_coerce_config_value_list_already_list(self):
        result = coerce_config_value(["already"], list)
        assert result == ["already"]

    def test_coerce_config_value_dict_from_json(self):
        result = coerce_config_value('{"key": "val"}', dict)
        assert result == {"key": "val"}

    def test_coerce_config_value_dict_already_dict(self):
        result = coerce_config_value({"key": "val"}, dict)
        assert result == {"key": "val"}

    def test_coerce_config_value_dict_non_dict_json_raises(self):
        with pytest.raises(ValueError):
            coerce_config_value("[1, 2]", dict)

    def test_coerce_config_value_union_type_members(self):
        result = coerce_config_value("67", int | None)
        assert result == 67

        result = coerce_config_value(None, int | None)
        assert result is None

    def test_coerce_config_value_union_type_invalid_member_raises(self):
        with pytest.raises(ValueError):
            coerce_config_value("foobar", int | None)

    def test_coerce_config_value_enum_type_string(self):
        assert coerce_config_value("option_a", FakeEnum) == FakeEnum.OPTION_A

    def test_coerce_config_value_enum_type_enum_passthrough(self):
        assert coerce_config_value(FakeEnum.OPTION_B, FakeEnum) == FakeEnum.OPTION_B

    def test_coerce_config_value_enum_type_invalid_raises(self):
        with pytest.raises(ValueError):
            coerce_config_value("invalid_option", FakeEnum)

    def test_coerce_config_value_unsupported_target_type_raises(self):
        with pytest.raises(ValueError):
            coerce_config_value("anything", set)


class TestNormalizeConfigKey:
    """Tests for normalize_config_key."""

    def test_normalize_config_key(self):
        assert normalize_config_key("NOT-NORMAL") == "not_normal"
        assert normalize_config_key("already_normal") == "already_normal"


class TestProcessConfigUpdate:
    """Tests for process_config_update."""

    def test_process_config_update_uses_getter_and_validator(self):
        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=lambda obj, value: setattr(obj, "value", value),
            getter=lambda: "42",
            validator=lambda value: value + 1,
        )
        target = SimpleNamespace()

        result = process_config_update("key", descriptor)

        assert result == 43
        assert not hasattr(target, "value")

    def test_process_config_update_returns_none_without_value(self):
        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=lambda obj, value: setattr(obj, "value", value),
        )

        result = process_config_update("key", descriptor, value=None)

        assert result is None

    def test_process_config_update_raises_on_getter_failure(self):
        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=lambda obj, value: setattr(obj, "value", value),
            getter=lambda: (_ for _ in ()).throw(ValueError("boom")),
            fallback=99,
        )

        with pytest.raises(ValueError, match="Unable to retrieve value for key"):
            process_config_update("key", descriptor)

    def test_process_config_update_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Unrecognized config key"):
            process_config_update("missing", None)


class TestApplyConfigUpdate:
    """Tests for apply_config_update."""

    def test_apply_config_update_calls_setter(self):
        target = SimpleNamespace()
        descriptor = ConfigFieldDescriptor(
            target_type=str,
            setter=lambda obj, value: setattr(obj, "name", value),
        )

        result = apply_config_update(
            target_obj=target,
            key="name",
            descriptor=descriptor,
            value="agent",
        )

        assert result == "agent"
        assert target.name == "agent"

    def test_apply_config_update_wraps_setter_errors(self):
        target = SimpleNamespace()

        def setter(_obj, _value):
            raise TypeError("read-only")

        descriptor = ConfigFieldDescriptor(target_type=str, setter=setter)

        with pytest.raises(RuntimeError, match="Could not apply setter"):
            apply_config_update(
                target_obj=target,
                key="name",
                descriptor=descriptor,
                value="agent",
            )

    def test_apply_config_update_raises_value_error_for_invalid_value(self):
        target = SimpleNamespace()
        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=lambda obj, value: setattr(obj, "value", value),
        )

        with pytest.raises(ValueError, match="Invalid value for key 'name'"):
            apply_config_update(
                target_obj=target,
                key="name",
                descriptor=descriptor,
                value="not-a-number",
            )

    def test_apply_config_update_uses_fallback_when_not_raise(self):
        target = SimpleNamespace(value=None)
        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=lambda obj, value: setattr(obj, "value", value),
            getter=lambda: "not-a-number",
            fallback=99,
            should_raise=False,
        )

        result = apply_config_update(target_obj=target, key="value", descriptor=descriptor)

        assert result == 99
        assert target.value == 99

    def test_apply_config_update_uses_fallback_when_setter_fails_and_not_raise(self):
        target = SimpleNamespace(value=None)

        def setter(_obj, _value):
            raise RuntimeError("write failed")

        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=setter,
            getter=lambda: "7",
            fallback=77,
            should_raise=False,
        )

        result = apply_config_update(target_obj=target, key="value", descriptor=descriptor)

        assert result == 77
        assert target.value is None


class TestApplyConfigMap:
    """Tests for apply_config_map."""

    def test_apply_config_map_applies_successful_updates(self):
        target = SimpleNamespace(first=None, second=None)

        config_field_map = {
            "first": ConfigFieldDescriptor(
                target_type=str,
                setter=lambda obj, value: setattr(obj, "first", value),
                getter=lambda: "alpha",
            ),
            "second": ConfigFieldDescriptor(
                target_type=int,
                setter=lambda obj, value: setattr(obj, "second", value),
                getter=lambda: "2",
            ),
        }

        apply_config_map(target, config_field_map)

        assert target.first == "alpha"
        assert target.second == 2

    def test_apply_config_map_raises_value_error_without_fallback(self):
        target = SimpleNamespace(value=None)

        config_field_map = {
            "value": ConfigFieldDescriptor(
                target_type=int,
                setter=lambda obj, value: setattr(obj, "value", value),
                getter=lambda: "not-a-number",
            ),
        }

        with pytest.raises(ValueError, match="Invalid value for key 'value'"):
            apply_config_map(target, config_field_map)

    def test_apply_config_map_raises_runtime_error_without_fallback(self):
        target = SimpleNamespace(value=None)

        config_field_map = {
            "value": ConfigFieldDescriptor(
                target_type=str,
                getter=lambda: "alpha",
                setter=lambda _obj, _value: (_ for _ in ()).throw(RuntimeError("bad")),
            ),
        }

        with pytest.raises(RuntimeError, match="bad"):
            apply_config_map(target, config_field_map)

    def test_apply_config_map_uses_fallback_on_error(self):
        target = SimpleNamespace(value=None)

        config_field_map = {
            "value": ConfigFieldDescriptor(
                target_type=int,
                getter=lambda: "not-a-number",
                setter=lambda obj, value: setattr(obj, "value", value),
                fallback=99,
                should_raise=False,
            ),
        }

        apply_config_map(target, config_field_map)

        assert target.value == 99
