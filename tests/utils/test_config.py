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
import itertools
import logging
from types import SimpleNamespace

import pytest

from dapr_agents.utils.config import (
    ConfigFieldDescriptor,
    apply_config_map,
    apply_config_update,
    coerce_config_value,
    get_config_value,
    _normalize_config_key,
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

    def test_coerce_config_value_int_from_str(self):
        assert coerce_config_value("42", int) == 42

    def test_coerce_config_value_int_from_float_str(self):
        assert coerce_config_value("10.0", int) == 10

    def test_coerce_config_value_int_from_float(self):
        assert coerce_config_value(10.0, int) == 10

    @pytest.mark.parametrize("value", ["9007199254740993", "9007199254740993.0"])
    def test_coerce_config_value_int_from_large_integral_str(self, value):
        assert coerce_config_value(value, int) == 9007199254740993

    @pytest.mark.parametrize("value", ["10.5", "NaN", "Infinity"])
    def test_coerce_config_value_int_rejects_non_integral_values(self, value):
        with pytest.raises(ValueError):
            coerce_config_value(value, int)

    def test_coerce_config_value_int_already_int(self):
        assert coerce_config_value(7, int) == 7

    def test_coerce_config_value_bool_to_int_raises(self):
        with pytest.raises(ValueError):
            coerce_config_value(True, int)

    def test_coerce_config_value_int_invalid_raises(self):
        with pytest.raises((ValueError, TypeError)):
            coerce_config_value("not_a_number", int)

    def test_coerce_config_value_float_from_str(self):
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

    def test_coerce_config_value_list_wraps_single_str(self):
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

    def test_coerce_config_value_enum_type_str(self):
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
    """Tests for _normalize_config_key."""

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("NOT_NORMALIZED", "not_normalized"),
            ("not-normalized", "not_normalized"),
            ("already_normalized", "already_normalized"),
        ],
    )
    def test_normalize_config_key_normalizes_supported_conventions(self, key, expected):
        assert _normalize_config_key(key) == expected

    # TODO: Remove when deprecated key normalization support is removed
    @pytest.mark.parametrize(
        "key",
        ["Not-Normalized", "Not_normalized", "NOT-NORMALIZED", "Not_Normalized"],
    )
    def test_normalize_config_key_warns_but_normalizes_unsupported_conventions(
        self, key
    ):
        with pytest.warns(DeprecationWarning, match="deprecated naming convention"):
            assert _normalize_config_key(key) == "not_normalized"


class TestGetConfigValue:
    """Tests for get_config_value."""

    @pytest.mark.parametrize("key", ["TEST_KEY", "test_key", "test-key"])
    def test_get_config_value_finds_screaming_snake_case_key(self, key):
        assert get_config_value({"TEST_KEY": "val"}, key) == "val"

    @pytest.mark.parametrize("key", ["TEST_KEY", "test_key", "test-key"])
    def test_get_config_value_finds_snake_case_key(self, key):
        assert get_config_value({"test_key": "val"}, key) == "val"

    @pytest.mark.parametrize("key", ["TEST_KEY", "test_key", "test-key"])
    def test_get_config_value_finds_kebab_case_key(self, key):
        assert get_config_value({"test-key": "val"}, key) == "val"

    @pytest.mark.parametrize("key", ["TEST_KEY", "test_key", "test-key"])
    def test_get_config_value_prefers_screaming_snake_case_over_snake_case(self, key):
        mapping = {"TEST_KEY": "val1", "test_key": "val2"}
        assert get_config_value(mapping, key) == "val1"

    @pytest.mark.parametrize("key", ["TEST_KEY", "test_key", "test-key"])
    def test_get_config_value_prefers_snake_case_over_kebab_case(self, key):
        mapping = {"test_key": "val1", "test-key": "val2"}
        assert get_config_value(mapping, key) == "val1"

    @pytest.mark.parametrize("key", ["TEST_KEY", "test_key", "test-key"])
    def test_get_config_value_full_precedence(self, key):
        mapping = {"TEST_KEY": "val1", "test_key": "val2", "test-key": "val3"}
        assert get_config_value(mapping, key) == "val1"

    @pytest.mark.parametrize(
        "mapping",
        [
            {k: v for d in perm for k, v in d.items()}
            for perm in itertools.permutations(
                [{"TEST_KEY": "val1"}, {"test_key": "val2"}, {"test-key": "val3"}]
            )
        ],
    )
    @pytest.mark.parametrize("key", ["TEST_KEY", "test_key", "test-key"])
    def test_get_config_value_random_key_order(self, key, mapping):
        """Test that get_config_value does not depend on the order of keys in the mapping."""
        assert get_config_value(mapping, key) == "val1"

    @pytest.mark.parametrize("key", ["TEST_KEY", "test_key", "test-key"])
    def test_get_config_value_ignores_empty_values(self, key):
        mapping = {"TEST_KEY": "", "test_key": "", "test-key": "val"}
        assert get_config_value(mapping, key) == "val"

    @pytest.mark.parametrize("key", ["TEST_KEY", "test_key", "test-key"])
    def test_get_config_value_ignores_none_values(self, key):
        mapping = {"TEST_KEY": None, "test_key": "val1", "test-key": "val2"}
        assert get_config_value(mapping, key) == "val1"

    @pytest.mark.parametrize("key", ["TEST_KEY", "test_key", "test-key"])
    def test_get_config_value_ignores_unsupported_mapping_keys(self, key):
        mapping = {"TEST-KEY": "val1", "Test_Key": "val2", "test_key": "val3"}
        assert get_config_value(mapping, key) == "val3"

    @pytest.mark.parametrize("key", ["Test-Key", "Test_key", "TEST-KEY", "Test_Key"])
    def test_get_config_value_rejects_unsupported_requested_keys(self, key):
        assert get_config_value({"test_key": "val"}, key) is None

    @pytest.mark.parametrize("key", ["TEST_KEY", "test_key", "test-key"])
    def test_get_config_value_defaults_to_none(self, key):
        mapping = {"TEST_KEY": None, "test_key": None, "test-key": None}
        assert get_config_value(mapping, key) is None


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

    def test_process_config_update_applies_pre_validator_before_coercion(self):
        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=lambda obj, value: setattr(obj, "value", value),
            pre_validator=lambda value: f"{value}0",
        )

        result = process_config_update("key", descriptor, value="4")

        assert result == 40

    def test_process_config_update_raises_when_pre_validator_fails(self):
        def pre_validator(_value):
            raise ValueError("invalid raw value")

        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=lambda obj, value: setattr(obj, "value", value),
            pre_validator=pre_validator,
        )

        with pytest.raises(ValueError, match="Invalid value for key"):
            process_config_update("key", descriptor, value="4")

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

    def test_process_config_update_does_not_expose_sensitive_value_on_failure(self):
        secret = {"api_key": "secret-key"}
        descriptor = ConfigFieldDescriptor(
            target_type=bool,
            setter=lambda obj, value: setattr(obj, "api_key", value),
            sensitive=True,
        )

        with pytest.raises(ValueError) as exc_info:
            process_config_update("api_key", descriptor, value=secret)

        assert secret["api_key"] not in str(exc_info.value)


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

        with pytest.raises(ValueError, match="Invalid value for key"):
            apply_config_update(
                target_obj=target,
                key="name",
                descriptor=descriptor,
                value="not-a-number",
            )

    def test_apply_config_update_uses_fallback_without_raise_on_error(self):
        target = SimpleNamespace(value=None)
        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=lambda obj, value: setattr(obj, "value", value),
            getter=lambda: "not-a-number",
            fallback=99,
            raise_on_error=False,
        )

        result = apply_config_update(
            target_obj=target, key="value", descriptor=descriptor
        )

        assert result == 99
        assert target.value == 99

    def test_apply_config_update_uses_fallback_when_setter_fails_without_raise_on_error(
        self,
    ):
        target = SimpleNamespace(value=None)

        def setter(_obj, _value):
            raise RuntimeError("write failed")

        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=setter,
            getter=lambda: "7",
            fallback=77,
            raise_on_error=False,
        )

        result = apply_config_update(
            target_obj=target, key="value", descriptor=descriptor
        )

        assert result is None
        assert target.value is None

    def test_apply_config_update_noop_and_warns_when_setter_fails_without_fallback(
        self, caplog
    ):
        target = SimpleNamespace(value=None)

        def setter(_obj, _value):
            raise RuntimeError("write failed")

        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=setter,
            fallback=None,
            raise_on_error=False,
        )

        with caplog.at_level(logging.WARNING):
            result = apply_config_update(
                target_obj=target, key="value", descriptor=descriptor, value=7
            )

        assert result is None
        assert target.value is None
        assert "Ignoring failed config update for key" in caplog.text

    def test_apply_config_update_does_not_log_sensitive_value_on_processing_failure(
        self, caplog
    ):
        secret = "secret-key"
        descriptor = ConfigFieldDescriptor(
            target_type=dict,
            setter=lambda obj, value: setattr(obj, "config", value),
            raise_on_error=False,
            sensitive=True,
        )

        with caplog.at_level(logging.WARNING):
            apply_config_update(
                target_obj=SimpleNamespace(),
                key="config",
                descriptor=descriptor,
                value=secret,
            )

        assert secret not in caplog.text

    def test_apply_config_update_does_not_log_sensitive_value_on_setter_failure(
        self, caplog
    ):
        def setter(_obj, value):
            raise ValueError(f"setter rejected {value}")

        secret = "secret-key"
        descriptor = ConfigFieldDescriptor(
            target_type=dict,
            setter=setter,
            raise_on_error=False,
            sensitive=True,
        )

        with caplog.at_level(logging.WARNING):
            apply_config_update(
                target_obj=SimpleNamespace(),
                key="config",
                descriptor=descriptor,
                value=secret,
            )

        assert secret not in caplog.text

    def test_apply_config_update_does_not_log_sensitive_value_on_fallback_failure(
        self, caplog
    ):
        def setter(_obj, value):
            raise ValueError(f"setter rejected {value}")

        secret = "secret-key"
        descriptor = ConfigFieldDescriptor(
            target_type=dict,
            setter=setter,
            fallback="default-key",
            raise_on_error=False,
            sensitive=True,
        )

        with caplog.at_level(logging.DEBUG):
            apply_config_update(
                target_obj=SimpleNamespace(),
                key="config",
                descriptor=descriptor,
                value=secret,
            )

        assert secret not in caplog.text

    def test_apply_config_update_noop_and_warns_when_fallback_setter_fails(
        self, caplog
    ):
        def setter(_obj, _value):
            raise RuntimeError("write failed")

        target = SimpleNamespace(value=None)
        descriptor = ConfigFieldDescriptor(
            target_type=int,
            setter=setter,
            fallback=77,
            raise_on_error=False,
        )

        with caplog.at_level(logging.WARNING):
            result = apply_config_update(
                target_obj=target, key="value", descriptor=descriptor, value=7
            )

        assert result is None
        assert target.value is None
        assert "Failed to apply fallback for key" in caplog.text


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
            "third": ConfigFieldDescriptor(
                target_type=bool,
                setter=lambda obj, value: setattr(obj, "third", value),
                getter=lambda: "true",
            ),
        }

        apply_config_map(target, config_field_map)

        assert target.first == "alpha"
        assert target.second == 2
        assert target.third is True

    def test_apply_config_map_raises_value_error_without_fallback(self):
        target = SimpleNamespace(value=None)

        config_field_map = {
            "value": ConfigFieldDescriptor(
                target_type=int,
                setter=lambda obj, value: setattr(obj, "value", value),
                getter=lambda: "not-a-number",
            ),
        }

        with pytest.raises(ValueError, match="Invalid value for key"):
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

        with pytest.raises(RuntimeError, match="Could not apply setter for key"):
            apply_config_map(target, config_field_map)

    def test_apply_config_map_uses_fallback_on_error(self):
        target = SimpleNamespace(value=None)

        config_field_map = {
            "value": ConfigFieldDescriptor(
                target_type=int,
                getter=lambda: "not-a-number",
                setter=lambda obj, value: setattr(obj, "value", value),
                fallback=99,
                raise_on_error=False,
            ),
        }

        apply_config_map(target, config_field_map)

        assert target.value == 99
