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

import pytest

from dapr_agents.agents.configs import validate_max_iterations, _validate_integral


# TODO: Remove this compatibility test when ConfigFieldDescriptor is removed.
def test_deprecated_config_field_descriptor_remains_available() -> None:
    """Test that the positional ``ConfigFieldDescriptor``
    remains importable from the original import path but raises a deprecation warning."""
    with pytest.warns(DeprecationWarning, match="ConfigFieldDescriptor.*deprecated"):
        from dapr_agents.agents.configs import ConfigFieldDescriptor

    def setter(_obj: object, _value: object) -> None:
        return None

    def validator(value: str) -> str:
        return value.strip()

    descriptor = ConfigFieldDescriptor(str, setter, True, validator, True, True)

    assert descriptor.target_type is str
    assert descriptor.setter is setter
    assert descriptor.sensitive is True
    assert descriptor.validator is validator
    assert descriptor.rebuilds_prompt is True
    assert descriptor.triggers_otel_reload is True


def test_deprecated_config_field_descriptor_defaults() -> None:
    """Test that the positional ``ConfigFieldDescriptor`` defaults are preserved."""
    with pytest.warns(DeprecationWarning, match="ConfigFieldDescriptor.*deprecated"):
        from dapr_agents.agents.configs import ConfigFieldDescriptor

    descriptor = ConfigFieldDescriptor(int, None)

    assert descriptor.target_type is int
    assert descriptor.setter is None
    assert descriptor.sensitive is False
    assert descriptor.validator is None
    assert descriptor.rebuilds_prompt is False
    assert descriptor.triggers_otel_reload is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (10, 10),
    ],
)
def test_validate_max_iterations_accepts_positive_integers(value, expected):
    assert validate_max_iterations(value) == expected


@pytest.mark.parametrize(
    ("value", "error_message"),
    [
        (0, "max_iterations must be >= 1"),
        (-1, "max_iterations must be >= 1"),
        (0.99, "max_iterations must be an integer"),
    ],
)
def test_validate_max_iterations_rejects_invalid_values(value, error_message):
    with pytest.raises(ValueError, match=error_message):
        validate_max_iterations(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (-1, -1),
        ("1", "1"),
        ("-1", "-1"),
        (" 10 ", " 10 "),
        ("9007199254740993", "9007199254740993"),
        ("9007199254740993.0", "9007199254740993.0"),
        (10.0, 10.0),
        ("10.0", "10.0"),
    ],
)
def test_validate_integral_accepts_integral_values(value, expected):
    assert _validate_integral(value) == expected


@pytest.mark.parametrize("value", [True, False, 10.5, "10.5", "abc"])
def test_validate_integral_rejects_non_integral_values(value):
    with pytest.raises(ValueError):
        _validate_integral(value)
