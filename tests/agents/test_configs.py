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

from dapr_agents.agents.configs import (
    validate_max_iterations,
    validate_positive_int,
    validate_integral,
)


def test_validate_max_iterations_aliases_validate_positive_int():
    """Test that validate_max_iterations is preserved as an alias for validate_positive_int."""
    assert validate_max_iterations is validate_positive_int
    assert validate_max_iterations(1) == 1
    with pytest.raises(ValueError):
        validate_max_iterations(0)


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
    assert validate_integral(value) == expected


@pytest.mark.parametrize("value", [True, False, 10.5, "10.5", "abc"])
def test_validate_integral_rejects_non_integral_values(value):
    with pytest.raises(ValueError):
        validate_integral(value)
