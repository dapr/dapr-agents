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

from dapr_agents.agents.configs import validate_max_iterations, validate_positive_int


def test_validate_max_iterations_aliases_validate_positive_int():
    """Test that validate_max_iterations is preserved as an alias for validate_positive_int."""
    assert validate_max_iterations is validate_positive_int
    assert validate_max_iterations(1) == 1
    with pytest.raises(ValueError):
        validate_max_iterations(0)
