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

"""Tests for legacy model helper imports from workflow core utilities."""

from dataclasses import dataclass

from dapr_agents.utils.models import (
    is_pydantic_model,
    is_supported_model as models_is_supported_model,
    is_supported_model_instance as models_is_supported_model_instance,
    is_valid_routable_model as models_is_valid_routable_model,
)
from dapr_agents.workflow.utils.core import (
    is_pydantic_model as workflow_is_pydantic_model,
    is_supported_model,
    is_supported_model_instance,
    is_valid_routable_model,
)


@dataclass
class RoutableMessage:
    value: str


def test_core_preserves_legacy_model_helper_imports():
    assert workflow_is_pydantic_model is is_pydantic_model
    assert is_supported_model is models_is_supported_model
    assert is_supported_model_instance is models_is_supported_model_instance
    assert is_valid_routable_model is models_is_valid_routable_model
    assert is_supported_model(dict)
    assert is_supported_model_instance({})
    assert is_valid_routable_model(RoutableMessage)
    assert not is_valid_routable_model(dict)
