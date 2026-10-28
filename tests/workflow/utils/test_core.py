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

"""Tests for workflow core utilities."""

from dataclasses import dataclass

from pydantic import BaseModel

from dapr_agents.workflow.utils.core import is_valid_routable_model


@dataclass
class RoutableMessage:
    value: str


class PydanticRoutableMessage(BaseModel):
    value: str


def test_core_preserves_legacy_model_helper_imports():
    import dapr_agents.utils.models as models
    import dapr_agents.workflow.utils.core as core

    assert core.is_pydantic_model is models.is_pydantic_model
    assert core.is_supported_model is models.is_supported_model
    assert core.is_supported_model_instance is models.is_supported_model_instance


def test_is_valid_routable_model():
    assert is_valid_routable_model(RoutableMessage)
    assert is_valid_routable_model(PydanticRoutableMessage)
    assert not is_valid_routable_model(dict)
