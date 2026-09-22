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

"""Tests for ToolHelper.infer_func_schema."""

from dapr_agents.tool.utils.tool import ToolHelper


def sample_func(a: int, b: str = "x"):
    """A sample function with parameters."""
    return a


def test_infer_func_schema_uses_provided_name():
    model = ToolHelper.infer_func_schema(sample_func, name="MyCustomModel")
    assert model.__name__ == "MyCustomModel"


def test_infer_func_schema_default_name_not_last_param():
    model = ToolHelper.infer_func_schema(sample_func)
    assert model.__name__ == "sample_funcModel"


def test_infer_func_schema_fields_still_correct():
    model = ToolHelper.infer_func_schema(sample_func)
    assert set(model.model_fields) == {"a", "b"}
