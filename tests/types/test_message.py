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
from pydantic import ValidationError

from dapr_agents.types.message import FunctionCall


def test_function_call_accepts_valid_json_string():
    call = FunctionCall(name="get_weather", arguments='{"city": "Seattle"}')
    assert call.arguments == '{"city": "Seattle"}'
    assert call.arguments_dict == {"city": "Seattle"}


def test_function_call_serializes_dict_arguments():
    call = FunctionCall(name="get_weather", arguments={"city": "Seattle"})
    assert call.arguments == '{"city": "Seattle"}'
    assert call.arguments_dict == {"city": "Seattle"}


def test_function_call_invalid_json_string_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        FunctionCall(name="get_weather", arguments="{not valid json")
    assert "Invalid JSON format" in str(exc_info.value)


def test_function_call_non_serializable_dict_raises_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        FunctionCall(name="get_weather", arguments={"key": object()})
    assert "Invalid data type in dictionary" in str(exc_info.value)


def test_function_call_unsupported_type_raises_type_error():
    with pytest.raises(TypeError):
        FunctionCall(name="get_weather", arguments=123)
