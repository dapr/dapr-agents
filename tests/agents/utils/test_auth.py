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

"""Tests for the legacy authentication module import shim."""

from dapr_agents.agents.utils.auth import construct_auth_headers
from dapr_agents.agents.utils.headers import construct_auth_headers as headers_auth


def test_construct_auth_headers_is_preserved():
    assert construct_auth_headers is headers_auth
