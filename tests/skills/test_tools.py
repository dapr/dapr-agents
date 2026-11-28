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

from dapr_agents.skills._tools import build_skill_tools


def test_skill_tools_have_stable_names_and_argument_schemas() -> None:
    load_tool, read_tool = build_skill_tools()

    assert load_tool.name == "load_skill"
    assert read_tool.name == "read_skill_resource"
    assert set(load_tool.args_model.model_json_schema()["properties"]) == {"name"}
    assert set(read_tool.args_model.model_json_schema()["properties"]) == {
        "name",
        "path",
    }
