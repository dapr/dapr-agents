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

"""Integration tests for 07-data-agent-mcp-chainlit example."""

import pytest


@pytest.mark.integration
class TestDataAgentChainlitQuickstart:
    """Integration tests for 07-data-agent-mcp-chainlit example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir):
        """Setup test environment."""

    def test_data_agent_chainlit(self, dapr_runtime):  # noqa: ARG002
        pytest.skip(
            "Skipping 07-data-agent-mcp-chainlit test because it requires a browser and chainlit to be installed."
        )
