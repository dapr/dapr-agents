"""Integration tests for 07-data-agent-mcp-chainlit example."""

import pytest


@pytest.mark.integration
class TestDataAgentChainlitQuickstart:
    @pytest.fixture(autouse=True)
    def setup(self, examples_dir):
        """Setup test environment."""

    def test_data_agent_chainlit(self, dapr_runtime):  # noqa: ARG002
        pytest.skip(
            "Skipping 08_data_agent_chainlit.py test because it requires a browser and chainlit to be installed."
        )
