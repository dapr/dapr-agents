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
