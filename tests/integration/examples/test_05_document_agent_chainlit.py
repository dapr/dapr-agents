"""Integration tests for 05-document-agent-chainlit example."""

import pytest


@pytest.mark.integration
class TestDocumentAgentChainlitQuickstart:
    """Integration tests for 05-document-agent-chainlit example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir):
        """Setup test environment."""

    def test_document_agent_chainlit(self, dapr_runtime):  # noqa: ARG002
        pytest.skip(
            "Skipping 05-document-agent-chainlit test because it requires a browser and chainlit to be installed."
        )
