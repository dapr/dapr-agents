"""Integration tests for 06-document-agent-chainlit quickstart."""
import pytest


@pytest.mark.integration
class TestDocumentAgentChainlitQuickstart:
    """Integration tests for 06-document-agent-chainlit quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir):
        """Setup test environment."""

    def test_document_agent_chainlit(self, dapr_runtime):  # noqa: ARG002
        pytest.skip(
            "Skipping 06_document_agent_chainlit.py test because it requires a browser and chainlit to be installed."
        )
