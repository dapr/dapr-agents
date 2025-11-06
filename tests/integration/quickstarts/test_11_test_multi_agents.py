"""Integration tests for 11-test-multi-agents quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestMultiAgentsQuickstart:
    """Integration tests for 11-test-multi-agents quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "11-test-multi-agents"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_multi_agent_system(self, dapr_runtime):  # noqa: ARG002
        """Test multi-agent system (requires dapr run -f dapr-llm.yaml).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        # Note: This quickstart requires multiple services to be running
        # For integration tests, we'll test one of the services
        # The full system requires separate services for each agent

        # Test the client service which connects to the agents
        client_script = self.quickstart_dir / "services" / "client" / "http_client.py"
        if client_script.exists():
            result = run_quickstart_script(
                client_script,
                cwd=self.quickstart_dir,
                env=self.env,
                timeout=60,
                use_dapr=True,
                app_id="multi-agent-client",
            )

            assert result.returncode == 0, (
                f"Quickstart failed with return code {result.returncode}.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
