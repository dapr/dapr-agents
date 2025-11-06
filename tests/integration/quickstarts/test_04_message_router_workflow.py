"""Integration tests for 04-message-router-workflow quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestMessageRouterWorkflowQuickstart:
    """Integration tests for 04-message-router-workflow quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "04-message-router-workflow"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_message_router_workflow(self, dapr_runtime):  # noqa: ARG002
        """Test message router workflow (app.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        # Note: This quickstart requires two processes:
        # 1. The workflow app (app.py)
        # 2. The message client (message_client.py)
        # For integration tests, we'll just test that the app can start
        script = self.quickstart_dir / "app.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=30,  # Just test startup, not full execution
            use_dapr=True,
            app_id="message-router-app",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
