"""Integration tests for 02_llm_call_dapr quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestLLMCallDaprQuickstart:
    """Integration tests for 02_llm_call_dapr quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "02_llm_call_dapr"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_text_completion(self, dapr_runtime):  # noqa: ARG002
        """Test text completion using DaprChatClient (text_completion.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "text_completion.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=60,
            use_dapr=True,
            app_id="dapr-llm-test",
        )

        # Quickstart should run successfully without errors
        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # Verify it produced some output
        assert "Response:" in result.stdout or "response" in result.stdout.lower()
