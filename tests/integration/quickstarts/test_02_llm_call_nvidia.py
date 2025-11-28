"""Integration tests for 02-llm-call-nvidia quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestLLMCallNvidiaQuickstart:
    """Integration tests for 02-llm-call-nvidia quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, nvidia_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "02-llm-call-nvidia"
        self.env = {"NVIDIA_API_KEY": nvidia_api_key}

    def test_text_completion(self):
        """Test text completion example (text_completion.py)."""
        script = self.quickstart_dir / "text_completion.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=60,
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert "Response:" in result.stdout or len(result.stdout) > 0
