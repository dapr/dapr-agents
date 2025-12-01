"""Integration tests for 02-llm-call-elevenlabs quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestLLMCallElevenLabsQuickstart:
    """Integration tests for 02-llm-call-elevenlabs quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, elevenlabs_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "02-llm-call-elevenlabs"
        self.env = {"ELEVENLABS_API_KEY": elevenlabs_api_key}

    def test_text_to_speech(self):
        """Test text to speech example (text_to_speech.py)."""
        script = self.quickstart_dir / "text_to_speech.py"
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
        assert "Audio saved" in result.stdout or "saved" in result.stdout.lower()
