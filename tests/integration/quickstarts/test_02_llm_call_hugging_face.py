"""Integration tests for 02_llm_call_hugging_face quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestLLMCallHuggingFaceQuickstart:
    """Integration tests for 02_llm_call_hugging_face quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, huggingface_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "02_llm_call_hugging_face"
        self.env = {"HUGGINGFACE_API_KEY": huggingface_api_key}

    def test_text_completion(self):
        """Test text completion example (text_completion.py)."""
        script = self.quickstart_dir / "text_completion.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=120,  # HuggingFace models can be slow
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert "Response:" in result.stdout or len(result.stdout) > 0
