"""Integration tests for 02-llm-call-hugging-face quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestLLMCallHuggingFaceQuickstart:
    """Integration tests for 02-llm-call-hugging-face quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, huggingface_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "02-llm-call-hugging-face"
        self.env = {"HUGGINGFACE_API_KEY": huggingface_api_key}

    def test_text_completion(self):
        """Test text completion example (text_completion.py)."""
        script = self.quickstart_dir / "text_completion.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # HuggingFace models can be slow
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert "Response:" in result.stdout or len(result.stdout) > 0

    def test_text_completion_stream(self):
        """Test text completion stream example (text_completion_stream.py)."""
        script = self.quickstart_dir / "text_completion_stream.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # HuggingFace models can be slow
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_text_completion_stream_with_tools(self):
        """Test text completion stream with tools example (text_completion_stream_with_tools.py)."""
        script = self.quickstart_dir / "text_completion_stream_with_tools.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # HuggingFace models can be slow
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_text_completion_with_tools(self):
        """Test text completion with tools example (text_completion_with_tools.py)."""
        script = self.quickstart_dir / "text_completion_with_tools.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # HuggingFace models can be slow
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_structured_completion(self):
        """Test structured completion example (structured_completion.py)."""
        script = self.quickstart_dir / "structured_completion.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # HuggingFace models can be slow
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0
