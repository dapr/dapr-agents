#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Integration tests for 01-llm-call-dapr example."""

import os

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_script


@pytest.mark.integration
class TestLLMCallDaprQuickstart:
    """Integration tests for 01-llm-call-dapr example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir, openai_api_key, is_ollama):
        """Setup test environment."""
        self.quickstart_dir = examples_dir / "01-llm-call-dapr"
        self.env = {
            "OPENAI_API_KEY": openai_api_key,
            "OPENAI_MODEL": os.environ.get("OLLAMA_MODEL", "gpt-4-turbo")
            if is_ollama
            else "gpt-4-turbo",
            "OPENAI_BASE_URL": os.environ.get("OLLAMA_ENDPOINT", "")
            if is_ollama
            else "",
        }

    def test_text_completion(self, dapr_runtime):  # noqa: ARG002
        """Test text completion using DaprChatClient (text_completion.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "text_completion.py"
        result = run_quickstart_or_examples_script(
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
