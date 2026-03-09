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

"""Integration tests for 01-llm-call-nvidia example."""

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_script


@pytest.mark.integration
class TestLLMCallNvidiaQuickstart:
    """Integration tests for 01-llm-call-nvidia example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir, nvidia_api_key):
        """Setup test environment."""
        self.quickstart_dir = examples_dir / "01-llm-call-nvidia"
        self.env = {"NVIDIA_API_KEY": nvidia_api_key}

    def test_text_completion(self):
        """Test text completion example (text_completion.py)."""
        script = self.quickstart_dir / "text_completion.py"
        result = run_quickstart_or_examples_script(
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
