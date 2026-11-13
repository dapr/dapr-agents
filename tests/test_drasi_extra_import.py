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

"""Verify importing dapr_agents and extensions outside the repository root.

Regression test for issue #938:
Installing the drasi extra must not break importing dapr_agents from directories
outside the repository root (e.g. temporary directories, quickstarts, examples).
"""

import subprocess
import sys
import tempfile
import textwrap


def _run_outside_repo(code: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp_dir:
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )


def test_dapr_agents_import_outside_repo_root():
    result = _run_outside_repo(
        """
        import dapr_agents
        from dapr_agents import DurableAgent, AgentRunner

        assert dapr_agents.__file__ is not None, "dapr_agents must not be an empty namespace"
        assert DurableAgent is not None
        assert AgentRunner is not None
        print("ok")
        """
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert result.stdout.strip() == "ok"


def test_drasi_extension_import_outside_repo_root():
    result = _run_outside_repo(
        """
        from dapr_agents import DurableAgent
        import dapr_agents.ext.drasi
        from dapr_agents.ext.drasi import drasi_trigger, DrasiChangeEvent

        assert DurableAgent is not None
        assert drasi_trigger is not None
        assert DrasiChangeEvent is not None
        print("ok")
        """
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert result.stdout.strip() == "ok"
