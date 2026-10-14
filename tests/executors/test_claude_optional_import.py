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

"""``dapr_agents`` must import without the optional ``claude`` extra.

Each check runs in a fresh interpreter where ``claude_agent_sdk`` is made
unimportable, so the result does not depend on what this process loaded.
"""

import subprocess
import sys
import textwrap

import pytest

_BLOCK_SDK = "import sys; sys.modules['claude_agent_sdk'] = None\n"


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_SDK + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_package_imports_without_sdk():
    result = _run(
        """
        import dapr_agents
        from dapr_agents import *  # noqa: F403
        from dapr_agents.agents.executors import *  # noqa: F403
        from dapr_agents.agents.durable import DurableAgent
        from dapr_agents.observability.wrappers.executor import (
            ExecutorObserverWrapper,
        )
        from dapr_agents import ClaudeAgentExecutorConfig, DaprSessionStore
        ClaudeAgentExecutorConfig(model="m")
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize("module", ["dapr_agents", "dapr_agents.agents.executors"])
def test_executor_access_raises_install_hint(module):
    result = _run(
        f"""
        import importlib
        mod = importlib.import_module({module!r})
        try:
            mod.ClaudeAgentExecutor
        except ImportError as exc:
            print(exc)
        """
    )
    assert result.returncode == 0, result.stderr
    assert 'pip install "dapr-agents[claude]"' in result.stdout


def test_unknown_attribute_still_raises_attribute_error():
    import dapr_agents.agents.executors as executors

    with pytest.raises(AttributeError):
        executors.NotAThing  # noqa: B018
